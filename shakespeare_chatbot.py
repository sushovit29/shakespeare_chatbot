# shakespeare_chatbot.py

import random
import re
import numpy as np # For dot product and norm
import sys # Added for platform check
import asyncio # Added for event loop policy

# Fix for RuntimeError: no running event loop on Windows with asyncio
if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Attempt to import necessary libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("BardBot Warning: 'requests' library not found. Corpus download functionality will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline 
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer # Added for quantization
    import torch 
    import torch.quantization # Added for quantization
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("BardBot Warning: 'transformers', 'sentence-transformers', or 'torch' library not found. RAG functionality (and quantization) will be disabled.")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit library not found. Please install it using 'pip install streamlit' to run the UI.")

# --- Constants and Configuration ---
BOT_NAME = "BardBot" 
CORPUS_URL = "https://www.gutenberg.org/files/100/100-0.txt"
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
QA_MODEL_NAME = 'distilbert-base-uncased-distilled-squad'
CHUNK_SIZE = 8 
CHUNK_OVERLAP = 2 

# --- 1. Knowledge Base Module (Corpus Focused) ---
class KnowledgeBase:
    def __init__(self):
        # Hardcoded knowledge (quotes, summaries, character info) is now removed.
        # The chatbot will rely on the corpus and RAG.
        self.all_corpus_lines = []
        self.corpus_chunks = []
        self.chunk_embeddings = None
        self.corpus_loaded_successfully = False
        self.embeddings_generated = False
        self._load_and_process_corpus()
        if self.corpus_loaded_successfully:
            self._chunk_corpus()
            if TRANSFORMERS_AVAILABLE and self.corpus_chunks:
                self._build_document_embeddings()

    def _download_corpus(self, url):
        if not REQUESTS_AVAILABLE: return None
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"{BOT_NAME} Error (console): Failed to download corpus: {e}")
            return None

    def _preprocess_corpus(self, raw_text):
        if not raw_text: return []
        lines = raw_text.splitlines()
        start_patterns = ["*** START OF THIS PROJECT GUTENBERG EBOOK", "*** START OF THE PROJECT GUTENBERG EBOOK"]
        end_patterns = ["*** END OF THIS PROJECT GUTENBERG EBOOK", "*** END OF THE PROJECT GUTENBERG EBOOK"]
        start_index, end_index, start_marker_found = -1, len(lines), False
        for i, line in enumerate(lines):
            if any(p in line for p in start_patterns): start_index, start_marker_found = i + 1, True; continue
            if start_marker_found and any(p in line for p in end_patterns): end_index = i; break
        if not start_marker_found: start_index = 0
        content_lines = [line.strip() for line in lines[start_index:end_index] if line.strip()]
        return content_lines

    def _load_and_process_corpus(self):
        print(f"{BOT_NAME} (console): Attempting to load Shakespeare corpus...")
        raw_corpus_text = self._download_corpus(CORPUS_URL)
        if raw_corpus_text:
            self.all_corpus_lines = self._preprocess_corpus(raw_corpus_text)
            if self.all_corpus_lines:
                self.corpus_loaded_successfully = True
                print(f"{BOT_NAME} (console): Corpus loaded and processed. ({len(self.all_corpus_lines)} lines)")

    def _chunk_corpus(self):
        if not self.all_corpus_lines: return
        self.corpus_chunks = []
        for i in range(0, len(self.all_corpus_lines), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = self.all_corpus_lines[i : i + CHUNK_SIZE]
            if chunk: self.corpus_chunks.append(" ".join(chunk))
        print(f"{BOT_NAME} (console): Corpus chunked into {len(self.corpus_chunks)} chunks (Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")

    def _build_document_embeddings(self):
        if not self.corpus_chunks or not TRANSFORMERS_AVAILABLE: return
        try:
            print(f"{BOT_NAME} (console): Initializing sentence transformer model '{SENTENCE_MODEL_NAME}' for embeddings...")
            model = SentenceTransformer(SENTENCE_MODEL_NAME)
            print(f"{BOT_NAME} (console): Generating embeddings for {len(self.corpus_chunks)} chunks...")
            self.chunk_embeddings = model.encode(self.corpus_chunks, show_progress_bar=False) 
            self.embeddings_generated = True
            print(f"{BOT_NAME} (console): Document embeddings generated successfully.")
        except Exception as e:
            print(f"{BOT_NAME} Error (console): Failed to build document embeddings: {e}")
            self.embeddings_generated = False
            
    def find_lines_in_corpus(self, query_term, max_results=5):
        # This method remains useful for direct keyword search if needed,
        # separate from RAG's semantic search.
        if not self.corpus_loaded_successfully:
            return ["The corpus is not loaded, so I cannot search it at this time."]
        matches = [line for line in self.all_corpus_lines if query_term.lower() in line.lower()]
        return matches[:max_results]

# --- RAG System ---
class RAGSystem:
    def __init__(self):
        self.sentence_model = None
        self.qa_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"{BOT_NAME} (console): Initializing RAG sentence model '{SENTENCE_MODEL_NAME}' (not quantized)...")
                self.sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
                
                print(f"{BOT_NAME} (console): Initializing and quantizing RAG QA model '{QA_MODEL_NAME}'...")
                tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
                orig_qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)

                orig_qa_model.eval() 
                quantized_qa_model = torch.quantization.quantize_dynamic(
                    orig_qa_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print(f"{BOT_NAME} (console): QA model dynamically quantized.")
                qa_pipeline_device = -1 
                
                self.qa_pipeline = hf_pipeline("question-answering", model=quantized_qa_model, tokenizer=tokenizer, device=qa_pipeline_device)
                print(f"{BOT_NAME} (console): RAG models (QA model quantized) loaded successfully.")

            except Exception as e:
                print(f"{BOT_NAME} Error (console): Failed to load/quantize RAG models: {e}")
                self.sentence_model = None
                self.qa_pipeline = None
        else:
            print(f"{BOT_NAME} (console): RAG models cannot be loaded because 'transformers' library is missing.")

    def _cosine_similarity(self, vec1, vec2): 
        vec1 = np.asarray(vec1); vec2 = np.asarray(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1); norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def retrieve_context(self, query, chunks, chunk_embeddings, top_k=3): 
        if not self.sentence_model or chunk_embeddings is None or not chunks: return []
        query_embedding = self.sentence_model.encode([query])[0]
        similarities = [self._cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
        sorted_indices = np.argsort(similarities)[::-1]
        relevant_chunks = [chunks[i] for i in sorted_indices[:top_k] if similarities[i] > 0.3] 
        return relevant_chunks

    def generate_answer(self, question, context_list): 
        if not self.qa_pipeline or not context_list:
            return "I couldn't find enough information to answer that."
        full_context = " ".join(context_list)
        if not full_context.strip(): return "The retrieved context was empty."
        try:
            result = self.qa_pipeline(question=question, context=full_context)
            answer = result.get('answer')
            if answer and len(answer.strip()) > 1 : 
                return answer
            else:
                return "I found some context, but couldn't extract a specific answer from it."
        except Exception as e:
            print(f"{BOT_NAME} Error (console) during QA generation: {e}")
            return "I encountered an issue while trying to formulate the answer."

# --- Input Processor ---
class InputProcessor: 
    def process(self, user_input_str):
        processed_input = user_input_str.lower().strip()
        return re.sub(r'[^\w\s\'-]', '', processed_input) 

# --- Intent Recognizer ---
class IntentRecognizer: 
    def __init__(self):
        self.intent_keywords = {
            "greet": ["hello", "hi", "greetings", "good day", "hey"],
            "farewell": ["bye", "goodbye", "see you", "later", "exit"],
            # Removed: "get_quote", "about_play", "about_character" as they are now RAG based
            "search_corpus": ["find in corpus", "search corpus for", "look for text", "corpus search", "find text"],
            "ask_quote_attribution": ["who said", "who spoke the line", "who says"], 
            "ask_factual_question": [
                "who ", "what ", "when ", "where ", "why ", "how ", "explain ", 
                "which ", "in which ", "tell me why ", "tell me how ", "tell me what ",
                "tell me about ", "summary of ", "plot of " # Added general info queries here
            ],
            "thank": ["thank you", "thanks", "appreciate it"],
            "affirmative": ["yes", "yep", "yeah", "ok", "sure"],
            "negative": ["no", "nope", "not really"],
        }

    def _clean_text_for_matching(self, text):
        if not text: return ""
        return re.sub(r'[^\w\s]', '', text.lower())

    def recognize_intent(self, processed_input):
        # Priority 1: Quote Attribution
        quote_attr_patterns = [
            r"\b(?:who said|who says|who spoke the line)\s+['\"]([^'\"]+)['\"]", 
            r"\b(?:who said|who says|who spoke the line)\s+(.+)" 
        ]
        for pattern in quote_attr_patterns:
            quote_attr_match = re.search(pattern, processed_input)
            if quote_attr_match and quote_attr_match.group(1):
                # Avoid misclassifying "who is [character]" as quote attribution
                if not processed_input.startswith("who is "):
                    return {"intent": "ask_quote_attribution", "entities": {"dialogue_text": quote_attr_match.group(1).strip()}}

        # Priority 2: General Factual Questions (RAG candidates)
        # This will now catch queries like "who is hamlet", "tell me about macbeth", "what is the plot of romeo and juliet"
        for keyword in self.intent_keywords["ask_factual_question"]:
            is_part_of_attribution = any(processed_input.startswith(attr_kw) for attr_kw in self.intent_keywords["ask_quote_attribution"])
            if processed_input.startswith(keyword) and not is_part_of_attribution:
                return {"intent": "ask_factual_question", "entities": {"query": processed_input}}

        # Priority 3: Search Corpus (direct keyword search)
        for trigger in self.intent_keywords["search_corpus"]:
            if processed_input.startswith(trigger): 
                query_term = processed_input[len(trigger):].strip()
                if query_term.lower().startswith("for "): query_term = query_term[4:].strip()
                return {"intent": "search_corpus", "entities": {"query_term": query_term or None}}
        
        # Priority 4: Other keyword-based intents (greet, farewell, thank, etc.)
        for intent, keywords in self.intent_keywords.items():
            if intent in ["ask_quote_attribution", "ask_factual_question", "search_corpus"]:
                continue # Already handled
            
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", processed_input):
                    return {"intent": intent, "entities": {}} # No specific entities for these simple intents
        
        return {"intent": "unknown", "entities": {"query": processed_input}}

# --- Dialog Manager ---
class DialogManager: 
    def __init__(self, knowledge_base, rag_system_instance):
        self.knowledge_base = knowledge_base
        self.rag_system = rag_system_instance
        self.conversation_context = {} 
        # intent_recognizer instance is now passed in ShakespeareChatbot __init__
        self.intent_recognizer = None 


    def _clean_text_for_matching(self, text):
        if not text: return ""
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def manage_dialog(self, intent_data, processed_input):
        intent = intent_data["intent"]
        entities = intent_data["entities"]
        response_data = {"type": "general", "message": ""}

        if intent == "greet":
            response_data["message"] = random.choice(["Hello!", "Hi there! How can I help?"])
        elif intent == "farewell": 
            response_data["message"] = random.choice(["Goodbye!", "See you later!"])
            response_data["type"] = "end_conversation" 

        elif intent == "ask_quote_attribution":
            # This intent now solely relies on RAG as curated quotes are removed.
            # The question passed to RAG should be the original "who said..." query.
            dialogue_text_entity = entities.get("dialogue_text") # This might be the extracted quote
            query_for_rag = processed_input # Use the full original processed input for RAG

            print(f"{BOT_NAME} (console): Quote attribution for '{dialogue_text_entity}' (query: '{query_for_rag}') using RAG.")
            if self.rag_system and self.rag_system.qa_pipeline and \
               self.knowledge_base.corpus_loaded_successfully and self.knowledge_base.embeddings_generated:
                context = self.rag_system.retrieve_context(query_for_rag, self.knowledge_base.corpus_chunks, self.knowledge_base.chunk_embeddings)
                if context:
                    answer = self.rag_system.generate_answer(query_for_rag, context)
                    response_data["message"] = answer if answer else "I searched the corpus but couldn't determine who said that."
                else:
                    response_data["message"] = "I couldn't find relevant context in the corpus for that dialogue."
            else:
                response_data["message"] = "My advanced search for quote attribution is currently unavailable."
        
        elif intent == "search_corpus": # Direct keyword search
            query = entities.get("query_term")
            if query:
                lines = self.knowledge_base.find_lines_in_corpus(query)
                if lines and not (len(lines) == 1 and "corpus is not loaded" in lines[0]): 
                    response_data["message"] = f"Found these lines for '{query}':\n" + "\n".join([f"- \"{l}\"" for l in lines])
                elif lines and "corpus is not loaded" in lines[0]:
                     response_data["message"] = lines[0]
                else: response_data["message"] = f"Sorry, no lines found containing '{query}'."
            else: response_data["message"] = "What text would you like me to search for in the corpus?"
        
        elif intent == "ask_factual_question":
            # This now handles questions about plays, characters, quotes (if not attribution), and general facts.
            query = entities.get("query")
            if not self.rag_system or not self.rag_system.qa_pipeline: 
                response_data["message"] = "Sorry, my advanced understanding (RAG) system is not available right now (QA model issue)."
            elif not self.knowledge_base.corpus_loaded_successfully or not self.knowledge_base.embeddings_generated:
                 response_data["message"] = "Sorry, the corpus or its embeddings are not ready for advanced questions. Please check console for details."
            else:
                print(f"{BOT_NAME} (console): RAG: Retrieving context for: {query}") 
                context = self.rag_system.retrieve_context(query, self.knowledge_base.corpus_chunks, self.knowledge_base.chunk_embeddings)
                if context:
                    print(f"{BOT_NAME} (console): RAG: Generating answer with context...")
                    answer = self.rag_system.generate_answer(query, context)
                    response_data["message"] = answer
                else:
                    response_data["message"] = "I couldn't find relevant information in the corpus to answer that precisely."
        
        elif intent == "thank": response_data["message"] = random.choice(["You're welcome!", "No problem."])
        elif intent == "affirmative": response_data["message"] = random.choice(["Okay.", "Sure."])
        elif intent == "negative": response_data["message"] = random.choice(["Okay.", "Understood."])
        
        elif intent == "unknown":
            query = entities.get("query")
            # Check if self.intent_recognizer is available before accessing its attributes
            factual_q_keywords = self.intent_recognizer.intent_keywords["ask_factual_question"] if self.intent_recognizer else []
            if query and self.rag_system and self.rag_system.qa_pipeline and \
               self.knowledge_base.corpus_loaded_successfully and self.knowledge_base.embeddings_generated and \
               any(query.startswith(q_word) for q_word in factual_q_keywords): 
                print(f"{BOT_NAME} (console): RAG (fallback for unknown): Retrieving context for: {query}")
                context = self.rag_system.retrieve_context(query, self.knowledge_base.corpus_chunks, self.knowledge_base.chunk_embeddings)
                if context:
                    answer = self.rag_system.generate_answer(query, context)
                    response_data["message"] = answer
                else:
                    response_data["message"] = "I'm not sure how to help with that, and couldn't find specific info in the corpus for it."
            else:
                response_data["message"] = "I'm not sure how to respond. Try asking about Shakespeare's works or quotes."
        else: 
            response_data["message"] = "I'm not sure what you mean. Could you try asking differently?"
        return response_data

# --- ResponseGenerator ---
class ResponseGenerator: 
    def generate_response(self, response_data):
        return response_data.get("message", "I'm not sure how to respond.")

# --- Main Chatbot Orchestrator ---
class ShakespeareChatbot: 
    def __init__(self):
        self.knowledge_base = KnowledgeBase() 
        self.rag_system = RAGSystem() if TRANSFORMERS_AVAILABLE else None
        self.input_processor = InputProcessor()
        self.intent_recognizer = IntentRecognizer() 
        self.dialog_manager = DialogManager(self.knowledge_base, self.rag_system)
        self.dialog_manager.intent_recognizer = self.intent_recognizer # Pass instance for keyword access

        self.response_generator = ResponseGenerator()
        self.initial_status = [] 
        if not REQUESTS_AVAILABLE: self.initial_status.append("Warning: 'requests' library missing. Corpus download disabled.")
        if not TRANSFORMERS_AVAILABLE: self.initial_status.append("Warning: Key 'transformers' libraries missing. RAG features (and quantization) disabled.")
        if self.knowledge_base.corpus_loaded_successfully:
            self.initial_status.append("Shakespeare corpus loaded successfully.")
            if self.knowledge_base.embeddings_generated: self.initial_status.append("Corpus embeddings for RAG generated.")
            elif TRANSFORMERS_AVAILABLE: self.initial_status.append("Corpus embeddings for RAG could not be generated. Check console.")
        else: self.initial_status.append("Shakespeare corpus could not be loaded. Check console for errors.")
        if TRANSFORMERS_AVAILABLE and self.rag_system and self.rag_system.qa_pipeline: 
            self.initial_status.append("RAG system is active (QA model quantized).")
        elif TRANSFORMERS_AVAILABLE: self.initial_status.append("RAG system could not be fully initialized. Check console.")

    def get_response(self, user_input_str):
        processed_input = self.input_processor.process(user_input_str)
        if not processed_input: return "Please type a question or command.", "general" 
        intent_data = self.intent_recognizer.recognize_intent(processed_input)
        dialog_response_data = self.dialog_manager.manage_dialog(intent_data, processed_input)
        return self.response_generator.generate_response(dialog_response_data), dialog_response_data.get("type", "general")

# --- Streamlit UI Application ---
def run_streamlit_app(): 
    if not STREAMLIT_AVAILABLE:
        print("Cannot run Streamlit app because 'streamlit' library is not installed.")
        return
    st.set_page_config(page_title="BardBot - Shakespeare Chatbot", layout="wide")
    st.title("BardBot: Your Shakespearean Chat Companion 📚")
    st.caption("Ask me about Shakespeare's plays, characters, quotes, or search the complete works!")
    @st.cache_resource 
    def load_chatbot_instance():
        with st.spinner("BardBot is waking up... This might take a moment on first launch. Check console for progress."):
            chatbot_instance = ShakespeareChatbot()
        return chatbot_instance
    chatbot = load_chatbot_instance()
    if "messages" not in st.session_state:
        st.session_state.messages = []
        for status_msg in chatbot.initial_status:
             st.session_state.messages.append({"role": "assistant", "content": status_msg})
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask BardBot about Shakespeare:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text, _ = chatbot.get_response(prompt)
                st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("Streamlit is not installed. Please run 'pip install streamlit' and then 'streamlit run your_script_name.py'")
