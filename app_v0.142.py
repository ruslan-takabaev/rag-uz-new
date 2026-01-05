import os
import faiss
import requests
import numpy as np
import json
import pickle
import re
import gradio as gr
from typing import List, Dict, Any
from collections import deque
from datetime import datetime
from pathlib import Path

# --- Config ---
EMB_BASE_URL = os.getenv("EMB_API_URL", "https://p950-w002-runai-p950.runai-inference.dc.uz/v1/")
EMB_MODEL = "nvidia/llama-embed-nemotron-8b"
EMB_API_KEY = os.getenv("EMB_API_KEY", "")

LLM_BASE_URL = os.getenv("LLM_API_URL", "https://p950-w006x-runai-p950.runai-inference.dc.uz/v1/")
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

INDEX_DIR = Path("rag_build/index-articles/")

FAISS_INDEX_PATH_UZ = INDEX_DIR / "faiss-uz.index"
CHUNK_PATH_UZ = INDEX_DIR / "kb_chunks-uz.pkl"
DIRECT_SEARCH_LIST_UZ_PATH = INDEX_DIR / "direct_search_list-uz.json"

FAISS_INDEX_PATH_RU = INDEX_DIR / "faiss-ru.index"
CHUNK_PATH_RU = INDEX_DIR / "kb_chunks-ru.pkl"
DIRECT_SEARCH_LIST_RU_PATH = INDEX_DIR / "direct_search_list-ru.json"

MAX_HISTORY = 5
TOP_K_RETRIEVAL = 5

# --- Loading resources ---
loaded_resources = {}
PRIMARY_LAW_TITLES = {}

def load_all_resources():
    global PRIMARY_LAW_TITLES
    if DIRECT_SEARCH_LIST_RU_PATH.exists():
        with open(DIRECT_SEARCH_LIST_RU_PATH, "r", encoding="utf-8") as f: PRIMARY_LAW_TITLES['ru'] = json.load(f)
        print(f"Loaded {len(PRIMARY_LAW_TITLES['ru'])} primary Russian law titles.")
    if DIRECT_SEARCH_LIST_UZ_PATH.exists():
        with open(DIRECT_SEARCH_LIST_UZ_PATH, "r", encoding="utf-8") as f: PRIMARY_LAW_TITLES['uz'] = json.load(f)
        print(f"Loaded {len(PRIMARY_LAW_TITLES['uz'])} primary Uzbek law titles.")
    load_resources_for_lang('uz'); load_resources_for_lang('ru')

def load_resources_for_lang(lang: str) -> dict:
    if lang in loaded_resources: return loaded_resources[lang]
    paths = {'uz': (FAISS_INDEX_PATH_UZ, CHUNK_PATH_UZ), 'ru': (FAISS_INDEX_PATH_RU, CHUNK_PATH_RU)}
    faiss_path, chunk_path = paths.get(lang, (None, None))
    if not all(p and p.exists() for p in [faiss_path, chunk_path]): loaded_resources[lang] = {}; return {}
    try:
        index = faiss.read_index(str(faiss_path))
        with open(chunk_path, "rb") as f: chunks = pickle.load(f)
        resources = {"index": index, "chunks": chunks}; loaded_resources[lang] = resources
        print(f"--- Loaded {len(chunks)} chunks for {lang.upper()} ---")
        return resources
    except Exception as e: print(f"Error loading resources for {lang}: {e}"); loaded_resources[lang] = {}; return {}

# --- API Clients ---
class APIEmbedder:
    def __init__(self, base_url, model, api_key): self.endpoint = base_url.rstrip("/") + "/embeddings"; self.model, self.headers = model, {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    def encode(self, texts: List[str]) -> np.ndarray: r = requests.post(self.endpoint, headers=self.headers, json={"model": self.model, "input": texts}, timeout=120); r.raise_for_status(); return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)

class RemoteLLM:
    def __init__(self, base_url, model, api_key): self.endpoint = base_url.rstrip("/") + "/chat/completions"; self.model, self.headers = model, {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    def _make_request(self, messages, max_tokens=150, temperature=0.0, stream=False, timeout=(10, 600)):
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": stream}
        return requests.post(self.endpoint, headers=self.headers, json=payload, timeout=timeout, stream=stream)

    def detect_language(self, text: str) -> str:
        prompt = f"Analyze the language of the text. Respond with ONLY 'uz' or 'ru'.\n\nText: \"{text}\"\n\nResponse:"
        try:
            r = self._make_request([{"role": "user", "content": prompt}], max_tokens=10, timeout=(5, 20)); r.raise_for_status(); lang_code = r.json()["choices"][0]["message"]["content"].strip().lower()
            return lang_code if lang_code in ['uz', 'ru'] else 'uz'
        except Exception as e: print(f"LLM language detection failed: {e}. Defaulting to 'uz'."); return 'uz'

    # --- Router ---
    def plan_next_step(self, query: str, chat_history: str) -> str:
        prompt = f"""You are a planner for a legal RAG system for Uzbekistan's legislation. Your task is to decide the next action based on the user's query and the chat history.

Choose ONLY ONE of the following actions:
1.  `direct_search`: If the query asks for a NEW, specific article number of a law or code (e.g., "what about article 33 of the civil code?"). Choose this even if there's history on another topic.
2.  `semantic_search`: If the query is a NEW, general legal question (e.g., "what is the penalty for damaging trees?"). This is the default for new topics.
3.  `answer_from_history`: If the query is a direct follow-up that can be answered **using only the text already provided** in the chat history (e.g., "what was the fine amount you just mentioned?").

**CRITICAL RULE:** If the chat history is empty, you MUST choose `direct_search` or `semantic_search`.

---
*Example 1*
CHAT HISTORY: No history yet.
QUERY: "What is article 198 of the criminal code?"
ACTION: direct_search

*Example 2*
CHAT HISTORY: No history yet.
QUERY: "What are the rules for firing an employee?"
ACTION: semantic_search

*Example 3*
CHAT HISTORY:
User: What are the rules for firing an employee?
Assistant: [Provides text from the Labor Code about termination rules, including a 2-week notice period.]
QUERY: "What is the notice period you mentioned?"
ACTION: answer_from_history

*Example 4*
CHAT HISTORY:
User: What are the rules for firing an employee?
Assistant: [Provides text from the Labor Code.]
QUERY: "Ok, now what about article 33 of the civil code?"
ACTION: direct_search
---

**YOUR TASK:**
CHAT HISTORY:
{chat_history if chat_history else "No history yet."}
---
QUERY:
"{query}"
---
ACTION:"""
        try:
            r = self._make_request([{"role": "user", "content": prompt}], max_tokens=20, timeout=(10, 20))
            r.raise_for_status()
            response = r.json()["choices"][0]["message"]["content"].strip().lower()
            if "direct_search" in response: return "direct_search"
            if "answer_from_history" in response: return "answer_from_history"
            return "semantic_search" # Default
        except Exception as e:
            print(f"LLM planner failed: {e}. Defaulting to 'semantic_search'.")
            return "semantic_search"

    def extract_direct_search_entities(self, query: str, primary_titles_for_lang: List[str]) -> Dict:
        prompt = f"From the user's query about the legislation of **Uzbekistan**, extract the 'article_number' and the exact 'document_title' from the provided OFFICIAL LIST of **Uzbek laws**. Respond ONLY with a valid JSON object.\n\nQuery: \"{query}\"\n\n--- OFFICIAL LIST ---\n{json.dumps(primary_titles_for_lang, ensure_ascii=False)}\n---"
        try:
            r = self._make_request([{"role": "user", "content": prompt}], max_tokens=400, timeout=(10, 120)); r.raise_for_status()
            clean_response = re.sub(r"```json\s*|\s*```", "", r.json()["choices"][0]["message"]["content"].strip())
            return json.loads(clean_response)
        except Exception as e: print(f"LLM entity extraction failed: {e}. Returning empty dict."); return {}

    def rewrite_query(self, query: str, lang: str) -> str:
        prompt = f"""Rewrite the following user query to be more effective for a semantic search within a vector database of **Uzbekistan's legislation**.
The rewritten query must be in {'Russian' if lang == 'ru' else 'Uzbek'} and must focus on legal terms relevant **only to the laws of Uzbekistan**.
Do not mention or assume laws from other countries (like Russia).

Original query: '{query}'

Rewritten query for Uzbekistan's context:"""
        try:
            r = self._make_request([{"role": "user", "content": prompt}], max_tokens=300, timeout=(10, 45)); r.raise_for_status(); rewritten = r.json()["choices"][0]["message"]["content"].strip().strip("\"'")
            return rewritten if len(rewritten) > len(query) + 5 else query
        except Exception as e: print(f"LLM query rewriting failed: {e}. Using original query."); return query
            
    def generate_answer_from_history(self, question, chat_history, lang):
        prompt_map = {
            'ru': "Ты — ИИ-ассистент по законодательству Узбекистана.\n- Отвечай на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предыдущем КОНТЕКСТЕ из истории чата.",
            'uz': "Siz O'zbekiston qonunchiligi bo'yicha AI-yordamchisiz.\n- Foydalanuvchining savoliga suhbat tarixidagi oldingi KONTEKSTGA asoslanib javob bering."
        }
        system_prompt = prompt_map[lang]
        user_prompt = f"CHAT HISTORY:\n---\n{chat_history}\n---\n\nCURRENT QUESTION: {question}\n\nAnswer based on history:"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        response = self._make_request(messages, max_tokens=2048, temperature=0.1, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line and line.startswith(b'data: '):
                line_str = line[len(b'data: '):].decode('utf-8')
                if line_str == '[DONE]': break
                try: 
                    chunk = json.loads(line_str)
                    token = chunk['choices'][0].get('delta', {}).get('content')
                    if token: yield token
                except json.JSONDecodeError: continue

    def generate_answer_from_context(self, question, context_docs, chat_history, lang, is_direct_hit):
        prompts = {
            'ru': {
                'semantic': "Ты — ИИ-ассистент по законодательству Узбекистана.\n- Твой контекст — исключительно законы Республики Узбекистан.\n- Отвечай строго на основе предоставленного КОНТЕКСТА.\n- В ответе всегда цитируй основной закон (например, 'Уголовный кодекс'), используя `[Название документа](ссылка), статья номер`.\n- Если ответа в контексте нет, вежливо сообщи: \"К сожалению, я не смог найти точную информацию по вашему запросу в доступных документах.\".",
                'direct': "Ты — ИИ-ассистент по законодательству Узбекистана.\n- **Не повторяй текст статьи**, он уже показан пользователю.\n- Твоя задача — кратко ответить на вопрос пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на тексте предоставленной статьи."
            },
            'uz': {
                'semantic': "Siz O'zbekiston qonunchiligi bo'yicha AI-yordamchisiz.\n- Sizning bilim doirangiz faqat O'zbekiston Respublikasi qonunlari bilan cheklangan.\n- Faqat taqdim etilgan KONTEKST asosida javob bering.\n- Javobda har doim manbani keltiring: `[Hujjat nomi](havola), modda raqami`.\n- Agar kontekstda javob topilmasa, \"Afsuski, men sizning so'rovingiz bo'yicha mavjud hujjatlardan aniq ma'lumot topa olmadim,\" deb ayting.",
                'direct': "Siz O'zbekiston qonunchiligi bo'yicha AI-yordamchisiz.\n- **Modda matnini takrorlamang**, u foydalanuvchiga allaqachon ko'rsatilgan.\n- Sizning vazifangiz - taqdim etilgan modda matniga asoslanib, foydalanuvchining savoliga qisqa va aniq javob berish."
            }
        }
        system_prompt = prompts[lang]['direct'] if is_direct_hit else prompts[lang]['semantic']
        context_parts = []
        if context_docs:
            for d in context_docs:
                meta = d['meta']
                doc_id = meta.get('source_file', '').split('.')[0]
                link = f"https://lex.uz/docs/{doc_id}" if doc_id.isdigit() else "#"
                context_parts.append(f"Источник: {meta['document_title']}\nСтатья: {meta['article_number']}\nСсылка: {link}\nТекст: {d['text']}")
            context_str = "\n\n".join(context_parts)
        else:
            context_str = "Соответствующие документы не найдены."
        
        user_prompt = f"CHAT HISTORY:\n---\n{chat_history}\n---\n\nCONTEXT:\n---\n{context_str}\n---\n\nCURRENT QUESTION: {question}\n\nAnswer:"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self._make_request(messages, max_tokens=2048, temperature=0.1, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line and line.startswith(b'data: '):
                line_str = line[len(b'data: '):].decode('utf-8')
                if line_str == '[DONE]': break
                try: 
                    chunk = json.loads(line_str)
                    token = chunk['choices'][0].get('delta', {}).get('content')
                    if token: yield token
                except json.JSONDecodeError: continue

# --- RAG Core Logic ---
embedder = APIEmbedder(EMB_BASE_URL, EMB_MODEL, EMB_API_KEY)
llm = RemoteLLM(LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)
chat_histories = {}

def retrieve_by_metadata(article_number: str, document_title: str, lang: str) -> List[Dict]:
    print(f"Performing exact search for Article '{article_number}' in '{document_title}'")
    resources = load_resources_for_lang(lang)
    if not resources or 'chunks' not in resources: return []
    for chunk in resources['chunks']:
        meta = chunk['meta']
        if meta['article_number'] == article_number and meta['document_title'] == document_title:
            return [chunk]
    return []

def retrieve_documents_semantically(query: str, lang: str, top_k: int) -> List[Dict]:
    resources = load_resources_for_lang(lang)
    if not resources or 'index' not in resources: return []
    try:
        query_embedding = embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        _, indices = resources["index"].search(query_embedding, top_k)
        return [resources["chunks"][i] for i in indices[0] if i < len(resources["chunks"])]

    except Exception as e:
        print(f"Error during semantic retrieval: {e}"); return []

# --- Gradio Interface ---
def chat_response(message: str, history: List[List[str]], request: gr.Request):
    if not message.strip():
        yield ""
        return

    session_id = request.client.host
    print(f"\n{'='*50}\nTime: {datetime.now():%Y-%m-%d %H:%M:%S} | Session: {session_id}")
    print(f"Original Query: '{message}'")

    if not history and chat_histories.get(session_id):
        print(f"UI cleared. Clearing server-side history for session: {session_id}")
        chat_histories[session_id].clear()

    session_history_for_llm = chat_histories.get(session_id, deque(maxlen=MAX_HISTORY))
    chat_history_string = "\n".join([f"Q: {h['content']}" if h['role'] == 'user' else f"A: {h['content']}" for h in session_history_for_llm])
    
    print(">>> Detecting language...")
    detected_lang = llm.detect_language(message)
    print(f"<<< Language detected: {detected_lang}")

    print(">>> 1. LLM Call: Planning next step...")
    action = llm.plan_next_step(message, chat_history_string)
    print(f"<<< Planner decided action: '{action}'")

    full_answer = ""
    token_generator = None
    
    if action == "answer_from_history":
        print("Strategy: Answer from history.")
        token_generator = llm.generate_answer_from_history(message, chat_history_string, detected_lang)
    
    else: # This block handles both 'direct_search' and 'semantic_search'
        print(f"Strategy: Retrieval required ({action}).")
        docs = []
        is_direct_hit = False

        if action == "direct_search":
            primary_titles_for_lang = PRIMARY_LAW_TITLES.get(detected_lang, [])
            print(">>> Extracting entities for direct search...")
            entities = llm.extract_direct_search_entities(message, primary_titles_for_lang)
            print(f"<<< Entities extracted: {entities}")

            if entities.get("article_number") and entities.get("document_title"):
                doc_title = entities['document_title']
                if doc_title in primary_titles_for_lang:
                    docs = retrieve_by_metadata(str(entities['article_number']), doc_title, lang=detected_lang)
                    if len(docs) == 1:
                        is_direct_hit = True
                        doc = docs[0]
                        meta, text = doc['meta'], doc['text']
                        doc_id = meta.get('source_file', '').split('.')[0]
                        link = f"https://lex.uz/docs/{doc_id}" if doc_id.isdigit() else "#"
                        title_lang = "Извлечение из" if detected_lang == 'ru' else "Hujjatdan parcha"
                        article_lang = "Статья" if detected_lang == 'ru' else "Modda"
                        full_answer = (
                            f"**{title_lang}: [{meta['document_title']}]({link}), {article_lang} {meta['article_number']}**\n\n"
                            f"```\n{text}\n```\n\n---\n\n"
                        )
                else:
                    action = "semantic_search" # Fallback if title is wrong
            else:
                action = "semantic_search" # Fallback if entities not found
        
        if action == "semantic_search":
            print(">>> Rewriting query for semantic search...")
            rewritten_query = llm.rewrite_query(message, detected_lang)
            print(f"<<< Rewritten query: '{rewritten_query}'")
            docs = retrieve_documents_semantically(rewritten_query, detected_lang, top_k=TOP_K_RETRIEVAL)

        if docs: print(f"Retrieved {len(docs)} documents.")
        else: print("No relevant documents found for RAG path.")
        
        yield full_answer

        print(">>> 2. LLM Call: Generating final answer from retrieved context...")
        token_generator = llm.generate_answer_from_context(message, docs, chat_history_string, detected_lang, is_direct_hit)

    # --- UNIFIED STREAMING AND HISTORY UPDATE ---
    try:
        if token_generator:
            for token in token_generator:
                full_answer += token
                yield full_answer
        
        session_history_for_llm.append({"role": "user", "content": message})
        session_history_for_llm.append({"role": "assistant", "content": full_answer})
        chat_histories[session_id] = session_history_for_llm

    except Exception as e:
        print(f"ERROR generating final response stream: {e}")
        error_msg = full_answer + "\n\nAn error occurred while generating the response."
        yield error_msg

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), title="Uzbekistan Legislation AI Assistant") as app:
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; text-align: center; padding: 20px;">
                <span style="font-size: 2.5em; margin-right: 15px;">🏛️</span>
                <div>
                    <h1 style="margin: 0; font-size: 2.2em; font-weight: 600;">Uzbekistan Legislation AI Assistant</h1>
                    <p style="margin: 5px 0 0; color: #4B5563;">Your intelligent guide to the laws of Uzbekistan</p>
                </div>
            </div>
        """)

        gr.ChatInterface(
            fn=chat_response,
            chatbot=gr.Chatbot(
                height=600,
                type="messages",
                avatar_images=("user.png", "bot.png"),
                show_label=False
            ),
            textbox=gr.Textbox(placeholder="Ask your legal question in Uzbek or Russian...", container=False, scale=7),
            examples=[
                "О чем говорится в статье 21 уголовного кодекса?",
                "Mehnat shartnomasini bekor qilish tartibi qanday?",
                "Что такое индивидуальный трудовой спор?",
                "100-modda mehnat kodeksi haqida nima deydi?"
            ],
            cache_examples=False
        )

    return app

if __name__ == "__main__":
    load_all_resources()
    print("\nServer is ready and running.")
    interface = create_interface()
    interface.queue().launch(server_name="0.0.0.0", server_port=9987, max_threads=20, share=True)

