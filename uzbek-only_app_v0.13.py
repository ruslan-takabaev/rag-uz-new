import os
import faiss
import requests
import numpy as np
import json
import pickle
import re
from typing import List, Dict, Generator
from collections import deque
from pathlib import Path

# --- Config ---
EMB_BASE_URL = os.getenv("EMB_API_URL", "https://p950-w002-runai-p950.runai-inference.dc.uz/v1/")
EMB_MODEL = "nvidia/llama-embed-nemotron-8b"
EMB_API_KEY = os.getenv("EMB_API_KEY", "")

LLM_BASE_URL = os.getenv("LLM_API_URL", "https://p950-w006x-runai-p950.runai-inference.dc.uz/v1/")
LLM_MODEL = "openai/gpt-oss-120b"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

INDEX_DIR = Path("rag_build/index-articles/")
FAISS_INDEX_PATH = INDEX_DIR / "faiss-uz.index"
CHUNK_PATH = INDEX_DIR / "kb_chunks-uz.pkl"
DIRECT_SEARCH_LIST_PATH = INDEX_DIR / "direct_search_list-uz.json"

MAX_HISTORY = 5
TOP_K_RETRIEVAL = 3

# --- Global Resources ---
RESOURCES = {"index": None, "chunks": [], "titles": []}


def load_resources():
    """Loads Uzbek Knowledge Base for the Legal Module."""
    global RESOURCES
    print(">>> Loading Avatar Knowledge Base...")

    # Load Titles
    if DIRECT_SEARCH_LIST_PATH.exists():
        with open(DIRECT_SEARCH_LIST_PATH, "r", encoding="utf-8") as f:
            RESOURCES["titles"] = json.load(f)

    # Load Index & Chunks
    if FAISS_INDEX_PATH.exists() and CHUNK_PATH.exists():
        try:
            RESOURCES["index"] = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(CHUNK_PATH, "rb") as f:
                RESOURCES["chunks"] = pickle.load(f)
            print(f"Loaded Legal DB: {len(RESOURCES['chunks'])} chunks.")
            print(f"Loaded FAISS Index: {RESOURCES['index'].ntotal} vectors.")
        except Exception as e:
            print(f"CRITICAL ERROR loading RAG resources: {e}")
            RESOURCES["index"] = None
    else:
        print(f"WARNING: RAG files not found at {INDEX_DIR}. Legal module will be disabled.")


# --- API Clients ---
class APIEmbedder:
    def __init__(self, base_url, model, api_key):
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def encode(self, texts: List[str]) -> np.ndarray:
        payload = {"model": self.model, "input": texts}
        try:
            r = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "data" not in data:
                print(f"Embedder API Error: Invalid response format: {data}")
                return np.zeros((len(texts), 4096), dtype=np.float32)

            embeddings = [d["embedding"] for d in data["data"]]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"!!! EMBEDDER FAILED: {e}")
            return np.zeros((len(texts), 4096), dtype=np.float32)


class AvatarBrain:
    def __init__(self):
        self.embedder = APIEmbedder(EMB_BASE_URL, EMB_MODEL, EMB_API_KEY)
        self.llm_endpoint = LLM_BASE_URL.rstrip("/") + "/chat/completions"
        self.llm_headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
        self.history = deque(maxlen=MAX_HISTORY)

    def _llm_request(self, messages, max_tokens=500, temperature=0.7, stream=False):
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "include_reasoning": False
        }
        return requests.post(self.llm_endpoint, headers=self.llm_headers, json=payload, stream=stream, timeout=30)

    def plan_next_step(self, query: str) -> str:
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in self.history])

        prompt = f"""You are the Router for an Uzbek AI Avatar. Analyze the user's intent.

AVAILABLE ACTIONS:
1. `direct_search`: User asks for a SPECIFIC law article number (e.g., "Mehnat Kodeksi 12-modda").
2. `legal_rag`: User asks a question about LAWS, FINES, PROCEDURES, RIGHTS, or GOVERNMENT REGULATIONS in Uzbekistan (e.g., "Qanday ajrashish mumkin??", "Tezlikni oshirganlik uchun jarima").
3. `history_reply`: User asks a direct follow-up about the PREVIOUS answer (e.g., "Nima bu?", "Takrorlang").
4. `general_chat`: EVERYTHING ELSE. Greetings, philosophy, "Do you like cats?", "Who are you?", general knowledge, math, coding.

HISTORY:
{history_str if history_str else "No history."}
---
QUERY: "{query}"
---
Respond ONLY with suitable action name in plaintext and nothing else."""

        try:
            r = self._llm_request([{"role": "user", "content": prompt}], max_tokens=500, temperature=0.0)
            if r.status_code != 200:
                print(f"Router LLM Error: {r.text}")
                return "general_chat"

            action = r.json()["choices"][0]["message"]["content"].strip().lower()
            print(f"[Router] Decision: {action}")

            if "direct" in action: return "direct_search"
            if "legal" in action or "rag" in action: return "legal_rag"
            if "history" in action: return "history_reply"
            return "general_chat"
        except Exception as e:
            print(f"[Router] Exception: {e}")
            return "general_chat"

    def generate_pure_llm_response(self, query: str) -> Generator[str, None, None]:
        system_prompt = (
            "Siz O'zbekistonda ishlab chiqilgan zamonaviy 3D Avatar yordamchisiz. "
            "Sizning vazifangiz foydalanuvchi bilan turli mavzularda suhbatlashish (falsafa, hayot, ilm-fan va h.k.). "
            "Javoblaringiz o'zbek tilida, samimiy va madaniyatli bo'lsin. "
            "Agar savolga javob berish uchun qonun kitobi kerak bo'lmasa, o'z bilimingizdan foydalaning."
        )
        return self._stream_response([{"role": "system", "content": system_prompt}, {"role": "user", "content": query}])

    def generate_rag_response(self, query: str, context_docs: List[Dict], is_direct: bool) -> Generator[
        str, None, None]:
        if not context_docs:
            # Fallback message handled here
            msg = f"Afsuski, qonunchilik bazasidan '{query}' bo'yicha aniq ma'lumot topolmadim. Iltimos, savolni o'zgartirib ko'ring yoki umumiy savol bering."
            yield msg
            return

        parts = []
        for d in context_docs:
            m = d['meta']
            # Safeguard if meta is missing fields
            title = m.get('document_title', 'Hujjat')
            article = m.get('article_number', '')
            text_excerpt = d.get('text', '')
            parts.append(f"Hujjat: {title}\nModda: {article}\nMatn: {text_excerpt}")

        context_str = "\n\n".join(parts)

        sys_prompt = (
            "Siz O'zbekiston qonunchiligi bo'yicha ekspert avatarsiz. "
            "Foydalanuvchi savoliga faqat quyidagi KONTEKST asosida aniq javob bering. "
            "Javobingizni og'zaki nutqqa moslab, qisqa va lo'nda qiling (maksimum 2-3 gap). "
            "Agar kontekstda javob bo'lmasa, 'Ma'lumot yo'q' deb ayting."
        )

        user_prompt = f"KONTEKST:\n{context_str}\n\nSAVOL: {query}\n\nJAVOB:"
        for token in self._stream_response(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]):
            yield token

    def extract_direct_search_entities(self, query: str) -> Dict:
        titles_preview = json.dumps(RESOURCES['titles'], ensure_ascii=False) if RESOURCES['titles'] else "[]"
        prompt = f"""Extract 'article_number' and 'document_title' from this Uzbek query. Use Official List: {titles_preview}... Query: "{query}" \nJSON only:"""
        try:
            r = self._llm_request([{"role": "user", "content": prompt}], max_tokens=5000)
            clean = re.sub(r"```json\s*|\s*```", "", r.json()["choices"][0]["message"]["content"].strip())
            return json.loads(clean)
        except Exception:
            return {}

    def rewrite_query(self, query: str) -> str:
        """Rewrites user query for better semantic search."""
        prompt = f"Yuridik qidiruv uchun savolni qayta yozing (faqat savolning o'zini qaytaring, ortiqcha so'zsiz).: '{query}'"
        try:
            r = self._llm_request([{"role": "user", "content": prompt}], max_tokens=5000)
            rewritten = r.json()["choices"][0]["message"]["content"].strip()
            if len(rewritten) < 5: return query
            return rewritten
        except Exception:
            return query

    def retrieve_semantically(self, query: str) -> List[Dict]:
        """Robust semantic search."""
        if not RESOURCES["index"] or RESOURCES["index"].ntotal == 0:
            print("[Search] Index is empty or not loaded.")
            return []

        print(f"[Search] Embedding query: '{query}'")
        vec = self.embedder.encode([query])

        # Check if embedding failed (returned zeros)
        if np.all(vec == 0):
            print("[Search] Warning: Embedding was all zeros. Search aborted.")
            return []

        faiss.normalize_L2(vec)

        # Search
        try:
            distances, indices = RESOURCES["index"].search(vec, TOP_K_RETRIEVAL)
        except Exception as e:
            print(f"[Search] FAISS search failed: {e}")
            return []

        results = []
        for i in indices[0]:
            if i == -1: continue
            if i < len(RESOURCES["chunks"]):
                results.append(RESOURCES["chunks"][i])

        print(f"[Search] Found {len(results)} documents.")
        print("DOCS:", results)
        return results

    def retrieve_exact(self, article: str, title: str) -> List[Dict]:
        return [c for c in RESOURCES["chunks"] if
                c['meta']['article_number'] == article and c['meta']['document_title'] == title]

    def generate_history_response(self, query: str) -> Generator[str, None, None]:
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in self.history])
        sys_prompt = "Siz suhbat tarixini eslab qoluvchi yordamchisiz. Oldingi javoblarga asoslanib gapiring."
        return self._stream_response([{"role": "system", "content": sys_prompt}, {"role": "user",
                                                                                  "content": f"TARIX:\n{history_str}\n\nSAVOL: {query}\n\nJAVOB:"}])

    def _stream_response(self, messages):
        try:
            r = self._llm_request(messages, max_tokens=3500, stream=True)
            if r.status_code != 200:
                yield f" LLM Error ({r.status_code})"
                return

            for line in r.iter_lines():
                if line and line.startswith(b'data: '):
                    line_str = line[len(b'data: '):].decode('utf-8')
                    if line_str == '[DONE]': break
                    try:
                        token = json.loads(line_str)['choices'][0].get('delta', {}).get('content')
                        if token: yield token
                    except:
                        continue
        except Exception as e:
            yield f" [Connection Error: {e}]"

    def process_input(self, user_text: str) -> Generator[str, None, None]:
        print(f"\n[Input] '{user_text}'")

        # 1. Router
        action = self.plan_next_step(user_text)

        full_response = ""
        generator = None

        # 2. Execution Switch
        if action == "general_chat":
            generator = self.generate_pure_llm_response(user_text)

        elif action == "history_reply":
            generator = self.generate_history_response(user_text)

        else:
            # RAG Logic
            docs = []
            is_direct = False

            if action == "direct_search":
                print("[Logic] Attempting Direct Search...")
                entities = self.extract_direct_search_entities(user_text)
                if entities.get("article_number") and entities.get("document_title"):
                    docs = self.retrieve_exact(str(entities['article_number']), entities['document_title'])
                    if docs:
                        is_direct = True
                        print(f"[Logic] Direct Hit: {entities['document_title']} Article {entities['article_number']}")

                # FALLBACK
                if not docs:
                    print("[Logic] Direct search empty. Switching to Semantic.")
                    action = "legal_rag"

            if action == "legal_rag":
                print("[Logic] Performing Semantic Search...")
                rewritten = self.rewrite_query(user_text)
                print(f"[Logic] Rewritten: {rewritten}")
                docs = self.retrieve_semantically(rewritten)

            generator = self.generate_rag_response(user_text, docs, is_direct)

        # 3. Output Stream
        for token in generator:
            full_response += token
            yield token

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})


# --- Console Interface ---
def run_console_app():
    load_resources()
    avatar = AvatarBrain()

    print("\n" + "=" * 50)
    print("Muxlisa (Hybrid Mode: Legal + General)")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Siz: ").strip()
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input: continue

            print("Avatar: ", end="", flush=True)
            for token in avatar.process_input(user_input):
                print(token, end="", flush=True)
            print("\n" + "-" * 30)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_console_app()
