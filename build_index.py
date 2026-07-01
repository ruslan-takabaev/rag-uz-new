"""
Chunking files by articles;
Embedding chunks;
Creating metadata file;
"""

import os
import re
import sys
import json
import pickle
import faiss
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from bs4 import BeautifulSoup

# --- Config ---
BASE_URL = os.getenv("EMB_URL", "")
API_KEY = os.getenv("EMB_API_KEY", "")
EMBED_MODEL = "nvidia/llama-embed-nemotron-8b"

BATCH_SIZE = 16
OUTPUT_DIR = Path("rag_build/index-articles/")
TITLE_MAP_PATH = OUTPUT_DIR / "canonical_title_map.json"

# --- API Helper ---
def call_api_embeddings(texts: List[str]) -> List[List[float]]:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": texts}
    try:
        r = requests.post(BASE_URL.rstrip("/") + "/embeddings", headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        return [item["embedding"] for item in r.json()["data"]]
    except requests.RequestException as e:
        print(f"\nAPI Error during embedding: {e}. Skipping batch.")
        return []

# --- Chunking Logic ---
def build_chunks_with_canonical_titles(kb_folder: Path, lang: str, title_map: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """
    Builds chunks for a specific language using the correct, language-specific regex pattern.
    """
    print(f"[{lang}] Building chunks and applying canonical titles...")

    # Regex patterns for separating articles
    patterns = {
        'ru': re.compile(r"^(?:Статья|статья)\s+([0-9]+)[\-\.]?\s*(.*)", re.IGNORECASE),
        'uz': re.compile(r"^([0-9]+)[\-\s]?(?:modda|MODDA)\.?\s*(.*)", re.IGNORECASE)
    }
    p = patterns[lang]

    chunks_with_meta = []

    for path in tqdm(list(kb_folder.glob("*.txt")), desc=f"[{lang}] Chunking files"):
        title_info = title_map.get(path.name, {"title": path.stem.title()})
        canonical_title = title_info.get("title")

        text = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser").get_text()
        lines = text.split('\n')
        current_article_match = None
        current_article_text = []

        for line in lines:
            line = line.strip()
            if not line: continue
            match = p.match(line)
            if match:
                if current_article_match:
                    meta = {
                        "source_file": path.name,
                        "document_title": canonical_title,
                        "article_number": current_article_match.group(1).strip(),
                        "article_title": current_article_match.group(2).strip(),
                    }
                    chunks_with_meta.append({"text": " ".join(current_article_text).strip(), "meta": meta})
                current_article_match = match
                current_article_text = [line]
            elif current_article_match:
                current_article_text.append(line)

        if current_article_match:
            meta = {
                "source_file": path.name,
                "document_title": canonical_title,
                "article_number": current_article_match.group(1).strip(),
                "article_title": current_article_match.group(2).strip(),
            }
            chunks_with_meta.append({"text": " ".join(current_article_text).strip(), "meta": meta})

    return chunks_with_meta

# --- Main Processing ---
def process_language(lang: str, base_kb_folder: str, title_map: Dict[str, Dict]):
    """
    Processes a single language: chunks its files and creates its embeddings.
    """
    kb_folder = Path(f"{base_kb_folder}/{lang}")

    chunks = build_chunks_with_canonical_titles(kb_folder, lang, title_map)

    if not chunks:
        print(f"ERROR: Still no chunks generated for language '{lang}'. Please check the files in '{kb_folder}' and their format.")
        return

    n_chunks = len(chunks)
    print(f"[{lang}] Successfully generated {n_chunks} chunks.")
    with open(OUTPUT_DIR / f"kb_chunks-{lang}.pkl", "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Embed chunks and save the index
    print(f"[{lang}] Embedding {n_chunks} chunks...")
    index = None
    all_texts = [item['text'] for item in chunks]
    for i in tqdm(range(0, n_chunks, BATCH_SIZE), desc=f"[{lang}] Embedding batches"):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        embs = call_api_embeddings(batch_texts)
        if not embs: continue
        vectors = np.array(embs, dtype=np.float32)
        faiss.normalize_L2(vectors)
        if index is None: index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

    if index:
        faiss.write_index(index, str(OUTPUT_DIR / f"faiss-{lang}.index"))
        print(f"[{lang}] Successfully saved FAISS index.")
    else:
        print(f"[{lang}] ERROR: FAISS index was not created due to embedding failures.")

    # Save metadata for inspection
    all_meta = [item['meta'] for item in chunks]
    with open(OUTPUT_DIR / f"meta-{lang}.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    base_kb_folder = "laws"

    print("--- RAG System Chunk & Embedding Rebuilder ---")

    # Load the essential, pre-existing title map
    if not TITLE_MAP_PATH.exists():
        print(f"CRITICAL ERROR: The title map file '{TITLE_MAP_PATH}' was not found.")
        print("Please run the full 'build_index.py' script once to generate it.")
        sys.exit(1)

    with open(TITLE_MAP_PATH, "r", encoding="utf-8") as f:
        title_map = json.load(f)
    print(f"Successfully loaded existing title map with {len(title_map)} entries.")

    # Process each language separately with the corrected logic
    #process_language("ru", base_kb_folder, title_map)
    process_language("uz", base_kb_folder, title_map)

    print("\nRebuilding process complete.")
