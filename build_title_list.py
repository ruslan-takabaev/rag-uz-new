import pickle
import json
from pathlib import Path

INDEX_DIR = Path("rag_build/index-articles/")
CHUNK_PATHS = [
    INDEX_DIR / "kb_chunks-ru.pkl",
    INDEX_DIR / "kb_chunks-uz.pkl"
]
OUTPUT_FILE = INDEX_DIR / "canonical_titles.json"

def generate_canonical_title_list():
    """
    Reads the processed chunk files, extracts all unique document titles,
    and saves them to a JSON file.
    """
    if not INDEX_DIR.exists():
        print(f"Error: Index directory '{INDEX_DIR}' not found.")
        print("Please run the 'build_index.py' script first.")
        return

    print("--- Canonical Title List Generator ---")
    canonical_titles = set()

    for path in CHUNK_PATHS:
        if not path.exists():
            print(f"Warning: Chunk file not found at '{path}', skipping.")
            continue
        
        print(f"Loading chunks from '{path.name}'...")
        with open(path, "rb") as f:
            chunks = pickle.load(f)
        
        for chunk in chunks:
            title = chunk.get("meta", {}).get("document_title")
            if title:
                canonical_titles.add(title)

    if not canonical_titles:
        print("Error: No document titles were found. Cannot create list.")
        return

    # Convert set to a sorted list for consistent ordering
    sorted_titles = sorted(list(canonical_titles))
    
    print(f"\nFound {len(sorted_titles)} unique document titles.")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_titles, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved canonical title list to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    generate_canonical_title_list()

