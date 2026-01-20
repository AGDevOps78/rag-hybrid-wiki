import os
import json
import re
from rank_bm25 import BM25Okapi

CHUNKS_DIR = "data/chunks"
BM25_INDEX_PATH = "data/bm25_index.json"


# ---------------------------------------------------------
# Simple tokenizer for BM25
# ---------------------------------------------------------
def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


# ---------------------------------------------------------
# Load all chunks + metadata
# ---------------------------------------------------------
def load_chunks():
    txt_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".txt")])
    print(f"Found {len(txt_files)} chunk files for BM25 indexing.")

    documents = []
    metadata_list = []

    for txt_file in txt_files:
        chunk_uid = txt_file.replace(".txt", "")
        text_path = os.path.join(CHUNKS_DIR, txt_file)
        meta_path = os.path.join(CHUNKS_DIR, f"{chunk_uid}.json")

        # Load text
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Load metadata
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        documents.append(text)
        metadata_list.append({
            "chunk_uid": chunk_uid,
            "text": text,
            "metadata": metadata
        })

    return documents, metadata_list


# ---------------------------------------------------------
# Build BM25 index
# ---------------------------------------------------------
def build_bm25_index():
    documents, metadata_list = load_chunks()

    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    index_data = {
        "documents": documents,
        "tokenized_docs": tokenized_docs,
        "metadata": metadata_list
    }

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    print(f"BM25 index saved to {BM25_INDEX_PATH}")


# ---------------------------------------------------------
# Load BM25 index from disk
# ---------------------------------------------------------
def load_bm25_index():
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError("BM25 index not found. Run bm25_index.py to build it.")

    with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    bm25 = BM25Okapi(data["tokenized_docs"])
    return bm25, data["metadata"]


# ---------------------------------------------------------
# Search API
# ---------------------------------------------------------
def search_bm25(query: str, top_k: int = 5):
    bm25, metadata_list = load_bm25_index()

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(scores, metadata_list),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    results = []
    for score, meta in ranked:
        results.append({
            "score": float(score),
            "chunk_uid": meta["chunk_uid"],
            "text": meta["text"],
            "metadata": meta["metadata"]
        })

    return results


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    build_bm25_index()