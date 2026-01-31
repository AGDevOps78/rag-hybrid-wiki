import os
import json
import re
from rank_bm25 import BM25Okapi

CHUNKS_DIR = "data/chunks"
BM25_INDEX_PATH = "data/bm25_index.json"

CHUNKS_DIR_RANDOM = "data_random/chunks"
BM25_INDEX_PATH_RANDOM = "data_random/bm25_random_index.json"


# ---------------------------------------------------------
# Simple tokenizer for BM25
# ---------------------------------------------------------
def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


# ---------------------------------------------------------
# Load all chunks + metadata from a directory
# ---------------------------------------------------------
def load_chunks_from(chunks_dir: str):
    if not os.path.exists(chunks_dir):
        return [], []

    txt_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".txt")])
    print(f"Found {len(txt_files)} chunk files in {chunks_dir} for BM25 indexing.")

    documents = []
    metadata_list = []
    print(f"Loading chunks from {chunks_dir}...")
    for txt_file in txt_files:
        chunk_uid = txt_file.replace(".txt", "")
        text_path = os.path.join(chunks_dir, txt_file)
        meta_path = os.path.join(chunks_dir, f"{chunk_uid}.json")

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
# Build BM25 index for a given chunks dir -> index path
# ---------------------------------------------------------
def build_bm25_index_for(chunks_dir: str, index_path: str):
    documents, metadata_list = load_chunks_from(chunks_dir)
    print(f"Building BM25 index for {chunks_dir} documents to {index_path}...")
    if not documents:
        print(f"No documents found in {chunks_dir}; skipping index build for {index_path}.")
        return

    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    index_data = {
        "documents": documents,
        "tokenized_docs": tokenized_docs,
        "metadata": metadata_list
    }
    print(f"Saving BM25 index to {index_path}...")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    print(f"BM25 index saved to {index_path}")


# ---------------------------------------------------------
# Build both indexes if their source dirs exist
# ---------------------------------------------------------
def build_bm25_index():
    # Primary index
    if os.path.exists(CHUNKS_DIR):
        build_bm25_index_for(CHUNKS_DIR, BM25_INDEX_PATH)
    else:
        print(f"{CHUNKS_DIR} not found; skipping primary BM25 index.")

    # Random index
    if os.path.exists(CHUNKS_DIR_RANDOM):
        build_bm25_index_for(CHUNKS_DIR_RANDOM, BM25_INDEX_PATH_RANDOM)
    else:
        print(f"{CHUNKS_DIR_RANDOM} not found; skipping random BM25 index.")


# ---------------------------------------------------------
# Load BM25 index from disk (accepts a path, defaults to primary)
# - If called with index_path=None or "both", returns:
#     {"primary": (bm25, metadata) or None, "random": (bm25, metadata) or None}
# - Otherwise returns (bm25, metadata) as before.
# ---------------------------------------------------------
def load_bm25_index(index_path: str = BM25_INDEX_PATH):
    if index_path is None or index_path == "both":
        results = {}
        for name, path in (("primary", BM25_INDEX_PATH), ("random", BM25_INDEX_PATH_RANDOM)):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                bm25 = BM25Okapi(data["tokenized_docs"])
                results[name] = (bm25, data["metadata"])
            else:
                results[name] = None
        return results

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"BM25 index not found at {index_path}. Run bm25_embed.py to build it.")

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bm25 = BM25Okapi(data["tokenized_docs"])
    return bm25, data["metadata"]


# ---------------------------------------------------------
# Search API (defaults to both indexes; pass `index_path` to change)
# ---------------------------------------------------------
def search_bm25(query: str, top_k: int = 5, index_path: str = "both"):
    res = load_bm25_index(index_path)

    candidates = []
    query_tokens = tokenize(query)
    #check if returned value is dict or tuple
    if isinstance(res, dict):
        for entry in res.values():
            if entry is None:
                continue
            bm25_obj, metadata_list = entry
            scores = bm25_obj.get_scores(query_tokens)
            candidates.extend((float(s), m) for s, m in zip(scores, metadata_list))
    else:
        bm25_obj, metadata_list = res
        scores = bm25_obj.get_scores(query_tokens)
        candidates.extend((float(s), m) for s, m in zip(scores, metadata_list))

    ranked = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]

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