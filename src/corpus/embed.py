import os
import json
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = "data/chunks"
EMBED_OUTPUT = "data/embeddings.jsonl"

CHUNKS_DIR_RANDOM = "data_random/chunks"
EMBED_OUTPUT_RANDOM = "data_random/embeddings.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------
def load_model():
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# Load all chunk text + metadata from a directory
# ---------------------------------------------------------
def load_chunks_from(chunks_dir: str):
    if not os.path.exists(chunks_dir):
        return []

    txt_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".txt")])
    print(f"Found {len(txt_files)} chunk files in {chunks_dir}.")

    chunks = []
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

        chunks.append({
            "chunk_uid": chunk_uid,
            "text": text,
            "metadata": metadata
        })

    return chunks


# ---------------------------------------------------------
# Embed all chunks
# ---------------------------------------------------------
def embed_chunks(model, chunks):
    if not chunks:
        return []
    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# ---------------------------------------------------------
# Save embeddings to JSONL
# ---------------------------------------------------------
def save_embeddings(chunks, embeddings, output_path: str):
    if not chunks or embeddings is None or len(embeddings) == 0:
        print(f"No embeddings to save for {output_path}.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embeddings):
            record = {
                "chunk_uid": chunk["chunk_uid"],
                "text": chunk["text"],
                "embedding": emb.tolist(),
                "metadata": chunk["metadata"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved embeddings to {output_path}")


# ---------------------------------------------------------
# Main pipeline: primary skipped if EMBED_OUTPUT exists,
# random always (re)created.
# ---------------------------------------------------------
def main():
    model = load_model()

    # Primary: skip if EMBED_OUTPUT exists
    if os.path.exists(EMBED_OUTPUT):
        print(f"{EMBED_OUTPUT} already exists; skipping primary embeddings.")
    else:
        chunks = load_chunks_from(CHUNKS_DIR)
        if chunks:
            embeddings = embed_chunks(model, chunks)
            save_embeddings(chunks, embeddings, EMBED_OUTPUT)
        else:
            print(f"{CHUNKS_DIR} not found or empty; skipping primary embeddings.")

    # Random: always recreate if chunks present
    chunks_rand = load_chunks_from(CHUNKS_DIR_RANDOM)
    if chunks_rand:
        embeddings_rand = embed_chunks(model, chunks_rand)
        save_embeddings(chunks_rand, embeddings_rand, EMBED_OUTPUT_RANDOM)
    else:
        print(f"{CHUNKS_DIR_RANDOM} not found or empty; skipping random embeddings.")

    print("Embedding pipeline complete.")


if __name__ == "__main__":
    main()