import os
import json
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = "data/chunks"
EMBED_OUTPUT = "data/embeddings.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------
def load_model():
    print(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# Load all chunk text + metadata
# ---------------------------------------------------------
def load_chunks():
    txt_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".txt")])
    print(f"Found {len(txt_files)} chunk files.")

    chunks = []

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
    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# ---------------------------------------------------------
# Save embeddings to JSONL
# ---------------------------------------------------------
def save_embeddings(chunks, embeddings):
    os.makedirs(os.path.dirname(EMBED_OUTPUT), exist_ok=True)

    with open(EMBED_OUTPUT, "w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embeddings):
            record = {
                "chunk_uid": chunk["chunk_uid"],
                "text": chunk["text"],
                "embedding": emb.tolist(),
                "metadata": chunk["metadata"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved embeddings to {EMBED_OUTPUT}")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    model = load_model()
    chunks = load_chunks()
    embeddings = embed_chunks(model, chunks)
    save_embeddings(chunks, embeddings)
    print("Embedding pipeline complete.")


if __name__ == "__main__":
    main()