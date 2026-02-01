import os
import json
import random

def load_random_chunks(dir_path, n=50):
    # List all JSON metadata files
    json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

    # Randomly sample N files
    selected = random.sample(json_files, n)

    chunks = []

    for fname in selected:
        json_path = os.path.join(dir_path, fname)

        # Load metadata JSON
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        chunk_uid = meta["chunk_uid"]

        # Build path to matching .txt file
        txt_path = os.path.join(dir_path, f"{chunk_uid}.txt")

        # Load text
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = ""

        # Build final chunk object
        chunks.append({
            "chunk_id": meta.get("chunk_id"),
            "chunk_uid": chunk_uid,
            "title": meta.get("title"),
            "section": meta.get("section"),
            "wikipedia_url": meta.get("wikipedia_url"),
            "text": text
        })

    return chunks

import os
import json
import random

def load_random_chunks_stratified(dir_path, titles_count=10, chunks_per_title=5):
    # Load all JSON metadata files
    json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

    # Group chunks by title
    title_to_chunks = {}

    for fname in json_files:
        json_path = os.path.join(dir_path, fname)

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        title = meta.get("title")
        if not title:
            continue

        # Build matching .txt path
        chunk_uid = meta["chunk_uid"]
        txt_path = os.path.join(dir_path, f"{chunk_uid}.txt")

        # Load text
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        else:
            text = ""

        # Store chunk
        chunk_obj = {
            "chunk_id": meta.get("chunk_id"),
            "chunk_uid": chunk_uid,
            "title": title,
            "section": meta.get("section"),
            "wikipedia_url": meta.get("wikipedia_url"),
            "text": text
        }

        title_to_chunks.setdefault(title, []).append(chunk_obj)

    # Pick 10 random titles
    all_titles = list(title_to_chunks.keys())
    selected_titles = random.sample(all_titles, titles_count)

    # For each title, pick 5 chunks
    final_chunks = []
    for title in selected_titles:
        chunks = title_to_chunks[title]

        # If fewer than needed, sample with replacement
        if len(chunks) >= chunks_per_title:
            chosen = random.sample(chunks, chunks_per_title)
        else:
            chosen = random.choices(chunks, k=chunks_per_title)

        final_chunks.extend(chosen)

    return final_chunks

if __name__ == "__main__":
    dir_path = "data/chunks"
    random_chunks = load_random_chunks_stratified(dir_path, titles_count=2, chunks_per_title=5)
    for chunk in random_chunks:
        print(f"Chunk UID: {chunk['chunk_uid']}")
        print(f"Title: {chunk['title']}")
        print(f"Section: {chunk['section']}")
        print(f"Text Preview: {chunk['text'][:200]}...")
        print(f"URL: {chunk['wikipedia_url']}")
        print("-" * 40)