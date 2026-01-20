import os
import re
import json
import urllib.request

INPUT_DIR = "data/cleaned_text_final"
OUTPUT_DIR = "data/chunks"
##TITLE_DIR = "data/titles"   # store <pageid>.title.json from fetch step

MIN_SENTENCES = 3
MAX_SENTENCES = 8
OVERLAP_SENTENCES = 2


# ---------------------------------------------------------
# Sentence splitter (regex-based)
# ---------------------------------------------------------
def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Extract section headings from text
# Wikipedia headings look like:  == Early life ==
# ---------------------------------------------------------
def extract_sections(text: str):
    sections = []
    for match in re.finditer(r"(==+)\s*(.*?)\s*\1", text):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        sections.append((start, level, title))
    return sections


# ---------------------------------------------------------
# Determine which section a sentence belongs to
# ---------------------------------------------------------
def assign_section(sentence_start, sections):
    current_section = None
    for start, level, title in sections:
        if sentence_start >= start:
            current_section = title
        else:
            break
    return current_section or "Introduction"


# ---------------------------------------------------------
# Create sentence-window chunks
# ---------------------------------------------------------
def create_chunks(sentences):
    chunks = []
    i = 0
    n = len(sentences)

    while i < n:
        window = sentences[i : i + MAX_SENTENCES]
        if len(window) < MIN_SENTENCES:
            break

        chunks.append({
            "text": " ".join(window),
            "num_sentences": len(window),
            "start_sentence": i,
            "end_sentence": i + len(window) - 1
        })

        i += MAX_SENTENCES - OVERLAP_SENTENCES

    return chunks


# ---------------------------------------------------------
# Extract wiki pageid from filename
# ---------------------------------------------------------
def extract_pageid_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0]


# ---------------------------------------------------------
# Load title from title directory
# ---------------------------------------------------------

def load_title(pageid: str):
    """
    Fetch the Wikipedia page title online using the pageid.
    Includes a User-Agent header as required by Wikipedia API.
    """
    url = (
        f"https://en.wikipedia.org/w/api.php?"
        f"action=query&pageids={pageid}&format=json&prop=info"
    )

    headers = {
        "User-Agent": "RAG-Hybrid-Wiki/1.0 (contact: 2024aa05720@wilp.bits-pilani.ac.in)"
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))

        page = data.get("query", {}).get("pages", {}).get(pageid, {})
        title = page.get("title")

        if title:
            return title

    except Exception as e:
        print(f"Warning: Could not fetch title for pageid={pageid}: {e}")

    return "Unknown Title"

# ---------------------------------------------------------
# Process a single file
# ---------------------------------------------------------
def chunk_file(input_path: str, filename: str):
    pageid = extract_pageid_from_filename(filename)
    wikipedia_url = f"https://en.wikipedia.org/?curid={pageid}"
    title = load_title(pageid)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract sections
    sections = extract_sections(text)

    # Split into sentences
    sentences = split_into_sentences(text)

    # Create chunks
    chunks = create_chunks(sentences)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        chunk_uid = f"{pageid}_chunk_{idx}"
        chunk_text_path = os.path.join(OUTPUT_DIR, f"{chunk_uid}.txt")
        chunk_meta_path = os.path.join(OUTPUT_DIR, f"{chunk_uid}.json")

        # Save chunk text
        with open(chunk_text_path, "w", encoding="utf-8") as f:
            f.write(chunk["text"])

        # Determine section for this chunk
        section = assign_section(chunk["start_sentence"], sections)

        # Save metadata
        metadata = {
            "pageid": pageid,
            "wikipedia_url": wikipedia_url,
            "title": title,
            "section": section,
            "chunk_uid": chunk_uid,
            "chunk_id": idx,
            "num_sentences": chunk["num_sentences"],
            "start_sentence": chunk["start_sentence"],
            "end_sentence": chunk["end_sentence"],
            "source_file": input_path
        }

        with open(chunk_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"Created {len(chunks)} chunks for pageid={pageid}")


# ---------------------------------------------------------
# Process all files in INPUT_DIR
# ---------------------------------------------------------
def chunk_all_files():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"Found {len(files)} cleaned files to chunk.")

    for i, filename in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Chunking {filename}")
        input_path = os.path.join(INPUT_DIR, filename)
        chunk_file(input_path, filename)

    print("\nAll files chunked successfully.")


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    chunk_all_files()