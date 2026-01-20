import os
import re
import unicodedata
from bs4 import BeautifulSoup

INPUT_DIR = "data/cleaned_text"
OUTPUT_DIR = "data/cleaned_text_final"


# -------------------------------
# Cleaning functions
# -------------------------------
def strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def collapse_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_boilerplate(text: str) -> str:
    patterns = [
        r"©\s*\d{4}.*",
        r"All rights reserved.*",
        r"Page \d+ of \d+",
        r"^\s*Advertisement\s*$",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text

def clean_text(text: str, lowercase: bool = False) -> str:
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = collapse_whitespace(text)
    if lowercase:
        text = text.lower()
    return text


# -------------------------------
# Clean a single file
# -------------------------------
def clean_file(input_path: str, output_path: str, lowercase=False):
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw, lowercase=lowercase)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    return output_path


# -------------------------------
# Clean ALL files in INPUT_DIR
# -------------------------------
def clean_all_files(lowercase=False):
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"Found {len(files)} files in {INPUT_DIR}")

    for i, filename in enumerate(files, start=1):
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        print(f"[{i}/{len(files)}] Cleaning {filename}")
        clean_file(in_path, out_path, lowercase=lowercase)

    print("\nAll files cleaned successfully.")
    return True


# -------------------------------
# CLI entry point
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean all text files in a directory.")
    parser.add_argument("--lowercase", action="store_true", help="Convert text to lowercase")
    args = parser.parse_args()

    clean_all_files(lowercase=args.lowercase)