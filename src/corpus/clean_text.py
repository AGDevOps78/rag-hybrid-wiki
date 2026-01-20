import os
import re
import unicodedata
from bs4 import BeautifulSoup

def strip_html(text: str) -> str:
    """Remove HTML tags, scripts, and styles."""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (NFKC)."""
    return unicodedata.normalize("NFKC", text)

def collapse_whitespace(text: str) -> str:
    """Replace multiple spaces/newlines with a single space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns."""
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
    """Full cleaning pipeline."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = collapse_whitespace(text)

    if lowercase:
        text = text.lower()

    return text

def clean_file(input_path: str, output_path: str, lowercase: bool = False) -> str:
    """Read a .txt file, clean it, and save to output_path."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw, lowercase=lowercase)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    return output_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean a text file for RAG pipelines.")
    parser.add_argument("input", help="Path to input .txt file")
    parser.add_argument("output", help="Path to save cleaned .txt file")
    parser.add_argument("--lowercase", action="store_true", help="Convert text to lowercase")

    args = parser.parse_args()

    out = clean_file(args.input, args.output, lowercase=args.lowercase)
    print(f"Cleaned file saved to: {out}")