import requests
import os
import json
from urllib.parse import urlparse, parse_qs

HEADERS = {
    "User-Agent": "RAG-Hybrid-Wiki/1.0 (contact: 2024aa05720@wilp.bits-pilani.ac.in)"
}

CLEANED_TEXT_DIR = "data/cleaned_text"


# ---------------------------------------------------
# Extract pageid from Wikipedia URL
# ---------------------------------------------------
def extract_pageid(url: str) -> str | None:
    """
    Extract the pageid from a Wikipedia URL like:
    https://en.wikipedia.org/?curid=2148933
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    pageids = qs.get("curid")
    if not pageids:
        return None
    return pageids[0]


# ---------------------------------------------------
# Fetch plain text using Wikipedia API
# ---------------------------------------------------
def fetch_plain_text(pageid: str) -> str:
    """
    Fetch clean plain text for a Wikipedia page using the API.
    Returns empty string if page is missing or blocked.
    """
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "pageids": pageid,
        "format": "json"
    }

    try:
        resp = requests.get(api_url, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"[ERROR] API returned {resp.status_code} for pageid={pageid}")
            return ""

        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        page = pages.get(str(pageid), {})

        # Extract plain text
        text = page.get("extract", "")
        return text

    except Exception as e:
        print(f"[EXCEPTION] fetch_plain_text({pageid}): {e}")
        return ""


# ---------------------------------------------------
# Save cleaned text to disk
# ---------------------------------------------------
def save_clean_text(pageid: str, text: str) -> str:
    """
    Save cleaned text to data/cleaned_text/<pageid>.txt
    Returns the file path.
    """
    os.makedirs(CLEANED_TEXT_DIR, exist_ok=True)
    filepath = os.path.join(CLEANED_TEXT_DIR, f"{pageid}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


# ---------------------------------------------------
# Main function: fetch + save
# ---------------------------------------------------
def fetch_and_store(url: str) -> dict:
    """
    Given a Wikipedia URL:
    - Extract pageid
    - Fetch plain text
    - Save to disk
    - Return metadata dict
    """
    pageid = extract_pageid(url)
    if pageid is None:
        print(f"[WARNING] Could not extract pageid from URL: {url}")
        return {"url": url, "pageid": None, "text": "", "saved": False}

    text = fetch_plain_text(pageid)

    if not text.strip():
        print(f"[WARNING] Empty text for pageid={pageid} ({url})")
        return {"url": url, "pageid": pageid, "text": "", "saved": False}

    filepath = save_clean_text(pageid, text)

    return {
        "url": url,
        "pageid": pageid,
        "text": text,
        "saved": True,
        "filepath": filepath
    }

def fetch_all_from_fixed_urls(fixed_urls_path="data/fixed_urls.json"):
    """
    Read all URLs from fixed_urls.json and create cleaned text files.
    """
    if not os.path.exists(fixed_urls_path):
        print(f"[ERROR] {fixed_urls_path} does not exist")
        return

    # Load URLs
    with open(fixed_urls_path, "r", encoding="utf-8") as f:
        urls = json.load(f)

    print(f"Loaded {len(urls)} URLs from fixed_urls.json")

    results = []
    for i, url in enumerate(urls, start=1):
        print(f"\n[{i}/{len(urls)}] Fetching: {url}")
        result = fetch_and_store(url)
        results.append(result)

    print("\nFinished fetching all fixed URLs.")
    return results

# ---------------------------------------------------
# CLI entry point (optional)
# ---------------------------------------------------
if __name__ == "__main__":
    #test_url = "https://en.wikipedia.org/?curid=752"
    #result = fetch_and_store(test_url)
    result=fetch_all_from_fixed_urls()
    print(json.dumps(result, indent=2))