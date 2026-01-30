import os
import requests
from bs4 import BeautifulSoup
import json

URL_FILE = "data/fixed_urls.json"   # 200 URLs
OUTPUT_DIR = "data/cleaned_text"
RANDOM_FILE = "data/random_urls.json"
OUTPUT_RANDOM = "data_random/cleaned_text"
HEADERS = {
    "User-Agent": "RAG-Hybrid-Wiki/1.0 (contact: 2024aa05720@wilp.bits-pilani.ac.in/2024aa05224@wilp.bits-pilani.ac.in)"
}

def fetch_page(url):
    try:
        html = requests.get(url, headers=HEADERS, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)
        return text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def if_page_exists(pageid,out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{pageid}.txt")
    if os.path.exists(path):
        # Page already saved
        return True
    else:
        return False

def save_page(text, pageid,out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pageid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def main():
    with open(URL_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    for i, url in enumerate(urls, start=1):
        pageid = url.split("=")[-1]  # extract Wikipedia pageid from URL
        print(f"[{i}/{len(urls)}] Fetching {url}")
        if if_page_exists(pageid, OUTPUT_DIR)==False:
            text = fetch_page(url)
            if text:
                save_page(text, pageid,OUTPUT_DIR)
        else:
            print(f"Page already exists {pageid}")
            continue
    print("All pages fetched and saved to data/cleaned_text.")
    with open(RANDOM_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    for i, url in enumerate(urls, start=1):
        pageid = url.split("=")[-1]  # extract Wikipedia pageid from URL
        print(f"[{i}/{len(urls)}] Fetching {url}")
        if if_page_exists(pageid,OUTPUT_RANDOM)==False:
            text = fetch_page(url)
            if text:
                save_page(text, pageid, OUTPUT_RANDOM)
        else:
            print(f"Page already exists {pageid}")
            continue
    print("All pages fetched and saved to data/cleaned_text.")

if __name__ == "__main__":
    main()
