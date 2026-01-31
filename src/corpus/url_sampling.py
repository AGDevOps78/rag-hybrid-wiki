import requests
import random
import json
import os
from bs4 import BeautifulSoup

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
CATEGORIES = [
    "Physics", "Chemistry", "Biology", "Mathematics", "Computer_science",
    "History", "Geography", "Economics", "Psychology", "Philosophy",
    "Engineering", "Medicine", "Art", "Music", "Literature",
    "Sports", "Environment", "Technology", "Sociology", "Politics","Religion","Communication","Business","Education","Law","Anthropology","Linguistics","Astronomy","Geology",
    "Political_science","Statistics","Architecture","Cultural_studies","Film","Theatre","Dance","Machine_learning"]


MIN_WORDS = 150
TARGET_COUNT = 200
RANDOM_COUNT =  300
OUTPUT_PATH = "data/fixed_urls.json"
RANDOM_PATH = "data/random_urls.json"

# ---------------------------------------------
# HELPERS
# ---------------------------------------------
def get_pages_from_category(category):
    """Fetch up to 500 pages from a Wikipedia category."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",
        "format": "json"
    }
    headers = {
        "User-Agent": "RAG-Hybrid-Wiki/1.0 (contact: 2024aa05720@wilp.bits-pilani.ac.in)"
    }


    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print("Status:", response.status_code)
        print(response.text[:500])
        data = response.json()
        ##print(data)
        return data["query"]["categorymembers"]
    except Exception:
        print(Exception)
        return []

def dedupe_fixed_urls():
    if not os.path.exists(OUTPUT_PATH):
        print("fixed_urls.json does not exist")
        return

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        urls = json.load(f)

    print(f"Loaded {len(urls)} URLs")

    unique_urls = list(set(urls))
    removed = len(urls) - len(unique_urls)
    #print(unique_urls)
    if removed > 0:
        print(f"Removed {removed} duplicate URLs")
    else:
        print("No duplicates found")

    # Save cleaned list
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_urls, f, indent=2)

    print(f"Saved {len(unique_urls)} unique URLs back to fixed_urls.json")
    return unique_urls


def build_page_url(pageid):
    return f"https://en.wikipedia.org/?curid={pageid}"


def page_word_count(url):
    """Fetch page HTML and count words in <p> tags."""
    try:
        headers = {
        "User-Agent": "RAG-Hybrid-Wiki/1.0 (contact: 2024aa05720@wilp.bits-pilani.ac.in)"
        }
        html = requests.get(url,headers=headers, timeout=10).text
        #print("HTML text",html)
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        print(text)
        return len(text.split())
    except Exception:
        return 0


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------
def generate_fixed_urls():
    """Generate 200 unique Wikipedia URLs with ≥200 words."""
    collected = set()

    print("Sampling Wikipedia categories to build fixed URL set...")
    print(os.path.abspath(OUTPUT_PATH))
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = set(json.load(f))
        print(f"Found existing fixed_urls.json with {len(existing)} URLs")
    else:
        existing = set()
        print("No fixed_urls.json found — starting fresh")
    
    # Step 2: Determine how many more URLs are needed
    remaining = TARGET_COUNT - len(existing)
    if remaining <= 0:
        print("Already have 200 URLs — nothing to do")
        return

    print(f"Need to add {remaining} more URLs")


    while len(collected) < remaining:
        category = random.choice(CATEGORIES)
        print(f"chosen {category}")
        pages = get_pages_from_category(category)

        if not pages:
            continue

        page = random.choice(pages)
        url = build_page_url(page["pageid"])

        if url in collected:
            continue

        wc = page_word_count(url)
        if wc >= MIN_WORDS:
            collected.add(url)
            print(f"[{len(collected)}/{TARGET_COUNT}] Added: {url} ({wc} words)")
        else:
            print(f"Skipped (too short): {url} ({wc} words)")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    collected.update(existing)
    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(list(collected), f, indent=2)

    print(f"\nSaved {len(collected)} fixed URLs to {OUTPUT_PATH}")

# ---------------------------------------------
# GENERATE RANDOM URLS
# ---------------------------------------------
def generate_random_urls(fixed_urls,count=RANDOM_COUNT):
    """Generate 300 unique Wikipedia URLs with ≥200 words."""
    collected = set()
    print("Sampling Wikipedia categories to build random URL set...")
    print(os.path.abspath(RANDOM_PATH))
    if os.path.exists(RANDOM_PATH):
        with open(RANDOM_PATH, "r", encoding="utf-8") as f:
            existing = set(json.load(f))
        print(f"Found existing random_urls.json with {len(existing)} URLs")
    else:
        existing = set()
        print("No random_urls.json found — starting fresh")

    
    # Step 2: Determine how many more URLs are needed
    remaining = count - len(existing)
    if remaining <= 0:
        print(f"Already have {count} URLs — nothing to do")
        return

    while len(collected) < remaining:
        category = random.choice(CATEGORIES)
        pages = get_pages_from_category(category)

        if not pages:
            continue

        page = random.choice(pages)
        url = build_page_url(page["pageid"])

        if (url in collected) or (url in fixed_urls):
            continue
        
        wc = page_word_count(url)
        if wc >= MIN_WORDS:
            collected.add(url)
            print(f"[{len(collected)}/{count}] Added random URL: {url}")
        else:
            print(f"Skipped (too short): {url} ({wc} words)")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(RANDOM_PATH), exist_ok=True)
    collected.update(existing)
    # Save RANDOM URLs
    with open(RANDOM_PATH, "w", encoding="utf-8") as f:
        json.dump(list(collected), f, indent=2)

    print(f"Saved {len(collected)} random URLs to {RANDOM_PATH}")
    return list(collected)


# ---------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------
if __name__ == "__main__":
    uniqueurls =dedupe_fixed_urls()
    generate_fixed_urls()
    generate_random_urls(uniqueurls)