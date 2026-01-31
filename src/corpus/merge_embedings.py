import json
import os
from collections import Counter

def merge_embeddings_jsonl(files, output_path):
    """
    Merge multiple embedding JSONL files, log counts, detect duplicates
    by chunk_uid, and write a deduplicated merged JSONL.
    """

    print("\n=== MERGE EMBEDDINGS STARTED ===")
    print(f"Output: {output_path}\n")

    all_items = []
    id_counter = Counter()

    # ---------------------------------------------------------
    # Read all files and log counts
    # ---------------------------------------------------------
    for fpath in files:
        if not os.path.isfile(fpath):
            print(f"[WARN] File not found: {fpath}")
            continue

        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip blank lines

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON in {fpath}: {e}")
                    continue

                if "chunk_uid" not in obj:
                    print(f"[WARN] Skipping item without 'chunk_uid': {obj}")
                    continue

                all_items.append(obj)
                id_counter[obj["chunk_uid"]] += 1
                count += 1

        print(f"[INFO] Loaded {count} embedding items from {fpath}")

    # ---------------------------------------------------------
    # Detect duplicates
    # ---------------------------------------------------------
    duplicates = [cid for cid, c in id_counter.items() if c > 1]

    print("\n=== DUPLICATE CHECK ===")
    if duplicates:
        print(f"[WARN] Found {len(duplicates)} duplicate chunk_uids:")
        for cid in duplicates:
            print(f"  - {cid} (count={id_counter[cid]})")
    else:
        print("[OK] No duplicates found")

    # ---------------------------------------------------------
    # Deduplicate
    # ---------------------------------------------------------
    seen = set()
    deduped = []

    for item in all_items:
        cid = item["chunk_uid"]
        if cid not in seen:
            seen.add(cid)
            deduped.append(item)

    print("\n=== SUMMARY ===")
    print(f"Total items loaded: {len(all_items)}")
    print(f"Unique items kept:  {len(deduped)}")
    print(f"Duplicates removed: {len(all_items) - len(deduped)}")

    # ---------------------------------------------------------
    # Write merged JSONL
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for item in deduped:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[OK] Merged embeddings JSONL written to: {output_path}")
    print("=== MERGE EMBEDDINGS COMPLETE ===\n")


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    merge_embeddings_jsonl(
        files=[
            "data_random/embeddings.jsonl",
            "data/embeddings.jsonl"
        ],
        output_path="data/embeddings_merged.jsonl"
    )