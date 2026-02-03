import json
import math
import pandas as pd

EVAL_RESULTS = "/content/drive/MyDrive/Colab Notebooks/rag-hybrid-wiki-main/data/eval_results.jsonl"
K = 10


def ndcg_at_k_url_level(gold_urls, retrieved_urls, k=10):
    dcg = 0.0
    for i, url in enumerate(retrieved_urls[:k]):
        rel = 1 if url in gold_urls else 0
        dcg += (2**rel - 1) / math.log2(i + 2)

    ideal_rels = [1] * min(len(gold_urls), k)
    idcg = sum(
        (2**rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(ideal_rels)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ndcg_table(path=EVAL_RESULTS, k=10):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)
            ndcg = ndcg_at_k_url_level(
                r["gold_urls"],
                r["retrieved_urls"],
                k
            )

            rows.append({
                "ID": r["id"] + 1,
                "NDCG@10": ndcg,
                "Type": r["question_type"]
            })

    df = pd.DataFrame(rows)

    print("\n=== Summary Statistics ===")
    print(f"Mean NDCG@{k}: {df['NDCG@10'].mean():.4f}")
    print("-" * 60)

    print(f"\nMean NDCG@{k}: {df['NDCG@10'].mean():.4f}")
    print(df.head())

    return df


if __name__ == "__main__":
    evaluate_ndcg_table()
