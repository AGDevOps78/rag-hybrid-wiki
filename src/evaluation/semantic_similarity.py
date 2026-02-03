import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EVAL_RESULTS = "/content/drive/MyDrive/Colab Notebooks/rag-hybrid-wiki-main/data/eval_results.jsonl"
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(pred, gold):
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gold = model.encode(gold, convert_to_tensor=True)

    return float(
        cosine_similarity(
            emb_pred.cpu().numpy().reshape(1, -1),
            emb_gold.cpu().numpy().reshape(1, -1)
        )[0][0]
    )


def evaluate_semantic_similarity_table(path=EVAL_RESULTS):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)
            sim = semantic_similarity(
                r["pred_answer"],
                r["gold_answer"]
            )

            rows.append({
                "ID": r["id"] + 1,
                "SemanticSim": sim,
                "Type": r["question_type"]
            })

    df = pd.DataFrame(rows)

    print("\n=== Summary Statistics ===")
    print(f"Mean Semantic Similarity: {df['SemanticSim'].mean():.4f}")
    print("-" * 60)

    print(f"\nMean Semantic Similarity: {df['SemanticSim'].mean():.4f}")
    print(df.head())

    return df


if __name__ == "__main__":
    evaluate_semantic_similarity_table()
