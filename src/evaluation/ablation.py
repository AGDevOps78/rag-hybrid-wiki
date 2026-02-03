import json
import math
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.generator import Generator

QUESTIONS_FILE = "/content/drive/MyDrive/Colab Notebooks/rag-hybrid-wiki-main/data/generated_questions.jsonl"
TOP_K = 10

# -----------------------------
# Utility functions
# -----------------------------
def load_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def chunk_id_to_url(chunk_id):
    page_id = chunk_id.split("_chunk_")[0]
    return f"https://en.wikipedia.org/?curid={page_id}"


# -----------------------------
# Metrics
# -----------------------------
def mrr_url_level(gold_urls, retrieved_chunks):
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        url = chunk_id_to_url(chunk["chunk_id"])
        if url in gold_urls:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_urls, retrieved_chunks, k=10):
    for i, chunk in enumerate(retrieved_chunks[:k]):
        url = chunk_id_to_url(chunk["chunk_id"])
        if url in gold_urls:
            return 1 / math.log2(i + 2)
    return 0.0


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_similarity(pred, gold):
    emb = embedder.encode([pred, gold], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


# -----------------------------
# Ablation runner
# -----------------------------
def run_ablation(method="dense"):
    questions = load_questions(QUESTIONS_FILE)
    questions = questions[:2]

    dense = DenseRetriever()
    sparse = SparseRetriever(index_path="both")
    generator = Generator()

    mrr_scores, ndcg_scores, sem_scores = [], [], []

    for q in questions:
        query = q["question"]
        gold_answer = q["ground_truth"]
        gold_urls = q["wikipedia_url"]

        # ---- Retrieval selection ----
        if method == "dense":
            retrieved = dense.retrieve(query, top_k=TOP_K)

        elif method == "sparse":
            retrieved = sparse.retrieve(query, top_k=TOP_K)

        elif method == "hybrid":
            retrieved = retrieve_hybrid(query, dense, sparse, top_n=TOP_K)

        else:
            raise ValueError("Invalid method")

        # ---- Generation ----
        pred_answer = generator.generate(query, retrieved[:3])
        print(pred_answer)

        # ---- Metrics ----
        mrr_scores.append(mrr_url_level(gold_urls, retrieved))
        ndcg_scores.append(ndcg_at_k(gold_urls, retrieved, k=10))
        sem_scores.append(semantic_similarity(pred_answer, gold_answer))

    return {
        "MRR": float(np.mean(mrr_scores)),
        "NDCG@10": float(np.mean(ndcg_scores)),
        "SemanticSim": float(np.mean(sem_scores)),
    }

dense_results = run_ablation("dense")
sparse_results = run_ablation("sparse")
hybrid_results = run_ablation("hybrid")

print("\n=== Ablation Results ===")
print("Dense  :", dense_results)
print("Sparse :", sparse_results)
print("Hybrid :", hybrid_results)
