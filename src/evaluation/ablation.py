import json
import math
import numpy as np
import os, sys
from sentence_transformers import SentenceTransformer

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.generator import Generator

from src.evaluation.eval_MRR import  ndcg_at_k_from_chunks, precision_at_k, find_supporting_chunks,answer_f1, answer_from_which_chunk_rank

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
EVAL_QUESTIONS = os.path.join(ROOT, "data", "generated_questions.jsonl")

QUESTIONS_FILE = EVAL_QUESTIONS
TOP_K = 5

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
def run_ablation(method="dense", top_k=TOP_K):
    questions = load_questions(QUESTIONS_FILE)
    questions = questions[:10]  # Limit to 10 for faster testing

    dense = DenseRetriever()
    sparse = SparseRetriever(index_path="both")
    generator = Generator()

    mrr_scores, ndcg_scores, sem_scores, precision_scores, answer_f1_scores = [], [], [], [], []

    for q in questions:
        query = q["question"]
        gold_answer = q["ground_truth"]
        gold_urls = q["wikipedia_url"]

        # ---- Retrieval selection ----
        if method == "dense":
            retrieved = dense.retrieve(query, top_k)

        elif method == "sparse":
            retrieved = sparse.retrieve(query, top_k)

        elif method == "hybrid":
            retrieved = retrieve_hybrid(query, dense, sparse, 3, top_k, top_k)

        else:
            raise ValueError("Invalid method")

        # ---- Generation ----
        pred_answer = generator.generate(query, retrieved)
        print(pred_answer)

        # ---- Metrics ----
        mrr_scores.append(mrr_url_level(gold_urls, retrieved))
        ndcg_scores.append(ndcg_at_k(gold_urls, retrieved, k=TOP_K) if method != "hybrid" else ndcg_at_k_from_chunks(pred_answer, retrieved, k=TOP_K))
        
        answer_f1_score = answer_f1(pred_answer, gold_answer)
        answer_f1_scores.append(answer_f1_score)

        supporting_chunks = find_supporting_chunks(pred_answer, [(c["chunk_id"], c["text"]) for c in retrieved], k=len(retrieved))
        precision_at_k_score = precision_at_k([c["chunk_id"] for c in retrieved], supporting_chunks, k=len(retrieved))
        precision_scores.append(precision_at_k_score)

        sem_scores.append(semantic_similarity(pred_answer, gold_answer))
    ndcg_text = f"NDCG@{top_k}"
    precision_at_k_text = f"Precision@{top_k}"
    return {
        "MRR": float(np.mean(mrr_scores)),
        ndcg_text: float(np.mean(ndcg_scores)),
        "SemanticSim": float(np.mean(sem_scores)),
        precision_at_k_text: float(np.mean(precision_scores)),
        "AnswerF1": float(np.mean(answer_f1_scores)),
        "SemanticSim": float(np.mean(sem_scores)),
    }
k =3 # top-k for ablation
dense_results = run_ablation("dense", k)
sparse_results = run_ablation("sparse", k)
hybrid_results = run_ablation("hybrid", k)

print("\n=== Ablation Results @ k =", k, " ===")
print("Dense  :", dense_results)
print("Sparse :", sparse_results)
print("Hybrid :", hybrid_results)
