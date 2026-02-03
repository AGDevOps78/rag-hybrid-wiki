import json, re
import pandas as pd
from collections import Counter
import os,sys
os.chdir("..")
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.generator import Generator
from src.retrieval.hybrid import retrieve_hybrid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
EVAL_QUESTIONS = os.path.join(ROOT, "data", "generated_questions.jsonl")
EVAL_RESULTS = os.path.join(ROOT, "data", "eval_results.jsonl")

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def answer_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    overlap = sum((pred_counts & gold_counts).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def read_n_questions(path, n):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                items.append(json.loads(line))
    return items

def answer_from_which_chunk(answer, retrieved_chunks):
    ans_tokens = set(normalize(answer).split())
    best_chunk = None
    best_overlap = 0

    for chunk in retrieved_chunks:
        chunk_tokens = set(normalize(chunk["text"]).split())
        overlap = len(ans_tokens & chunk_tokens)

        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    return best_chunk, best_overlap

def answer_from_which_chunk_rank(answer, retrieved_chunks):
    ans_tokens = set(normalize(answer).split())
    best_chunk = None
    best_overlap = 0
    retrieved_url_by_rank = []
    best_rank = None
    for x, r in enumerate(retrieved_chunks):
        retrieved_url_by_rank.append(chunk_id_to_url(r["chunk_id"]))
        print(r["chunk_id"], r["score_rrf"], retrieved_url_by_rank[x])
    
    for rank,chunk in enumerate(retrieved_chunks) :
        chunk_tokens = set(normalize(chunk["text"]).split())
        overlap = len(ans_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk
            best_rank = rank+1

    return best_chunk, best_overlap, 1/best_rank if best_rank else 0.0, retrieved_url_by_rank[best_rank-1] if best_rank else None



def chunk_id_to_url(chunk_id: str) -> str:
    """
    Convert a chunk ID like '29816_chunk_9' into a Wikipedia URL.
    """
    page_id = chunk_id.split("_chunk_")[0]
    return f"https://en.wikipedia.org/?curid={page_id}"


def mrr_url_level(gold_urls,retrieved_url_by_rank):
    rank = None
    for idx, url in enumerate(retrieved_url_by_rank):
        if url in gold_urls:
            rank = idx+1
            break

    rr_scores = 1.0 / rank if rank else 0.0

    return rr_scores

def evaluate_rag(n=20, path="data/generated_questions.jsonl"):
    questions = read_n_questions(path, n)
    results = []
    mrr =0.0
    for i,q in enumerate(questions):
        question_text = q["question"]
        gold_answer = q["ground_truth"]
        gold_urls = q["wikipedia_url"]
        q_type= q["question_type"]
        # Load retrievers once
        dense_ret = DenseRetriever()
        sparse_ret = SparseRetriever(index_path="both")

            # Hybrid retrieval
        retrieved_chunks = retrieve_hybrid(question_text, dense_ret, sparse_ret, top_n=3)
        # LLM generation
        pred_answer = Generator().generate(question_text, retrieved_chunks)

        # Print answer
        print("\n=== Generated Answer ===\n")
        print(pred_answer)

        print("\n=== Gold Answer ===\n")
        print(gold_answer, gold_urls)
        # Print retrieved chunks + RRF scores
        print("\n=== Retrieved Chunks ===\n")
        retrieved_url_by_rank = []
        for x, r in enumerate(retrieved_chunks):
            retrieved_url_by_rank.append(chunk_id_to_url(r["chunk_id"]))
            print(r["chunk_id"], r["score_rrf"], retrieved_url_by_rank[x])

        mrr += mrr_url_level(gold_urls,retrieved_url_by_rank)
        print(f"RR for this question: {mrr_url_level(gold_urls,retrieved_url_by_rank)}\n")
        
        best_chunk_id,overlap,reciprocal_rank,best_url = answer_from_which_chunk_rank(pred_answer, retrieved_chunks)
        print(f" best chunk : {best_chunk_id} overlap: {overlap} Best URL contributing to answer: {best_url}, Reciprocal Rank: {reciprocal_rank}\n")
        
        # 3. Find supporting chunk
        best_chunk, overlap = answer_from_which_chunk(pred_answer, retrieved_chunks)

        print(f"Best supporting chunk ID: {best_chunk['chunk_id'] if best_chunk else 'None'}, Overlap: {overlap}")
        # 4. Compute F1
        f1 = answer_f1(pred_answer, gold_answer)
        print(f"F1 Score: {f1:.4f}\n")
        
        results.append({
            "id": i,
            "question": question_text,
            "pred_answer": pred_answer,
            "gold_answer": gold_answer,
            "f1": f1,
            "question_type": q_type,
            "retrieved_urls": retrieved_url_by_rank,
            "gold_urls": gold_urls,
            "reciprocal_rank": mrr_url_level(gold_urls,retrieved_url_by_rank),
            "supporting_chunk": best_chunk["chunk_id"] if best_chunk else None,
            "supporting_overlap": overlap
        })
    # 5. Compute MRR
    mrr = mrr / len(questions) if questions else 0.0
    print(f"MRR@{n}: {mrr:.4f}")
   
    
    return results , mrr

def print_table(results, limit=20):
    """
    Print a compact table of evaluation results and show mean F1 + mean RR.
    """
    if not results:
        print("No results to display.")
        return

    # Header
    print("\n=== Evaluation Table ===\n")
    print(f"{'ID':<5} {'F1':<10} {'RR':<10} {'overlap':<15} {'Type':<15}")
    print("-" * 60)

    # Print rows
    for row in results[:limit]:
        print(
            f"{row['id']:<5} "
            f"{row['f1']:<10.4f} "
            f"{row['reciprocal_rank']:<10.4f} "
            f"{row['supporting_overlap']:<15.4f} "
            f"{row['question_type']:<15}"
        )

    # Compute means
    mean_f1 = sum(r["f1"] for r in results) / len(results)
    mean_rr = sum(r["reciprocal_rank"] for r in results) / len(results)

    print("\n=== Summary Statistics ===")
    print(f"Mean F1: {mean_f1:.4f}")
    print(f"Mean Reciprocal Rank (RR): {mean_rr:.4f}")
    print("-" * 60)


def results_to_dataframe(results, limit=20):
    """
    Convert evaluation results into a pandas DataFrame and
    compute summary statistics.
    """
    if not results:
        return pd.DataFrame(), {"mean_f1": 0.0, "mean_rr": 0.0}

    # Build DataFrame
    df = pd.DataFrame([
        {
            "ID": row["id"],
            "F1": row["f1"],
            "RR": row["reciprocal_rank"],
            "Overlap": row["supporting_overlap"],
            "Type": row["question_type"]
        }
        for row in results[:limit]
    ])

    # Summary statistics
    mean_f1 = sum(r["f1"] for r in results) / len(results)
    mean_rr = sum(r["reciprocal_rank"] for r in results) / len(results)

    summary = {
        "mean_f1": mean_f1,
        "mean_rr": mean_rr
    }

    return df, summary

def save_results_jsonl(results, path):
    """
    Save evaluation results to a JSONL file.
    Appends if file exists, creates new file otherwise.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results → {path}")

def load_results_jsonl(path=EVAL_RESULTS):
    """
    Load a JSONL file and return a list of dicts with an added 'id' field.
    """
    if not os.path.exists(path):
        print(f"No existing results file found at: {path}")
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                obj = json.loads(line)
                obj["id"] = idx + 1
                rows.append(obj)

    print(f"Loaded {len(rows)} rows from {path}")
    return rows
# Example usage:
results, mrr = evaluate_rag( n=1, path=EVAL_QUESTIONS)
#print(results)
#print(f"Final MRR: {mrr}")

# Save results
#save_results_jsonl(results, EVAL_RESULTS)

# Load existing results with IDs
table = load_results_jsonl()
print_table(table[:50],50)   # preview first 5 rows
df, summary = results_to_dataframe(table)
print(f"Mean F1: {summary['mean_f1']:.4f}, Mean RR: {summary['mean_rr']:.4f}")
print(df.head())

