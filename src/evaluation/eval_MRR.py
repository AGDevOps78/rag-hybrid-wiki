import math
import json, re
import time
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer
import os,sys
os.chdir("..")
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.generator import Generator
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.qchecker import qCheckGenerator 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
EVAL_QUESTIONS = os.path.join(ROOT, "data", "generated_questions.jsonl")
EVAL_RESULTS = os.path.join(ROOT, "data", "eval_results.jsonl")

model = SentenceTransformer("all-MiniLM-L6-v2")


from sentence_transformers import SentenceTransformer, util

class SentenceTransformerQuestionChecker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        self.valid_examples = [
            "What is the meaning of X?",
            "How does this work?",
            "Why does this happen?",
            "Where can I find information about X?",
            "Who is responsible for X?",
            "When does X occur?"
        ]

        self.invalid_examples = [
            "Hello world",
            "This is a sentence.",
            "The quick brown fox",
            "How are you doing today?",
            "Hello, can you help me with this?",
            "What a wonderful day",
            "Greetings"
        ]

        # Encode once
        self.valid_emb = self.model.encode(self.valid_examples, convert_to_tensor=True)
        self.invalid_emb = self.model.encode(self.invalid_examples, convert_to_tensor=True)

    def is_valid(self, query, threshold=0.12):
        q_emb = self.model.encode(query, convert_to_tensor=True)

        sim_valid = util.cos_sim(q_emb, self.valid_emb).mean().item()
        sim_invalid = util.cos_sim(q_emb, self.invalid_emb).mean().item()

        score = sim_valid - sim_invalid
        return score > threshold, score, sim_valid, sim_invalid

                     

QUESTION_STOPWORDS = {
    "the","is","are","a","an","and","or","of","to","in","on","for","with",
    "as","by","at","from","that","this","it","be","was","were","can","may",
    "not","but","if","into","their","its","they","them","these","those",
    # question words
    "what","why","how","when","where","who","which","whom","whose",
    "define","explain","compare","contrast"
}

QUESTION_CUES = {
    "what","why","how","when","where","who","which","whom","whose","define","explain","compare","contrast",
    "does","do","is","are","can","should","would","will","could"
}

def is_valid_question(q: str, min_len: int = 5) -> bool:
    if not isinstance(q, str):
        return False

    q_clean = q.strip().lower()

    # Basic length check
    if len(q_clean) < min_len:
        return False

    # Must contain alphabetic characters
    if not re.search(r"[a-zA-Z]", q_clean):
        return False

    # Tokenize
    tokens = re.findall(r"[a-zA-Z]+", q_clean)

    # Reject if all tokens are stopwords (e.g., "what is", "define", "explain")
    meaningful_tokens = [t for t in tokens if t not in QUESTION_STOPWORDS]
    if len(meaningful_tokens) == 0:
        return False

    # Interrogative structure check
    starts_with_qword = tokens[0] in QUESTION_CUES
    ends_with_qmark = q_clean.endswith("?")

    if starts_with_qword or ends_with_qmark:
        return True

    return False


qchecker = qCheckGenerator()
def is_valid_question_strict(q: str, llm=qchecker) -> bool:
    """
    First uses deterministic rules.
    If they fail and an LLM is provided, uses semantic fallback.
    """
    if not is_valid_question(q):
        return False
    
    print(f"Basic checks passed for question: '{q}'")

    '''checker = SentenceTransformerQuestionChecker()
    is_valid, score, sim_valid, sim_invalid = checker.is_valid(q)

    print(f"ST-based check: {is_valid}, score: {score}, sim_valid: {sim_valid}, sim_invalid: {sim_invalid}  for question: '{q}'")
    if not is_valid:
        return False
        '''

    if llm is not None:
        return llm.llm_semantic_question_check(q)

    return False


def semantic_similarity(pred, gold):
    '''
    Semantic similarity between predicted and gold answer using cosine similarity of sentence embeddings from a pre-trained model (all-MiniLM-L6-v2). (answer-level evaluation)
    This captures the overall semantic closeness of the generated answer to the ground truth answer, beyond just token overlap. It can recognize when two answers are saying the same thing in different words, 
    which is important for evaluating generative models where exact token matches may be less common.
    
    This works by encoding both the predicted answer and the gold answer into dense vector representations using the SentenceTransformer model, and then computing the cosine similarity between these two vectors.
    Works better than simple token overlap metrics like F1 because it can capture semantic equivalence even when there are few or no shared tokens, as long as the overall meaning is similar.
    However, it may not be as interpretable as token-based metrics and can sometimes give high similarity scores to answers that are semantically related but not actually correct. Therefore, it's best used in conjunction with other metrics like F1 for a more comprehensive evaluation.

    More useful for larger answers where there is more content to capture semantic meaning, and less useful for very short answers where token overlap may be more indicative of correctness.

    :param pred: Answer generated by the RAG system for a given question from K-chunks. This is generated by a LLM (Google Flan-T5-Base) using retrieved K chunks as context.
    :param gold: ground truth from a set of questions with known answers. Questions are generatred using SLM (Google Flan-T5-Base) supplying random retrieved chunks as context
    ground truth answers are answers generated by the same model (Flan-T5-Base) with the same question but with gold retrieved chunks as context. This way we have a "best possible answer" for each question 
    that is still generated by the same model and not human annotated, which would be unfairly harsh.

    '''
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gold = model.encode(gold, convert_to_tensor=True)

    return float(
        cosine_similarity(
            emb_pred.cpu().numpy().reshape(1, -1),
            emb_gold.cpu().numpy().reshape(1, -1)
        )[0][0]
    )

STOPWORDS = set([
    "the","a","an","and","or","of","to","in","on","for","with","is","are","was","were",
    "that","this","it","as","by","from","at","be","which","into","their","its"
])

stemmer = PorterStemmer()

def meaningful_tokens(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return {stemmer.stem(t) for t in tokens}

def relevance_score(answer, chunk):
    ans_tokens = meaningful_tokens(answer)
    chunk_tokens = meaningful_tokens(chunk)
    
    overlap = ans_tokens & chunk_tokens
    count = len(overlap)

    if count == 0:
        return 0
    elif count <= 2:
        return 1
    elif count <= 5:
        return 2
    else:
        return 3

def ndcg_at_k_from_chunks(answer,retrieved_chunks, k=0):
    '''NDCG@K based on relevance scores computed from token overlap between the generated answer and the retrieved chunk text. (retrieval-level evaluation)
    Advantage over precision@k is that it gives more weight to chunks that are more relevant (have higher token overlap) rather than treating all supporting chunks equally. 
    This way we can capture not just whether a chunk supports the answer, but how strongly it supports it based on content overlap.
    # Sort by rank (in case input isn't sorted)
    # 1. remove stopwords from answer and chunk text
    # 2. use PorterStemmer to stem tokens
    # 2. compute relevance score based on overlap
    # 3. compute DCG and IDCG
    '''
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x["score_rrf"], reverse=True)
    if k==0:
        k= len(retrieved_chunks)
    dcg = 0.0
    relevance_scores = {}
    for i, chunk in enumerate(retrieved_chunks[:k]):
        r_score = relevance_score(answer, chunk["text"])
        relevance_scores[chunk["chunk_id"]] = r_score
        dcg += r_score / math.log2(i + 2)  # i+2 because rank starts at 1
    # Ideal DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def find_supporting_chunks(answer, ranked_chunks, k, min_overlap=2):
    """
    Identify which of the top-K chunks support the generated answer.
    """
    answer_tokens = meaningful_tokens(answer)
    top_k = ranked_chunks[:k]

    supporting = []

    for cid, text in top_k:
        chunk_tokens = meaningful_tokens(text)
        overlap = answer_tokens & chunk_tokens

        if len(overlap) >= min_overlap:
            supporting.append(cid)

    return set(supporting)

def precision_at_k(ranked_chunk_ids, supporting_chunk_ids, k):
    """
    precision@k for supporting chunks in the top-K retrieved chunks.(retrieval-level evaluation)

    Here we check how many of the top-K retrieved chunks are actually supporting the answer (based on token overlap) and compute precision accordingly.(count of supporting chunks in top-K / K)
    
    :param ranked_chunk_ids: List of chunk IDs in descending order of relevance.
    :param supporting_chunk_ids: Set of chunk IDs that support the answer.
    :param k: Number of top chunks to consider for precision calculation.
    """
    top_k = ranked_chunk_ids[:k]
    hits = sum(1 for cid in top_k if cid in supporting_chunk_ids)
    return hits / k if k > 0 else 0.0

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def answer_f1(pred, gold):
    '''
    F1 score between ground truth and predicted answer based on token overlap after normalization.(answer-level evaluation)
    
    :param pred: Answer generated by the RAG system for a given question. This is generated by a LLM (Google Flan-T5-Base) using retrieved K chunks as context.
    :param gold: ground truth from a set of questions with known answers. Questions are generatred using SLM (Google Flan-T5-Base) supplying random retrieved chunks as context
    ground truth answers are answers generated by the same model (Flan-T5-Base) with the same question but with gold retrieved chunks as context. This way we have a "best possible answer" for each question 
    that is still generated by the same model and not human annotated, which would be unfairly harsh.

    The random questions and answers generated, ensure no human bias, and when compared with the RAG generated answer, we can see how close the RAG system gets to the best possible answer at k chunks that the model 
    can generate given perfect retrieval.
    
    Overlap of tokens is computed after normalization which includes:
    - Lowercasing, stopword removal, stemming, and punctuation removal. This is to ensure that we are comparing the core content of the answers rather than surface-level differences.


    '''
    pred_tokens = meaningful_tokens(pred)
    gold_tokens = meaningful_tokens(gold)

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

    best_url_chunk = retrieved_url_by_rank[best_rank-1] if best_rank else None

    for rank, url in enumerate(retrieved_url_by_rank):
        if url == best_url_chunk:
            best_rank_url = rank + 1
            break

    return best_chunk, best_overlap, 1/best_rank_url if best_rank_url else 0.0, retrieved_url_by_rank[best_rank_url-1] if best_rank_url else None


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
        time_start = time.time()
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
        time_end = time.time()
        response_time = time_end - time_start
        print(f"Response time for question {i+1}: {response_time:.2f}s")
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

        #5. ndcg@k

        semantic_similarity_score = semantic_similarity(pred_answer, gold_answer)
        print(f"Semantic Similarity Score: {semantic_similarity_score:.4f}\n")

        ndcg_score = ndcg_at_k_from_chunks(pred_answer, retrieved_chunks)
        print(f"NDCG@{len(retrieved_chunks)} Score: {ndcg_score:.4f}\n")
        
        ndcg_text = f"NDCG@{len(retrieved_chunks)}"

        #6 precision@k
        supporting_chunks = find_supporting_chunks(pred_answer, [(c["chunk_id"], c["text"]) for c in retrieved_chunks], k=len(retrieved_chunks))
        precision_at_k_score = precision_at_k([c["chunk_id"] for c in retrieved_chunks], supporting_chunks, k=len(retrieved_chunks))
        print(f"Precision@{len(retrieved_chunks)}: {precision_at_k_score:.4f}\n")
        precision_text = f"Precision@{len(retrieved_chunks)}"

        results.append({
            "id": i,
            "question": question_text,
            "pred_answer": pred_answer,
            "gold_answer": gold_answer,
            "f1": f1,
            "semantic_similarity": semantic_similarity_score,
            "question_type": q_type,
            "retrieved_urls": retrieved_url_by_rank,
            "gold_urls": gold_urls,
            "reciprocal_rank": mrr_url_level(gold_urls,retrieved_url_by_rank),
            ndcg_text: ndcg_score,
            "supporting_chunk": best_chunk["chunk_id"] if best_chunk else None,
            precision_text: precision_at_k_score,
            "response_time": response_time
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

    ndcg_text=f"NDCG@{len(results[0]['retrieved_urls'])}" if results else "NDCG@0"
    precision_at_k_text=f"Precision@{len(results[0]['retrieved_urls'])}" if results else "Precision@0"
    # Build DataFrame
    df = pd.DataFrame([
        {
            "ID": row["id"],
            "question": row["question"],
            "ground_truth": row["gold_answer"],
            "F1": row["f1"],
            "Semantic Similarity": row["semantic_similarity"],
            "RR": row["reciprocal_rank"],
            ndcg_text: row.get(f"NDCG@{len(row['retrieved_urls'])}", 0.0),
            precision_at_k_text: row.get(precision_at_k_text, 0.0),
            "Type": row["question_type"],
            "response_time": row["response_time"]
        }
        for row in results[:limit]
    ])

    # Summary statistics
    mean_f1 = sum(r["f1"] for r in results) / len(results)
    mean_rr = sum(r["reciprocal_rank"] for r in results) / len(results)
    mean_ndcg = sum(r[ndcg_text] for r in results) / len(results) if results else 0.0
    mean_semantic_similarity = sum(r["semantic_similarity"] for r in results) / len(results)
    mean_response_time = sum(r["response_time"] for r in results) / len(results) if results else 0.0
    mean_precision_at_k = sum(r.get(precision_at_k_text, 0.0) for r in results) / len(results) if results else 0.0

    summary = {
        "mean_f1": mean_f1,
        "mean_rr": mean_rr,
        ndcg_text: mean_ndcg,
        "mean_semantic_similarity": mean_semantic_similarity,
        "mean_response_time": mean_response_time,
        precision_at_k_text: mean_precision_at_k
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


if __name__ == "__main__":
    # Example usage:
    results, mrr = evaluate_rag( n=1, path=EVAL_QUESTIONS)
    print(results)
    print(f"Final MRR: {mrr}")


    table = load_results_jsonl()
    #print_table(table[:50],50)   # preview first 5 rows
    df, summary = results_to_dataframe(table)
    print(f"Mean F1: {summary['mean_f1']:.4f}, Mean RR: {summary['mean_rr']:.4f}, Mean NDCG: {summary.get('NDCG@3', 0.0):.4f}, Mean Precision@3: {summary.get('Precision@3', 0.0):.4f}")
    print(df.head())


