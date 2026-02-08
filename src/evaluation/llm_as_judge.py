import os
import sys
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------------------------------------------------
# Ensure project root is on sys.path
# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

# -------------------------------------------------------------------
# Imports AFTER sys.path is fixed
# -------------------------------------------------------------------
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.qgenerator import QGenerator
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.generator import Generator
from src.evaluation.get_chunks import load_random_chunks   
from src.evaluation.eval_MRR import ndcg_at_k_from_chunks, semantic_similarity,chunk_id_to_url, mrr_url_level

# -------------------------------------------------------------------
# Load retrievers
# -------------------------------------------------------------------
dense = DenseRetriever()
sparse = SparseRetriever(index_path="both")

print("Loaded dense and sparse retrievers.")
# -------------------------------------------------------------------
# Load 50 random chunks from data/chunks
# -------------------------------------------------------------------
CHUNK_DIR = os.path.join(ROOT, "data", "chunks")
CHUNK_DIR_RANDOM = os.path.join(ROOT, "data_random", "chunks")
chunks = load_random_chunks(CHUNK_DIR, n=120)
chunks += load_random_chunks(CHUNK_DIR_RANDOM, n=100)


print(f"Loaded {len(chunks)} random chunks for question generation.")

def eval_response(gold, response, results):
    # 1. semantic similarity
    similarity_score = semantic_similarity(response, gold)
    print(f"\nSemantic similarity between generated answer and ground truth: {similarity_score:.4f}")

    # 2. NDCG
    ndcg_score = ndcg_at_k_from_chunks(gold, results, len(results))
    print(f"NDCG@3 score for retrieved chunks against ground truth: {ndcg_score:.4f}")

    # 4. Combined score
    score = ndcg_score * similarity_score

    if score < 0.3:
        print(f"response generated poor {score}")
    elif score < 0.5:
        print(f"fair response = {score}")
    elif score < 0.8:
        print(f"good response score = {score}")
    else:
        print(f"excellent response score = {score}")
    return score



def llm_as_judge(num=5):  
    # -------------------------------------------------------------------
    # Create question generator
    # -------------------------------------------------------------------
    qg = QGenerator(dense, sparse)

    # -------------------------------------------------------------------
    # Generate a question and answer pair for Judge evaluation
    # -------------------------------------------------------------------
    print(f"LLM as a judge generating {num} evaluation questions")
    questions = qg.generate_dataset(chunks, target_count=num)
    print(f"Generated a Q&A pair for LLM-as-Judge evaluation: {questions[0]['question']} -> {questions[0]['ground_truth']} {questions[0]['wikipedia_url']} ")

    # -------------------------------------------------------------------
    # Use the generated question to retrieve chunks and generate an answer, which will be compared to the ground truth
    score = 0.0
    scores =[]
    print(f"LLM as a Judge using {num} generated questions to judge Group 15's retrieval system")
    for i, question in enumerate(questions):
        query = question['question']
        results = retrieve_hybrid(query, dense, sparse, top_n=3)
        # LLM generation
        answer = Generator().generate(query, results)

        # Print answer
        print("\n=== Generated Answer ===\n")
        print(f"question {query} \n answer: {answer}")

        temp = eval_response(question['ground_truth'],answer,results)
        scores.append(temp)
        score = score+temp
    avgscore = score/len(questions)
    print(f"avg score for {len(questions)} question(s) = {avgscore})")
    print(f"scores: {scores}")
    return avgscore


if __name__ == "__main__":
    numQ =10 
    score = llm_as_judge(numQ)
    print(f"LLM as a Judge verdict:")
    if score < 0.3:
        print(f"response generated poor {score} for {numQ} questions evaluated")
    elif score < 0.5:
        print(f"fair response = {score} for {numQ} questions evaluated")
    elif score < 0.8:
        print(f"good response score = {score} for {numQ} questions evaluated")
    else:
        print(f"excellent response score = {score} for {numQ} questions evaluated")