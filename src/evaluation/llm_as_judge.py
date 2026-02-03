import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.generator import Generator


QUESTIONS_FILE = "/content/drive/MyDrive/Colab Notebooks/rag-hybrid-wiki-main/data/generated_questions.jsonl"

JUDGE_PROMPT = """
You are an expert evaluator for a question answering system.

Question:
{question}

Ground Truth Answer:
{gold}

Model Answer:
{pred}

Score the model answer from 1 (very poor) to 5 (excellent) on:

1. Factual Accuracy
2. Completeness
3. Relevance
4. Coherence

Return ONLY valid JSON in this format:
{{
  "accuracy": <int>,
  "completeness": <int>,
  "relevance": <int>,
  "coherence": <int>,
  "explanation": "<short explanation>"
}}
"""


# ---------------------------
# Load LLM Judge
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


def judge_answer(question, gold, pred):
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        pred=pred
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=256)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        parsed = json.loads(text)


        if isinstance(parsed, dict) and all(
            k in parsed for k in ["accuracy", "completeness", "relevance", "coherence"]
        ):
            return parsed

        else:
            raise ValueError("JSON is not a valid score object")

    except Exception:
        return {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "coherence": 0,
            "explanation": f"Invalid judge output: {text}"
        }



# ---------------------------
# Main Evaluation
# ---------------------------
def run_llm_judge(n=20):
    dense = DenseRetriever()
    sparse = SparseRetriever(index_path="both")
    generator = Generator()

    results = []

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            q = json.loads(line)

            query = q["question"]
            gold_answer = q["ground_truth"]
            print(gold_answer)

            # Hybrid retrieval
            chunks = retrieve_hybrid(query, dense, sparse, top_n=3)

            # Generate answer
            pred_answer = generator.generate(query, chunks)
            print(pred_answer)

            # LLM judge
            scores = judge_answer(query, gold_answer, pred_answer)

            results.append({
                "id": i,
                "question_type": q["question_type"],
                "accuracy": scores["accuracy"],
                "completeness": scores["completeness"],
                "relevance": scores["relevance"],
                "coherence": scores["coherence"],
                "explanation": scores["explanation"]
            })

    return results

import pandas as pd
results = run_llm_judge(n=50)
df = pd.DataFrame(results)

print("\n=== LLM-as-Judge Summary ===")
print(df.mean(numeric_only=True))
print("\nBy Question Type:")
print(df.groupby("question_type").mean(numeric_only=True))