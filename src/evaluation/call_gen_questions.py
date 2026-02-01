import os
import sys
import json

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
from src.evaluation.get_chunks import load_random_chunks   

# -------------------------------------------------------------------
# Load retrievers
# -------------------------------------------------------------------
dense = DenseRetriever()
sparse = SparseRetriever(index_path="both")

# -------------------------------------------------------------------
# Load 50 random chunks from data/chunks
# -------------------------------------------------------------------
CHUNK_DIR = os.path.join(ROOT, "data", "chunks")
CHUNK_DIR_RANDOM = os.path.join(ROOT, "data_random", "chunks")
chunks = load_random_chunks(CHUNK_DIR, n=120)
chunks += load_random_chunks(CHUNK_DIR_RANDOM, n=100)


print(f"Loaded {len(chunks)} random chunks for question generation.")

# -------------------------------------------------------------------
# Create question generator
# -------------------------------------------------------------------
qg = QGenerator(dense, sparse)

# -------------------------------------------------------------------
# Generate 100 Q&A pairs
# -------------------------------------------------------------------
questions = qg.generate_dataset(chunks, target_count=20)
# -------------------------------------------------------------------
# Save output
# -------------------------------------------------------------------
OUTPUT_PATH = os.path.join(ROOT, "data", "generated_questions.jsonl")

qg.save(questions, OUTPUT_PATH)

print("Generated 100 Q&A pairs.")
print(f"Saved to: {OUTPUT_PATH}")