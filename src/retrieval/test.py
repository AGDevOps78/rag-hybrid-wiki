import os,sys
os.chdir("..")
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.generator import Generator
from src.retrieval.hybrid import retrieve_hybrid

# Load retrievers once
dense_ret = DenseRetriever()
sparse_ret = SparseRetriever(index_path="both")

# Query
query = "what is the difference between the two subfields of philosophy?"
results = retrieve_hybrid(query, dense_ret, sparse_ret, top_n=5)
# LLM generation
answer = Generator().generate(query, results)

# Print answer
print("\n=== Generated Answer ===\n")
print(answer)

# Print retrieved chunks + RRF scores
print("\n=== Retrieved Chunks ===\n")
for r in results:
    print(r["chunk_id"], r["score_rrf"])
