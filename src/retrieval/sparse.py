import os,sys
os.chdir("..")
from src.corpus.bm25_embed import search_bm25

class SparseRetriever:
    def __init__(self, index_path="both"):
        self.index_path = index_path

    def retrieve(self, query, top_k=10):
        results=[]
        results = search_bm25(query, top_k, self.index_path)
        print(f"SparseRetriever: {len(results)} results for query: '{query}' top_k={top_k}")
        # Normalize output to match dense retriever format
        normalized = []
        for r in results:
            normalized.append({
                "chunk_id": r["chunk_uid"],     # rename for consistency
                "text": r["text"],
                "score_sparse": r["score"],
                "metadata": r.get("metadata", {})
            })

        return normalized