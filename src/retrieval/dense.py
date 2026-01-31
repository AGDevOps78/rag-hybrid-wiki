
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os,sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
emb_path = os.path.join(ROOT, "data", "embeddings_merged.jsonl")
print(f"Setting EMB_PATH to: {emb_path}")
EMB_PATH = emb_path
print
class DenseRetriever:
    def __init__(self, emb_path=EMB_PATH, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = []
    
        with open(emb_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(obj)
                self.embeddings.append(obj["embedding"])

        self.embeddings = np.array(self.embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=10):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            item = self.chunks[idx]
            results.append({
                "chunk_id": item.get("chunk_id", item.get("chunk_uid")),
                "text": item.get("text", ""),
                "score_dense": float(score)
            })
        return results