# src/retrieval/hybrid.py
def rrf_fusion(dense_results, sparse_results, k=60, top_n=10):
    # build rank maps
    dense_rank = {r["chunk_id"]: i for i, r in enumerate(dense_results)}
    sparse_rank = {r["chunk_id"]: i for i, r in enumerate(sparse_results)}

    all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
    fused = {}

    for cid in all_ids:
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (k + dense_rank[cid] + 1)
        if cid in sparse_rank:
            score += 1.0 / (k + sparse_rank[cid] + 1)
        fused[cid] = score

    # build merged objects with scores from both sides
    by_id_dense = {r["chunk_id"]: r for r in dense_results}
    by_id_sparse = {r["chunk_id"]: r for r in sparse_results}

    merged = []
    for cid, score_rrf in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        d = by_id_dense.get(cid, {})
        s = by_id_sparse.get(cid, {})
        merged.append({
            "chunk_id": cid,
            "text": d.get("text") or s.get("text", ""),
            "score_dense": d.get("score_dense"),
            "score_sparse": s.get("score_sparse"),
            "score_rrf": score_rrf,
        })
    return merged

def retrieve_hybrid(query, dense_ret, sparse_ret, k_dense=60, k_sparse=60, top_n=10):
    dense_results = dense_ret.retrieve(query, top_k=k_dense)
    sparse_results = sparse_ret.retrieve(query, top_k=k_sparse)
    return rrf_fusion(dense_results, sparse_results, top_n=top_n)