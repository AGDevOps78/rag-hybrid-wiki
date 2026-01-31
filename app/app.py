# app.py
import time
import streamlit as st

import os, sys

# Compute project root (one level above this file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.generator import Generator

st.title("Group 15 Hybrid RAG over Wikipedia")

@st.cache_resource
def load_components():
    dense = DenseRetriever()
    sparse = SparseRetriever(index_path="both")  # reuse same chunks
    gen = Generator()
    return dense, sparse, gen

dense_ret, sparse_ret, generator = load_components()

query = st.text_input("Enter your question")

top_k = st.slider("Top-K per retriever", 3, 40, 20)
top_n = st.slider("Top-N after RRF", 1, 5, 3)

if st.button("Ask") and query:
    t0 = time.time()
    dense_results = dense_ret.retrieve(query, top_k=top_k)
    sparse_results = sparse_ret.retrieve(query, top_k=top_k)
    fused = retrieve_hybrid(query, dense_ret, sparse_ret, k_dense=top_k, k_sparse=top_k, top_n=top_n)
    answer = generator.generate(query, fused)
    elapsed = time.time() - t0

    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Response time: {elapsed:.2f}s")

    st.subheader("Top retrieved chunks (RRF)")
    for i, c in enumerate(fused, 1):
        st.markdown(f"**Chunk {i}**")
        st.write(c["text"])
        st.json({
            "score_dense": c["score_dense"],
            "score_sparse": c["score_sparse"],
            "score_rrf": c["score_rrf"],
            "chunk_id": c["chunk_id"],
        })