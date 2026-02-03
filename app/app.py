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
from src.evaluation.eval_MRR import answer_from_which_chunk_rank
from src.evaluation.eval_MRR import chunk_id_to_url
from src.evaluation.eval_MRR import load_results_jsonl, results_to_dataframe

st.title("Group 15 Hybrid RAG over Wikipedia")

@st.cache_resource
def load_components():
    dense = DenseRetriever()
    sparse = SparseRetriever(index_path="both")  # reuse same chunks
    gen = Generator()
    return dense, sparse, gen
#streamlit reruns the entire script on every interaction, so we use session_state to store MRR and num
if "MRR" not in st.session_state:
    st.session_state.MRR = 0.0

if "num" not in st.session_state:
    st.session_state.num = 0

if "eval_results" not in st.session_state:
    st.session_state.eval_results = load_results_jsonl()    

dense_ret, sparse_ret, generator= load_components()
df, summary = results_to_dataframe(st.session_state.eval_results)
st.dataframe(df)
st.caption(f"Mean F1: {summary['mean_f1']:.4f}, Mean RR: {summary['mean_rr']:.4f}")
query = st.text_input("Enter your question")

top_k = st.slider("Top-K per retriever", 3, 40, 20)
top_n = st.slider("Top-N after RRF", 1, 5, 3)

if st.button("Ask") and query:
    t0 = time.time()
    dense_results = dense_ret.retrieve(query, top_k=top_k)
    sparse_results = sparse_ret.retrieve(query, top_k=top_k)
    fused = retrieve_hybrid(query, dense_ret, sparse_ret, k_dense=top_k, k_sparse=top_k, top_n=top_n)
    answer = generator.generate(query, fused)
    best_chunk, overlap, reciprocal_rank, best_url = answer_from_which_chunk_rank(answer, fused)
    elapsed = time.time() - t0

    print(f"num: {st.session_state.num}, MRR: {st.session_state.MRR}")

    st.session_state.num += 1
    st.session_state.MRR = (st.session_state.MRR * (st.session_state.num - 1)+ reciprocal_rank)/ st.session_state.num

    print(f"num: {st.session_state.num}, MRR: {st.session_state.MRR}")


    st.caption(f"Mean Reciprocal Rank: {st.session_state.MRR:.4f}")

    st.subheader("Answer")
    st.write(answer)
    st.markdown(f"**Best supporting chunk ID:** {best_chunk['chunk_id'] if best_chunk else 'None'}  \n**Overlap:** {overlap}  \n**Best URL contributing to answer:** {best_url}  \n**Reciprocal Rank:** {reciprocal_rank:.4f}")
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
            "url": chunk_id_to_url(c["chunk_id"])
        })