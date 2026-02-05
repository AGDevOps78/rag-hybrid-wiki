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
from src.evaluation.eval_MRR import answer_from_which_chunk_rank,ndcg_at_k_from_chunks,precision_at_k,find_supporting_chunks
from src.evaluation.eval_MRR import chunk_id_to_url
from src.evaluation.eval_MRR import load_results_jsonl, results_to_dataframe

st.title("Group 15 Hybrid RAG over Wikipedia")

import plotly.express as px
import pandas as pd
import streamlit as st

def bar_chart_results(df):
    # df must contain: ["Type", "F1", "RR"]

    # Melt the DataFrame so F1 and RR become separate metric rows
    
    df_grouped = (df.groupby("Type")[["F1", "RR", "NDCG@3", "Precision@3"]]
                    .mean()
                    .reset_index()
                )


    df_melted = df_grouped.melt(
    id_vars="Type",
    value_vars=["F1", "RR", "NDCG@3", "Precision@3"],
    var_name="Metric",
    value_name="Score"
    )

    fig = px.bar(
    df_melted,
    x="Type",
    y="Score",
    color="Metric",
    barmode="group",
    title="Metrics by Question Type",
    text_auto=".3f"
    )

    fig.update_layout(
    xaxis_title="Question Type",
    yaxis_title="Score",
    legend_title="Metric",
    bargap=0.25
    )

    return fig



@st.cache_resource
def load_components():
    dense = DenseRetriever()
    sparse = SparseRetriever(index_path="both")  # reuse same chunks
    gen = Generator()
    return dense, sparse, gen
#streamlit reruns the entire script on every interaction, so we use session_state to store MRR and num
if "MRR" not in st.session_state:
    st.session_state.MRR = 0.0

if "NDCG" not in st.session_state:
    st.session_state.NDCG = 0.0

if "num" not in st.session_state:
    st.session_state.num = 0

if "precision" not in st.session_state:
    st.session_state.precision = 0.0

if "eval_results" not in st.session_state:
    st.session_state.eval_results = load_results_jsonl() 

if "avg_response_time" not in st.session_state:
    st.session_state.avg_response_time = 0.0

dense_ret, sparse_ret, generator= load_components()
df, summary = results_to_dataframe(st.session_state.eval_results)
st.dataframe(df)
st.caption(summary)
fig = bar_chart_results(df)
st.plotly_chart(fig)
query = st.text_input("Enter your question")

top_k = st.slider("Top-K per retriever", 3, 10, 5)
top_n = st.slider("Top-N after RRF", 1, 5, 3)

if st.button("Ask") and query:
    t0 = time.time()
    dense_results = dense_ret.retrieve(query, top_k=top_k)
    sparse_results = sparse_ret.retrieve(query, top_k=top_k)
    fused = retrieve_hybrid(query, dense_ret, sparse_ret, k_dense=top_k, k_sparse=top_k, top_n=top_n)
    answer = generator.generate(query, fused)
    best_chunk, overlap, reciprocal_rank, best_url = answer_from_which_chunk_rank(answer, fused)
    ndcg = ndcg_at_k_from_chunks(answer, fused, k=top_n)
    supporting_chunks= find_supporting_chunks(answer, [(c["chunk_id"], c["text"]) for c in fused], k=top_n)
    precision = precision_at_k([c["chunk_id"] for c in fused], supporting_chunks, k=top_n)
    elapsed = time.time() - t0

    print(f"num: {st.session_state.num}, MRR: {st.session_state.MRR}")

    st.session_state.num += 1
    st.session_state.MRR = (st.session_state.MRR * (st.session_state.num - 1)+ reciprocal_rank)/ st.session_state.num
    st.session_state.NDCG = (st.session_state.NDCG * (st.session_state.num - 1)+ ndcg)/ st.session_state.num
    st.session_state.precision = (st.session_state.precision * (st.session_state.num - 1)+ precision)/ st.session_state.num
    st.session_state.avg_response_time = (st.session_state.avg_response_time * (st.session_state.num - 1)+ elapsed)/ st.session_state.num
    print(f"num: {st.session_state.num}, MRR: {st.session_state.MRR}, NDCG: {st.session_state.NDCG}")


    st.caption(f"Mean Reciprocal Rank (URL): {st.session_state.MRR:.4f} over {st.session_state.num} questions")
    st.caption(f"Mean NDCG: {st.session_state.NDCG:.4f} over {st.session_state.num} questions at top {top_n} retrieved chunks")
    st.caption(f"Mean Precision@{top_n}: {st.session_state.precision:.4f} over {st.session_state.num} questions")
    st.caption(f"Mean Response Time: {st.session_state.avg_response_time:.2f}s over {st.session_state.num} questions")

    st.caption(f"View information source: {best_url}")

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