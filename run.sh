#!/bin/bash

set -e  # Exit on error
echo "▶ Installing dependencies..."
pip install -r requirements.txt

echo "=============================="
echo " Hybrid RAG Pipeline Started "
echo "=============================="

# -----------------------------
# 1. Dataset Creation
# -----------------------------
echo "▶ Sampling Wikipedia URLs..."
python src/corpus/url_sampling.py

echo "▶ Fetching Wikipedia content..."
python src/corpus/fetch_wikipedia.py

echo "▶ Cleaning fixed dataset..."
python src/corpus/clean_text.py \
  --input data/cleaned_text \
  --output data/cleaned_text_final

echo "▶ Cleaning random dataset..."
python src/corpus/clean_text.py \
  --input data_random/cleaned_text \
  --output data_random/cleaned_text_final

echo "▶ Chunking documents..."
python src/corpus/chunker.py

# -----------------------------
# 2. Index Construction
# -----------------------------
echo "▶ Creating dense embeddings..."
python src/corpus/embed.py

echo "▶ Creating BM25 index..."
python src/corpus/bm25_embed.py

echo "▶ Merging embeddings..."
python src/corpus/merge_embedings.py

# -----------------------------
# 3. Question Generation
# -----------------------------
echo "▶ Generating evaluation questions..."
python src/evaluation/call_gen_questions.py

# -----------------------------
# 4. Evaluation
# -----------------------------
echo "▶ Running MRR evaluation..."
python src/evaluation/eval_MRR.py

echo "▶ Running ablation study..."
python src/evaluation/ablation.py

echo "▶ Running LLM-as-Judge study..."
python src/evaluation/llm_as_judge.py
# -----------------------------
# 5. Done
# -----------------------------
echo "=============================="
echo " Pipeline Completed Successfully "
echo "=============================="
