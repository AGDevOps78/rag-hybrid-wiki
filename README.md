# rag-hybrid-wiki
Hybrid RAG  implementation

src\corpus\url_sampling.py  > connect to Wikipedia and get 200 wiki links (fixed) currently limit set to 20
src\corpus\fetch_wikipedia.py > fetch text store it with the pagename
src\corpus\clean_text.py > basic cleaning saving cleaned text
src\corpus\chunker.py > as asked using sentence chunking to create chunked<uid>.txt and json with headers and metadata for BM25
src\corpus\embed.py > dense embedding creates jsonl file that can be used by any vector db
src\corpus\bm25_embed.py > bm25_index.json of files created

