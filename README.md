# rag-hybrid-wiki
Hybrid RAG  implementation

src\corpus\url_sampling.py  > connect to Wikipedia and get 200 wiki links (fixed) currently limit set to 20
src\corpus\fetch_wikipedia.py > fetch text store it with the pagename
src\corpus\clean_text.py > basic cleaning saving cleaned text
src\corpus\chunker.py > as asked using sentence chunking to create chunked<uid>.txt and json with headers and metadata for BM25
src\corpus\embed.py > dense embedding creates jsonl file that can be used by any vector db
src\corpus\bm25_embed.py > bm25_index.json of files created




What is Mathematics ?
A common approach is to define mathematics by its object of study.
{"score_dense":0.661025881767273,"score_sparse":10.659591806371015,"score_rrf":0.03128054740957967,"chunk_id":"18831_chunk_42"}

why study mathematics ?
for the needs of empirical sciences and mathematics itself

{"score_dense":0.6120296120643616,"score_sparse":9.12672757018377,"score_rrf":0.032266458495966696,"chunk_id":"18831_chunk_0"}

Why do we study history?
to uncover the truth
{"score_dense":0.6715508699417114,"score_sparse":9.842688918503717,"score_rrf":0.031099324975891997,"chunk_id":"10772350_chunk_0"}

What is Philosophy?
The study of the biggest patterns of the world as a whole or as the attempt to answer the big questions
{"score_dense":0.6386522054672241,"score_sparse":11.410130454903936,"score_rrf":0.030798389007344232,"chunk_id":"13692155_chunk_4"}

what is difference between pure and applied mathematics ?
In the present day, the distinction between pure and applied mathematics is more a question of personal research aim of mathematicians than a division of mathematics into broad areas.

{"score_dense":0.6296842098236084,"score_sparse":22.891224726020035,"score_rrf":0.03200204813108039,"chunk_id":"18831_chunk_32"}


For how many years the President of United States of America elected?
4
{"score_dense":0.5404127836227417,"score_sparse":43.4176295169535,"score_rrf":0.03278688524590164,"chunk_id":"10826158_chunk_3"}

what is the term in office for the president of united states of america?
four-year term
{"score_dense":0.45881953835487366,"score_sparse":41.6887034324625,"score_rrf":0.03225806451612903,"chunk_id":"10826158_chunk_3"}


What is music?
the arrangement of sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content

{"score_dense":0.6873643398284912,"score_sparse":12.0459325277828,"score_rrf":0.03200204813108039,"chunk_id":"18839_chunk_0"}

Define  Geography.

The core concepts of geography consistent between all approaches are a focus on space, place, time, and scale.

{"score_dense":0.6818385720252991,"score_sparse":8.774100860738717,"score_rrf":0.03225806451612903,"chunk_id":"18963910_chunk_1"}

what are the different branches of philosophy?

epistemology, ethics, logic, and metaphysics
{"score_dense":0.5519967079162598,"score_sparse":22.53826758351399,"score_rrf":0.03131881575727918,"chunk_id":"13692155_chunk_1"}

what  is physics?
a branch of fundamental science

{"score_dense":0.5634846687316895,"score_sparse":11.786887068572872,"score_rrf":0.030886196246139225,"chunk_id":"22939_chunk_20"}

