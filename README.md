# rag-hybrid-wiki
Hybrid RAG  implementation

1. Dataset Requirements
rag-hybrid-wiki-main/src/corpus/url_sampling.py  > connect to Wikipedia and get 200 wiki links (data\fixed_urls.json) if less than 200, Get 300 random links if required
..rag-hybrid-wiki-main> python -m src.corpus.url_sampling
rag-hybrid-wiki-main/src/corpus/fetch_wikipedia.py > fetch text store it with the pagename, if "data/embeddings.jsonl" for fixed url present it will not do this for 200 fixed url.. however due to zip size limit its not in the zip
..rag-hybrid-wiki-main> python -m src.corpus.fetch_wikipedia     
rag-hybrid-wiki-main/src\corpus/clean_text.py > basic cleaning saving cleaned text
..rag-hybrid-wiki-main>  
..rag-hybrid-wiki-main> python -m src.corpus.clean_text --input data_random\cleaned_text --output data_random/cleaned_text_final  
rag-hybrid-wiki-main/src/corpus/chunker.py > as asked using sentence chunking to create chunked<uid>.txt and json with headers and metadata for BM25
..rag-hybrid-wiki-main> python -m src.corpus.chunker     
rag-hybrid-wiki-main/src/corpus/embed.py > dense embedding creates jsonl file that can be used by any vector db
..rag-hybrid-wiki-main> python -m src.corpus.embed
rag-hybrid-wiki-main/src/corpus/bm25_embed.py > bm25_index.json of files created
..rag-hybrid-wiki-main> python -m src.corpus.bm25_embed
rag-hybrid-wiki-main/src/corpus/merge_embedings.py  > run to merge dense indexes data_random\embeddings.jsonl and data\embeddings.jsonl to data\embeddings_merged.jsonl
..rag-hybrid-wiki-main> python -m src.corpus.merge_embedings

2. Hybrid RAG System
rag-hybrid-wiki-main/src/retrieval/dense.py > Dense retrival 
rag-hybrid-wiki-main/src/retrieval/sparse.py > Sparse retrival
rag-hybrid-wiki-main/src/retrieval/hybrid.py > Hybrid - Dense + Sparse retrival
rag-hybrid-wiki-main/src/retrieval/hybrid.py > RRF
rag-hybrid-wiki-main/src/retrieval/generator.py > Response Generator
rag-hybrid-wiki-main/src/retrieval/test.py > Test and give RRF score for all chunk IDs
..rag-hybrid-wiki-main>python -m src.retrieval.test
User Interface

..rag-hybrid-wiki-main\app>streamlit run app.py
3. Automated Evalution
src\evaluation\call_gen_questions.py
from project root run "python -m src.evaluation.call_gen_questions" >  create data\generated_questions.jsonl

rag-hybrid-wiki-main/src/evaluation/eval_MRR.py > MRR evlaution
from project root run "python -m src.evaluation.eval_MRR"

rag-hybrid-wiki-main/src/evaluation/ablation.py > Ablation evlaution
from project root run "python -m src.evaluation.ablation

4. Input validation
from src.evaluation.eval_MRR import  is_valid_question_strict

5. Application 
cd to .\App folder 
run streamlit run app.py


Below are some example reuslts

why is learning public speaking techniques important?
Public speaking is a fundamental component of rhetoric, analyzed by prominent thinkers. Public speaking was extensively studied in Ancient Greece and Ancient Rome, where it was a fundamental component of rhetoric, analyzed by prominent thinkers. Aristotle, the ancient Greek philosopher, identified three types of speeches: deliberative (political), forensic (judicial), and epideictic (ceremonial).[4] Public speaking is frequently directed at a select and sometimes restricted audience, consisting of individuals who may hold different perspectives. This audience can encompass enthusiastic supporters of the speaker, reluctant attendees with opposing views, or strangers with varying levels of interest in the speaker's topic.[5] Public speaking aims to either reassure an anxious audience or to alert a complacent audience of something important. Once the speaker has determined which of these approaches is required, they will use a combination of storytelling and informational approaches to achieve their goals.
chunk2
{"score_dense":0.5833410620689392,"score_sparse":16.43465369850526,"score_rrf":0.03149801587301587,"chunk_id":"25084_chunk_1"}

What is an Atom?
The atom is the basic unit of chemistry. It consists of a dense core called the atomic nucleus surrounded by a space occupied by an electron cloud. The nucleus is made up of positively charged protons and uncharged neutrons (together called nucleons), while the electron cloud consists of negatively charged electrons which orbit the nucleus. In a neutral atom, the negatively charged electrons balance out the positive charge of the protons. The nucleus is dense; the mass of a nucleon is approximately 1,836 times that of an electron, yet the radius of an atom is about 10,000 times that of its nucleus.[21] The atom is also the smallest entity that can be envisaged to retain the chemical properties of the element, such as electronegativity, ionization potential, preferred oxidation state(s), coordination number, and preferred types of bonds to form (e.g., metallic, ionic, covalent).[2][23] The atom is also the smallest entity that can be envisaged to retain the chemical properties of the element, such as electronegat
Chunk 2
{"score_dense":0.6775169372558594,"score_sparse":11.438774049328769,"score_rrf":0.029957522915269395,"chunk_id":"5180_chunk_3"}

what constitutes an atom?
A nucleus of protons and generally neutrons, surrounded by an electron cloud. The nucleus is made up of positively charged protons and uncharged neutrons (together called nucleons), while the electron cloud consists of negatively charged electrons which orbit the nucleus. The nucleus is dense; the mass of a nucleon is approximately 1,836 times that of an electron, yet the radius of an atom is about 10,000 times that of its nucleus.
Chunk 2
{"score_dense":0.7109595537185669,"score_sparse":8.330518731508192,"score_rrf":0.02889344262295082,"chunk_id":"902_chunk_0"}

what is religion ?
Religion is a unique identity variable with immense power.
chunk 3
{"score_dense":0.5141894817352295,"score_sparse":9.355086095971878,"score_rrf":0.03047794966520434,"chunk_id":"62473758_chunk_4"}

Who was "The First Teacher"?
Aristotle was well known among medieval Muslim intellectuals and revered as "The First Teacher" (Arabic: ).
Chunk 3
{"score_dense":0.30910518765449524,"score_sparse":12.07122427064592,"score_rrf":0.03036576949620428,"chunk_id":"32923_chunk_10"}

Why is Aristotle called "The First Teacher"?
Aristotle was widely considered the most pivotal figure in the development of philosophy, especially the Western tradition
chunk 1
{"score_dense":0.5760480761528015,"score_sparse":22.300606721072924,"score_rrf":0.03278688524590164,"chunk_id":"32923_chunk_10"}

What are different categories of art?
media, genre, styles, and form
{"score_dense":0.6272503137588501,"score_sparse":18.115757683356577,"score_rrf":0.03252247488101534,"chunk_id":"752_chunk_15"}

Definition of art.
There is no generally agreed definition of what constitutes art
{"score_dense":0.7314118146896362,"score_sparse":13.058879231893084,"score_rrf":0.03200204813108039,"chunk_id":"752_chunk_0"}

Name a few famous artists.
Michael Fried, T. J. Clark, Rosalind Krauss, Linda Nochlin and Griselda Pollock
{"score_dense":0.5002988576889038,"score_sparse":9.896562092261469,"score_rrf":0.030886196246139225,"chunk_id":"752_chunk_44"}

What  is astro physics?
Unanswerable
 no document on astro physics

What is Albert Einstein famous for?
Albert Einstein developed the theory of relativity that uses fundamentally these concepts.
{"score_dense":0.4562206566333771,"score_sparse":15.633653541997147,"score_rrf":0.03177805800756621,"chunk_id":"18831_chunk_34"}

Who was Albert Einstein?
Albert Einstein was a physicist.
Chunk 3
{"score_dense":null,"score_sparse":14.25087259413689,"score_rrf":0.01639344262295082,"chunk_id":"363430_chunk_1"}

what is Generational contract?
The term generational contract is a concept used in the research of the relations between generations within a society. It refers to an agreement or consensus regarding the roles and mutual responsibilities of different age groups or generations. The term does not define a legal contract, as no enforceable agreement exists between generations.
{"score_dense":0.7071757316589355,"score_sparse":22.723288080304663,"score_rrf":0.03252247488101534,"chunk_id":"81920874_chunk_0"}

What is Mathematics ?
A common approach is to define mathematics by its object of study. Aristotle defined mathematics as "the science of quantity" and this definition prevailed until the 18th century. However, Aristotle also noted a focus on quantity alone may not distinguish mathematics from sciences like physics; in his view, abstraction and studying quantity as a property "separable in thought" from real instances set mathematics apart. In the 19th century, when mathematicians began to address topics—such as infinite sets—which have no clear-cut relation to physical reality, a variety of new definitions were given. With the large number of new areas of mathematics that have appeared since the beginning of the 20th century, defining mathematics by its object of study has become increasingly difficult. For example, in lieu of a definition, Saunders Mac Lane in Mathematics, form and function summarizes the basics of several areas of mathematics, emphasizing their inter-connectedness, and observes: the development of Mathematics provides a tightly connected network of formal rules, concepts, and systems. Nodes of this network are closely bound to procedures useful in human activities and to questions arising in science.
chunk1
{"score_dense":0.661025881767273,"score_sparse":10.659591806371015,"score_rrf":0.03128054740957967,"chunk_id":"18831_chunk_42"}

why study mathematics ?
Mathematics is a field of study that discovers and organizes methods, theories, and theorems that are developed and proved for the needs of empirical sciences and mathematics itself.
Chunk 1
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

what is physics?

Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force.
chunk 2
{"score_dense":0.6238350868225098,"score_sparse":9.583735478424988,"score_rrf":0.029957522915269395,"chunk_id":"1653925_chunk_0"}

How to extend life?



The goal of life extension technology is to combine existing and predicted future biochemical and genetic techniques.

{"score_dense":0.5305856466293335,"score_sparse":10.302588128366931,"score_rrf":0.03149801587301587,"chunk_id":"23607241_chunk_2"}

discuss about Most Favored Nation Drug Pricing?

Most Favored Nation Drug Pricing is a policy advanced during the first and second Trump administrations in which drug prices in the United States are tied to foreign drug prices. Prescription drug prices in the United States are much higher than costs abroad.

{"score_dense":0.7244083285331726,"score_sparse":23.034341426656354,"score_rrf":0.03278688524590164,"chunk_id":"80019332_chunk_0"}

who is a Psychologist?

The term "psychologist" is a scientific term that refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. The term "psychologist" refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. The term "psychologist" refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. The term "psychologist" refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. The term "psychologist" refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. The term "psychologist" refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. The term "psychologist" refers to a professional who practices psychology and studies mental states, perceptual, cognitive, emotional, and social processes and behavior. 
{"score_dense":0.6555368900299072,"score_sparse":15.342330549739547,"score_rrf":0.031754032258064516,"chunk_id":"72132_chunk_0"}


explain the difference between democracy and autocracy

Answer
Democracy is based on popular sovereignty. Autocracies are either dictatorships (including military dictatorships) or absolute monarchies.

Best supporting chunk ID: 16027561_chunk_14
Overlap: 11
Best URL contributing to answer: https://en.wikipedia.org/?curid=16027561
Reciprocal Rank: 1.0000

Response time: 10.51s

Top retrieved chunks (RRF)
Chunk 1

The interactions between different political systems are the basis for global politics. Forms of government can be classified by several ways. In terms of the structure of power, there are monarchies (including constitutional monarchies) and republics (usually presidential, semi-presidential, or parliamentary). The separation of powers describes the degree of horizontal integration between the legislature, the executive, the judiciary, and other independent institutions. The source of power determines the difference between democracies, oligarchies, and autocracies. In a democracy, political legitimacy is based on popular sovereignty. Forms of democracy include representative democracy, direct democracy, and demarchy. These are separated by the way decisions are made, whether by elected representatives, referendums, or by citizen juries.

{
"score_dense":0.5453159809112549
"score_sparse":21.265834853055498
"score_rrf":0.032018442622950824
"chunk_id":"16027561_chunk_14"
"url":"https://en.wikipedia.org/?curid=16027561"
}


Answer quality Semantic similarity score vs F1 score example:

Question: What is the oldest piece of art found in Europe?
Answer:
Response time for question 22: 43.42s

=== Generated Answer ===

The Riesenhirschknochen der Einhornhöhle, dating back 51,000 years and made by Neanderthals, dates back 51,000 years and made by Neanderthals. The first undisputed sculptures and similar art pieces, like the Venus of Hohle Fels, are the numerous objects found at the Caves and Ice Age Art in the Swabian Jura UNESCO World Heritage Site, where the oldest non-stationary works of human art yet discovered were found, in the form of carved animal and humanoid figurines, in addition to the oldest musical instruments unearthed so far, with the artifacts dating between 43,000 and 35,000 BC, so being the first centre of human art. Many great traditions in art have a foundation in the art of one of the great ancient civilizations: Ancient Egypt, Mesopotamia, Persia, India, China, Ancient Greece, Rome, as well as Inca, Maya, and Olmec. Each of these centers of early civilization developed a unique and characteristic style in its art. Because of the size and duration of these civilizations, more of their art works have survived and more of their 

=== Gold Answer ===

The oldest piece of art found in Europe is the Riesenhirschknochen der Einhornhöhle, dating back 51,000 years and made by Neanderthals. ['https://en.wikipedia.org/?curid=752']

F1 Score: 0.2804

Semantic Similarity Score: 0.8378

NDCG@3 Score: 1.0000

Precision@3: 1.0000


Question: Where is Shopska salad considered a national dish of Bulgaria?
Answer:
Response time for question 23: 10.11s

=== Generated Answer ===

Shopska salad is considered a national dish of Bulgaria, is claimed by Bulgaria, Croatia, Czechia, North Macedonia, and Serbia.

=== Gold Answer ===

Bulgaria, Croatia, Czechia, North Macedonia, and Serbia ['https://en.wikipedia.org/?curid=68554701']

Best supporting chunk ID: 68554701_chunk_12, Overlap: 17
F1 Score: 0.6667

Semantic Similarity Score: 0.5187

Retrieval quality example, here we see that ndcg@3 is 0.86 while precision@3 is 1. While all three chunks have relevant text.
ndcg captures the fact that the 1st 2 have the relevant information

Question: What was the school of positivism?
Answer:
Response time for question 61: 9.62s

=== Generated Answer ===

Auguste Comte formulated the school of positivism.

=== Gold Answer ===

Auguste Comte formulated the school of positivism and aimed to discover general laws of history, similar to the laws of nature studied by physicists. ['https://en.wikipedia.org/?curid=10772350']

=== Retrieved Chunks ===

18717981_chunk_9 0.032018442622950824 https://en.wikipedia.org/?curid=18717981
18717981_chunk_11 0.03200204813108039 https://en.wikipedia.org/?curid=18717981
10772350_chunk_40 0.0315136476426799 https://en.wikipedia.org/?curid=10772350
RR for this question: 0.3333333333333333

18717981_chunk_9 0.032018442622950824 https://en.wikipedia.org/?curid=18717981
18717981_chunk_11 0.03200204813108039 https://en.wikipedia.org/?curid=18717981
10772350_chunk_40 0.0315136476426799 https://en.wikipedia.org/?curid=10772350
 best chunk : {'chunk_id': '10772350_chunk_40', 'text': 'In tune with this scientific outlook, Auguste Comte formulated the school of positivism and aimed to discover general laws of history, similar to the laws of nature studied by physicists. Building on the philosophy of Georg Wilhelm Friedrich Hegel, Karl Marx proposed one such general law in his theory of historical materialism, arguing that economic forces and class struggle are the fundamental drivers of historical change. Another influential development was the spread of European historiographical methods, which became the dominant approach to the academic study of the past worldwide. In the 20th century, traditional historical assumptions and practices were challenged while the scope of historical research broadened. The Annales school used insights from sociology, psychology, and economics to study long-term developments. Authoritarian regimes, like Nazi Germany, the Soviet Union, and China, manipulated historical narratives for ideological purposes. Various historians covered unconventional perspectives, focusing on the experiences of marginalized groups through approaches such as history from below, microhistory, oral history, and feminist history. Postcolonialism aimed to undermine the hegemony of the Western approach and postmodernism rejected the claim to a single universal truth in history.', 'score_dense': 0.5087823867797852, 'score_sparse': 17.5682166881247, 'score_rrf': 0.0315136476426799} overlap: 7 Best URL contributing to answer: https://en.wikipedia.org/?curid=10772350, Reciprocal Rank: 0.3333333333333333

Best supporting chunk ID: 10772350_chunk_40, Overlap: 7
F1 Score: 0.5263

Semantic Similarity Score: 0.8422

NDCG@3 Score: 0.8671

Precision@3: 1.0000