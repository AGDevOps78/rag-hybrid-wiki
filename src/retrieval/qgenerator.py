import json
import random
from typing import List, Dict
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from src.retrieval.generator import Generator   # your Flan‑T5 wrapper
from src.retrieval.hybrid import retrieve_hybrid


class QGenerator:
    """
    Automated question generator for building evaluation datasets.
    Uses Flan‑T5‑base with a strong prompt to avoid index-only answers.
    """


    def __init__(self, dense_ret=None, sparse_ret=None, model_name="google/flan-t5-base", max_input_tokens=1024, max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.dense = dense_ret
        self.sparse = sparse_ret


    # ---------------------------------------------------------
    # Strong prompt template (Flan‑T5‑friendly)
    # ---------------------------------------------------------
    def build_prompt(self, chunks: List[Dict], selected_qtype) -> str:
        ctx = ""
        for c in chunks:
            ctx += f"\n[Title: {c['title']}]\t{c['text']}\n"

        prompt = f"""You are an evaluator.Use the only provided context below to generate a question of type {selected_qtype}. 
Your task:
- Generate question of type {selected_qtype} from the text after Context: ONLY
- Ensure the question is of type {selected_qtype} ONLY
- Each question MUST be answerable ONLY from the provided Context below.
- Start question with Why, How, What, When, Where, Who, Define, Explain, Compare, or Contrast.
- Form question in natural language.
Context:{ctx}
            """
        return prompt  

    def build_prompt_ans(self, chunks: List[Dict], question: str) -> str:
        ctx = ""
        for c in chunks:
            ctx += f"\n[Chunk {c['chunk_uid']}]\t{c['text']}\n"

        prompt = f"""You are an expert assistant. Use the only provided context below to answer the question.
        Do NOT output a chunk number. 
        Answer STRICTLY from the context below.
        Write a natural language explanation.
        Context:{ctx}
        Question: {question}
       """
        
        return prompt 

    def build_prompt_qtype(self, chunks: List[Dict], question: str) -> str:
        
        prompt = f"""
You are an expert assistant.For the given question, determine its type from the options: factual, inferential, comparative, multi-hop.
Your task:
- Generate answer to the Question STARTING with "Answer:".
- The question MUST be answerable ONLY from the provided text below.
- Write a natural language answer in 1 or 2 sentences.
-DONOT repeat a sentence in the answer.
Question: {question}
            """
        return prompt               
    # ---------------------------------------------------------
    # Check if answer is grounded in chunks
    # ---------------------------------------------------------
    def answer_is_grounded(self, answer_raw, chunk_list):
        answer = answer_raw.lower()

        # Combine all chunk text
        combined = " ".join(c["text"] for c in chunk_list).lower()
        # If answer text appears inside chunk text → grounded
        return answer in combined
    
    def question_is_valid(self, question_raw,chunk_list, min_overlap_ratio=0.20, min_overlap_count=2):
        QUESTION_STOPWORDS = {
        "the","is","are","a","an","and","or","of","to","in","on","for","with",
        "as","by","at","from","that","this","it","be","was","were","can","may",
        "not","but","if","into","their","its","they","them","these","those",
        # question words
        "what","why","how","when","where","who","which","whom","whose", "define","explain","compare","contrast"
       }
        # Tokenize question
        q_tokens = set(re.findall(r"\w+", question_raw))
        q_tokens = {t for t in q_tokens if t not in QUESTION_STOPWORDS}

        if not q_tokens:
            return False
        # Combine all chunk text
        combined = " ".join(c["text"] for c in chunk_list).lower()

        # Tokenize chunk text
        chunk_tokens = set(re.findall(r"\w+", combined))
        chunk_tokens = {t for t in chunk_tokens if t not in QUESTION_STOPWORDS}

        # Compute overlap
        overlap = q_tokens & chunk_tokens

        # Overlap metrics
        overlap_count = len(overlap)
        overlap_ratio = overlap_count / max(len(q_tokens), 1)

        # Conditions for grounding
        if overlap_count >= min_overlap_count and overlap_ratio >= min_overlap_ratio:
          return True

        return False


    
    def answer_is_grounded_with_re(self,answer_raw, chunk_list, min_overlap_ratio=0.20, min_overlap_count=2):
       # Normalize answer
       ans = answer_raw.lower()
       STOPWORDS = {
        "the","is","are","a","an","and","or","of","to","in","on","for","with",
        "as","by","at","from","that","this","it","be","was","were","can","may",
        "not","but","if","into","their","its","they","them","these","those"
       }

       # Tokenize answer
       ans_tokens = set(re.findall(r"\w+", ans))
       ans_tokens = {t for t in ans_tokens if t not in STOPWORDS}

       if not ans_tokens:
          return False

       # Combine all chunk text
       combined = " ".join(c["text"] for c in chunk_list).lower()

      # Tokenize chunk text
       chunk_tokens = set(re.findall(r"\w+", combined))
       chunk_tokens = {t for t in chunk_tokens if t not in STOPWORDS}

       # Compute overlap
       overlap = ans_tokens & chunk_tokens

       # Overlap metrics
       overlap_count = len(overlap)
       overlap_ratio = overlap_count / max(len(ans_tokens), 1)

       # Debug (optional)
       # print("Tokens:", ans_tokens)
       # print("Overlap:", overlap)
       # print("Ratio:", overlap_ratio)

       # Conditions for grounding
       if overlap_count >= min_overlap_count and overlap_ratio >= min_overlap_ratio:
          return True
       return False
    # ---------------------------------------------------------
    # Generate questions for a pair of chunks
    # ---------------------------------------------------------
    def generate_for_chunks(self, chunk_list: List[Dict]) -> List[Dict]:
        # 1. Generate question (plain text)
        qtype = ["factual", "inferential", "comparative", "multi-hop"]
        random.shuffle(qtype)
        selected_qtype = qtype[0]

        prompt = self.build_prompt(chunk_list,selected_qtype)
        #print(f"Q prompt:\n{prompt}\n")
        question_raw = self.generate(prompt).strip()
        #print(f"Q output:\n{question_raw}\n")

        # 2. Generate answer (plain text)
        ansprompt = self.build_prompt_ans(chunk_list, question_raw)
        answer_raw = self.generate(ansprompt).strip()
        print(f"Q output:\n{question_raw}\n")
        print(f"Answer output:\n{answer_raw}\n")
        #qtypeprompt = self.build_prompt_qtype(chunk_list, question_raw)
        #qtype_raw = self.generate(qtypeprompt).strip()
        #print(f"QType output:\n{qtype_raw} {selected_qtype} \n")
        if not self.question_is_valid(question_raw, chunk_list):
            print("Question not grounded in chunks, skipping.\n")
            return []

        #check if answer is valid
        if not self.answer_is_grounded_with_re(answer_raw, chunk_list):
            print("Answer not grounded in chunks, skipping.\n")
            return []
        # 3. Wrap into JSON structure
        return [{
            "question_type": selected_qtype,
            "question": question_raw,
            "ground_truth": answer_raw,
            "source_ids": [c["chunk_uid"] for c in chunk_list],
            "wikipedia_url": [c["wikipedia_url"] for c in chunk_list]
              }]

    # ---------------------------------------------------------
    # Generate N questions total
    # ---------------------------------------------------------
    def generate_dataset(self, chunks: List[Dict], target_count=1) -> List[Dict]:
        results = []
        print(f"Generating {target_count} Q&A pairs from {len(chunks)} chunks...")
        while len(results) < target_count:
            c1 = random.choice(chunks)
            #c2 = random.choice(chunks)
            
            qset = []
            qset = self.generate_for_chunks([c1])
            if not qset:
                continue
            print(f"{qset}\n")
            results.extend(qset)

        return results[:target_count]

    # ---------------------------------------------------------
    # Save to JSONL
    # ---------------------------------------------------------
    def save(self, qlist: List[Dict], path="data/generated_questions.jsonl"):
        # Append instead of overwrite
        with open(path, "a", encoding="utf-8") as f:
            for q in qlist:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------
    # Generate text using Flan‑T5
    # ---------------------------------------------------------
    def generate(self, prompt):
        #prompt = self.build_prompt(query, chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)