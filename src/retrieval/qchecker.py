from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class qCheckGenerator:
    def __init__(self, model_name="google/flan-t5-base",
                 max_input_tokens=1024, max_new_tokens=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def build_prompt(self, query: str) -> str:
        return (
            "You are a STRICT classifier. Your job is to determine whether the text is QUESTION.\n\n"
        "A well‑formed question MUST:\n"
        "- Ask for information\n"
        "- Be interrogative in meaning\n"
        "- NOT be a greeting, statement, or fragment\n\n"
        "- QUESTION starts with What, Why, Where, When, How, Can, Could, Should, Would, May, Might, Explain,Define, Compare, or Contrast  \n"
        "- Starts with Explain, Define, Compare, or Contrast are also valid question starters.\n\n"
        "Examples of VALID questions:\n"
        "- What can cause vision with both eyes to be worse than with one eye alone?"
        "- What is the name of the process that produced most of the hydrogen, helium and a very small quantity of lithium?"
        "- What was the first public work Grace u Rofflu presented in ?"
        "- What are poison exons?\n"
        "- What is astro physics?"
        "- Where is Shopska salad considered a national dish of Bulgaria ?"
        "- How is carbon black different from other materials ?"
        "- Why is Turner's work closely linked to Durkheim's seminal work in The Ritual Process?"
        "- What is Mathematics ?\n"
        "- What is the capital of France?\n"
        "- What are the health benefits of green tea?\n"
        "- How does photosynthesis work?\n"
        "- What does the LinkML schema tries to anchor the meaning of free text strings by establishing identity via resolvable URIs?\n"
        "- Who is the president of the United States?\n"
        "- When was the Declaration of Independence signed?\n"
        "- Which country has the largest population?\n"
        "- What is Physics?\n"
        "- What is Term of Office?\n"
        "- why is the sky blue?\n"
        "- How does BM25 scoring work?\n\n"
        "- Explain the concept of dense retrieval.\n" \
        "- Define the term 'tokenization' in NLP.\n" \
        "- Compare and contrast dense and sparse retrieval methods.\n\n"
        "Examples of INVALID questions:\n"
        "- Hello world \n"
        "- Tell me about the weather\n"
        "- Hello world?\n"
        "- The quick brown fox\n"
        "- How are you doing today?\n"
        "- Greetings, how are you?\n"
        "- Hello, can you help me?\n\n"
        "STRICTLY Respond with ONLY YES or NO.\n\n"
        f"Text: {query}"
        )

    def llm_semantic_question_check(self, query: str) -> bool:
        prompt = self.build_prompt(query)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens
        )

        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = resp.strip().lower()
        bool_val = False
        positive_markers = ["yes", "valid", "true", "question", "interrogative", "well-formed"]

        if any(marker in resp for marker in positive_markers):
            bool_val = True

        print(f"LLM Question Check Response: '{resp}' for query: '{query}'")
        return bool_val