
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="google/flan-t5-base", max_input_tokens=1024, max_new_tokens=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def build_prompt(self, query, chunks):
        context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])
        print("You are a helpful assistant. Use only the context below to answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        return (
            "You are a helpful assistant. Use only the context below to answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    def generate(self, query, chunks):
        prompt = self.build_prompt(query, chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)