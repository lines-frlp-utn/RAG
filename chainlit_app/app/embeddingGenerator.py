import hashlib
import torch
from transformers import AutoModel, AutoTokenizer

model_bertMini = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingGenerator:
    def __init__(self, model_name=model_bertMini):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_id(self, text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_hidden_state.mean(dim=1)
        return embeddings

    def format_for_database(self, texts):
        embeddings = self.get_embeddings(texts)
        result = []
        for text, emb in zip(texts, embeddings):
            emb_list = emb.tolist()
            result.append({"id": self.generate_id(text), "text": text, "vector": emb_list})
        return result
