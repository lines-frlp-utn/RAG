import hashlib
import numpy as np
from openai import OpenAI
from app.config import conf
from langchain.embeddings.base import Embeddings  

# Endpoint OpenAI-style de Ollama
remote_service_url = f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1"

class EmbeddingGenerator(Embeddings):  
    def __init__(self):
        self.service_url = remote_service_url
        self.embedding_model = OpenAI(
            base_url=remote_service_url,
            api_key="ollama",  
            timeout=60,
        )

    def generate_id(self, text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.embedding_model.embeddings.create(
                input=texts,
                model="granite-embedding:278m",
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"[Embedding Error] {e}")
            raise e
        
    def format_for_database(self, embeddings: list[list[float]], chunks: list[str]) -> list[dict]:
        result = []
        for text, emb in zip(chunks, embeddings):
            emb_list = np.array(emb).tolist()
            result.append({
                "id": self.generate_id(text),
                "text": text,
                "vector": emb_list
            }) 
        return result
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.get_embeddings([text])[0]
