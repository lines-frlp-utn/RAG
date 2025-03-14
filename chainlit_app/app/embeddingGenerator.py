import hashlib

import numpy as np
from app.config import conf
from langchain_ollama import OllamaEmbeddings

# http://<IP_DEL_SERVIDOR>:<PUERTO>/api/embeddings
remote_service_url = f"{conf.MODEL_URL}:{conf.MODEL_PORT}"


class EmbeddingGenerator:
    def __init__(self):
        self.service_url = remote_service_url
        self.embedding_model = OllamaEmbeddings(
            model="granite-embedding:278m",
            base_url=remote_service_url,
        )

    def generate_id(self, text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts: list[str]):
        try:
            embeddings = []
            for text in texts:
                embeddings.append(self.embedding_model.embed_query(text))
        except Exception as e:
            print(f"Error al obtener embeddings: {e}")
        return embeddings

    def format_for_database(self, embeddings: list[list[float]], chunks: list[str]) -> list[dict]:
        result = []
        for text, emb in zip(chunks, embeddings):
            emb_list = np.array(emb).tolist()
            result.append({"id": self.generate_id(text), "text": text, "vector": emb_list})
        return result
