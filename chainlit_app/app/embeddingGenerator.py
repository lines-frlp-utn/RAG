import hashlib

import numpy as np
from app.config import conf
from openai import OpenAI

# http://<IP_DEL_SERVIDOR>:<PUERTO>/api/embeddings
remote_service_url = f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1"


class EmbeddingGenerator:
    def __init__(self):
        self.service_url = remote_service_url
        self.embedding_model = OpenAI(
            base_url=remote_service_url,
            api_key="ollama",
            timeout=15,
        )

    def generate_id(self, text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts: list[str]):
        try:
            embeddings = self.embedding_model.embeddings.create(
                input=texts, model="granite-embedding:278m"
            )
            transformed_embeddings = [
                embedding_object.embedding for embedding_object in embeddings.data
            ]
        except Exception as e:
            print(f"Error al obtener embeddings: {e}")
            raise e
        return transformed_embeddings

    def format_for_database(self, embeddings: list[list[float]], chunks: list[str]) -> list[dict]:
        result = []
        for text, emb in zip(chunks, embeddings):
            emb_list = np.array(emb).tolist()
            result.append({"id": self.generate_id(text), "text": text, "vector": emb_list})
        return result
