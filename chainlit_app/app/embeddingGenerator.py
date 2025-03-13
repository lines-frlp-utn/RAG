import hashlib

import numpy as np
import pymupdf
from app.config import conf
from langchain_openai import OpenAIEmbeddings

# http://<IP_DEL_SERVIDOR>:<PUERTO>/api/embeddings
remote_service_url = f"{conf.MODEL_URL}:{conf.MODEL_PORT}/api/embeddings"


class EmbeddingGenerator:
    def __init__(self, service_url=remote_service_url, model_name="granite-embedding"):
        self.service_url = service_url
        self.embedding_model = OpenAIEmbeddings(
            base_url=service_url,
            model=model_name,
            api_key="none",
        )

    def generate_id(self, text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts):
        try:
            embeddings = self.embedding_model.embed_documents(texts)
        except Exception as e:
            print(f"Error al obtener embeddings: {e}")
        return embeddings

    def format_for_database(self, texts):
        embeddings = self.get_embeddings(texts)
        result = []
        for text, emb in zip(texts, embeddings):
            emb_list = np.array(emb).tolist()
            result.append({"id": self.generate_id(text), "text": text, "vector": emb_list})
        return result


def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        texts.append(text)
    return texts
