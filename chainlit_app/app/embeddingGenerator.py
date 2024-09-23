import hashlib
import os

import numpy as np
import pymupdf
import requests
from app.config import conf
from dotenv import load_dotenv

load_dotenv()

# http://<IP_DEL_SERVIDOR>:<PUERTO>/api/embeddings
remote_service_url = f"{conf.MODEL_URL}:{conf.MODEL_PORT}/api/embeddings"


class EmbeddingGenerator:
    def __init__(self, service_url=remote_service_url, model_name="mxbai-embed-large"):
        self.service_url = service_url
        self.model_name = model_name

    def generate_id(self, text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.service_url,
                    json={
                        "model": self.model_name,
                        "prompt": f"Represent this sentence for searching relevant passages: {text}",
                    },
                )
                response.raise_for_status()
                response_data = response.json()
                embeddings.append(response_data["embedding"])
            except requests.exceptions.RequestException as e:
                print(f"Error en la solicitud: {e}")
            except ValueError as e:
                print(f"Error al decodificar JSON: {e}")
                print(f"Respuesta del servidor: {response.text}")
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
