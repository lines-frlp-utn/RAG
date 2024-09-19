import hashlib
import pymupdf  # PyMuPDF
import ollama
import numpy as np

# model_distibert = "distilbert-base-multilingual-cased"

model_ollama = "mxbai-embed-large"


class EmbeddingGenerator:
    def __init__(self, model_name=model_ollama):
        self.model_name = model_name

    def generate_id(self, text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

    def get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings

    def format_for_database(self, texts):
        embeddings = self.get_embeddings(texts)
        result = []
        for text, emb in zip(texts, embeddings):
            emb_array = np.array(emb)  
            emb_list = emb_array.tolist()
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


if __name__ == "__main__":
    pdf_path = "../tests/pdfs_prueba/algoritmos.pdf"  
    texts = extract_text_from_pdf(pdf_path)

    embedding_generator = EmbeddingGenerator()

    embedding_list = embedding_generator.format_for_database(texts)

    for item in embedding_list:
        print(f"Text: {item['text']}")
        # print(f"Embedding: {item['vector'][:5]}...")  # Imprime los primeros 5 valores de los embeddings