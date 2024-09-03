import hashlib
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch

model_distibert = "distilbert-base-multilingual-cased"
model_bertMini = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingGenerator:
    def __init__(self, model_name=model_bertMini):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_id(self,text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 8)

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
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

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        texts.append(text)
    return texts

if __name__ == "__main__":
    pdf_path = "../tests/pdfs_prueba/algoritmos.pdf"  # Reemplazar con la ruta del archivo PDF
    texts = extract_text_from_pdf(pdf_path)

    embedding_generator = EmbeddingGenerator()

    # Obtener la lista de diccionarios en el formato [{"text": <chunk>, "embeddings": <embedding>}]
    embedding_list = embedding_generator.format_for_database(texts)

    for item in embedding_list:
        print(f"Text: {item['text']}")
        print(f"Embedding: {item['embeddings'][:5]}...")  # Imprime los primeros 5 valores de los embeddings