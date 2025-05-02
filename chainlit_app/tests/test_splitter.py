import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pymupdf4llm
from app.embeddingGenerator import EmbeddingGenerator
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.base import Embeddings
from typing import List

# Definir la clase LangchainEmbeddingGenerator para compatibilidad con Langchain
class LangchainEmbeddingGenerator(Embeddings):
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_generator.get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_generator.get_embeddings([text])[0]

# Instancia de generador de embeddings
langchain_embeddings = LangchainEmbeddingGenerator()

# Inicializa el splitter semántico
semantic_splitter = SemanticChunker(
    embeddings=langchain_embeddings,
    breakpoint_threshold_type="percentile",  # Tipo de umbral
    breakpoint_threshold_amount=80.0,        # Umbral en el percentil 80
    min_chunk_size=100,                       # Tamaño mínimo de fragmento
)

# Función para extraer el texto de un PDF
def extract_text_from_pdf(pdf_path):
    text = pymupdf4llm.to_markdown(pdf_path)
    return text

# Función para dividir el texto usando el splitter semántico
def split_semantic(text: str, max_length: int = 4000) -> list[str]:
    """
    Aplica splitting semántico sobre el texto completo.
    Devuelve los fragmentos de longitud <= max_length.
    """
    final_chunks: list[str] = []

    # Aplicar el splitter semántico
    docs = semantic_splitter.create_documents([text])
    for doc in docs:
        chunk = doc.page_content.strip()

        # Solo almacenamos los chunks que sean más pequeños que max_length
        if len(chunk) <= max_length:
            final_chunks.append(chunk)

    # Normalización final: eliminar duplicados y espacios redundantes
    cleaned_chunks = [c for c in {chunk.strip() for chunk in final_chunks} if c]

    return cleaned_chunks

# Probar con archivo PDF
if __name__ == "__main__":
    pdf_path = "/workspace/chainlit_app/tests/pdfs_prueba/bitcoin_es.pdf"  # Reemplaza con la ruta de tu archivo PDF
    texto_pdf = extract_text_from_pdf(pdf_path)

    # Dividir el texto extraído
    chunks = split_semantic(texto_pdf)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---\n{chunk[:300]}...")
