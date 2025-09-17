import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pymupdf4llm
from app.embedding_generator import EmbeddingGenerator
from langchain_experimental.text_splitter import SemanticChunker

embedding_generator = EmbeddingGenerator()

# Inicializa el splitter semántico
semantic_splitter = SemanticChunker(
    embeddings=embedding_generator,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80.0,
    min_chunk_size=100,
)


# Función para extraer el texto de un PDF
def extract_text_from_pdf(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)


# Función para dividir el texto usando el splitter semántico
def split_semantic(text: str, max_length: int = 4000) -> list[str]:
    final_chunks: list[str] = []
    docs = semantic_splitter.create_documents([text])
    for doc in docs:
        chunk = doc.page_content.strip()
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
    return final_chunks


# Ejecutar test
if __name__ == "__main__":
    pdf_path = "/workspace/chainlit_app/tests/fixture/bitcoin_es.pdf"
    texto_pdf = extract_text_from_pdf(pdf_path)
    chunks = split_semantic(texto_pdf)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---\n{chunk[:300]}...")


# para ejecutar el test:
# python -m chainlit_app.tests.test_splitter
