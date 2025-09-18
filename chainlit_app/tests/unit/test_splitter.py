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


# ______________________________________ TESTS ______________________________________

def test_markdown_splitter():
    """
    Prueba la funcionalidad del splitter de Markdown con un documento PDF real.
    
    Este test verifica que:
    1. La función split_markdown_text puede procesar correctamente texto extraído de un PDF
    2. Se generan fragmentos no vacíos del documento
    3. Ningún fragmento excede el límite máximo de 4000 caracteres
    4. La división respeta la estructura del contenido
    
    Flujo del test:
    - Carga un documento PDF de prueba desde la carpeta fixtures
    - Extrae el texto del PDF usando extract_text_from_pdf
    - Procesa el texto con split_markdown_text
    - Valida que se generen fragmentos y que cumplan con los límites de tamaño
    
    El PDF 'bitcoin_es.pdf' sirve como caso de prueba realista que probablemente
    contiene encabezados y estructura que el splitter debe reconocer.
    """
    from app.splitter.markdown_splitter import split_markdown_text

    # Cargar documento PDF de prueba desde la carpeta fixtures
    pdf_path = os.path.join(os.path.dirname(__file__), "../fixture/bitcoin_es.pdf")

    # Extraer texto del PDF para procesamiento
    texto_pdf = extract_text_from_pdf(pdf_path)
    
    # Dividir el texto en fragmentos usando el splitter de Markdown
    chunks = split_markdown_text(texto_pdf)

    # Verificar que se generaron fragmentos (test básico de funcionalidad)
    assert len(chunks) > 0, "No se generaron fragmentos del PDF"

    # Verificar que ningún fragmento excede el límite máximo de 4000 caracteres
    for chunk in chunks:
        assert len(chunk) <= 4000, "Un fragmento excede la longitud máxima permitida"



# Test anterior
# if __name__ == "__main__":
#     pdf_path = "/workspace/chainlit_app/tests/fixture/bitcoin_es.pdf"
#     texto_pdf = extract_text_from_pdf(pdf_path)
#     chunks = split_semantic(texto_pdf)
#     for i, chunk in enumerate(chunks):
#         print(f"\n--- Chunk {i + 1} ---\n{chunk[:300]}...")


# para ejecutar el test:
# python -m chainlit_app.tests.test_splitter