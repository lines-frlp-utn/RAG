from langchain_experimental.text_splitter import SemanticChunker
from app.embeddingGenerator import EmbeddingGenerator

# Parámetros de chunking
DEFAULT_MAX_LENGTH = 4000

embedding_generator = EmbeddingGenerator()

semantic_splitter = SemanticChunker(
    embeddings=embedding_generator,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80.0, 
    min_chunk_size=100,
)

# Función principal
def split_semantic(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[str]:
    """
    Aplica splitting semántico sobre el texto completo.
    Los fragmentos estarán todos por debajo del tamaño máximo especificado.
    """
    final_chunks: list[str] = []

    # Aplicar el splitter semántico
    docs = semantic_splitter.create_documents([text])
    for doc in docs:
        chunk = doc.page_content.strip()

        # Solo almacenamos los chunks que sean más pequeños que max_length
        if len(chunk) <= max_length:
            final_chunks.append(chunk)

    # Retornamos los fragmentos generados que cumplen con el tamaño máximo
    return final_chunks
