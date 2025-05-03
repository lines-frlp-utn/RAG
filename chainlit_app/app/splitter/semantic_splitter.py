from app.embedding_generator import embedding_generator
from langchain_experimental.text_splitter import SemanticChunker

# Parámetros de chunking
DEFAULT_MAX_LENGTH = 4000

semantic_splitter = SemanticChunker(
    embeddings=embedding_generator,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80.0,
    min_chunk_size=100,
)


def split_semantic(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[str]:
    """
    Aplica splitting semántico sobre el texto completo.
    Los fragmentos estarán todos por debajo del tamaño máximo especificado.
    """
    final_chunks: list[str] = []

    docs = semantic_splitter.create_documents([text])
    for doc in docs:
        chunk = doc.page_content.strip()
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
    print(f"Fragmentos generados: {len(final_chunks)}")
    return final_chunks
