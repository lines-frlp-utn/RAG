from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.embeddingGenerator import EmbeddingGenerator

# Parámetros de chunking
DEFAULT_MAX_LENGTH = 4000
DEFAULT_CHUNK_OVERLAP = 200

# Wrapper para hacer compatible tu generador con SemanticChunker
class LangChainCompatibleEmbeddings:
    def __init__(self, generator):
        self.generator = generator

    def embed_documents(self, texts):
        # LangChain espera solo una lista de listas de floats
        return self.generator.get_embeddings(texts)

# Inicializa tu generador de embeddings y el wrapper
embedding_generator = EmbeddingGenerator()
embedding_model = LangChainCompatibleEmbeddings(embedding_generator)

# 1) Splitter semántico
semantic_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile",
    min_chunk_size=50,
)

# 2) Splitter recursivo para fallback cuando supere el límite
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_MAX_LENGTH,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    length_function=len,
)

def split_semantic(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    1. Aplica splitting semántico sobre el texto completo.
    2. Para los chunks semánticos muy largos (> max_length), usa fallback recursivo.

    Sólo devuelve fragments de longitud <= max_length.
    """
    final_chunks: list[str] = []

    # 1) Split semántico directo
    docs = semantic_splitter.create_documents([text])
    for doc in docs:
        chunk = doc.page_content.strip()
        # 2) Si el chunk semántico supera max_length, recurre al splitter recursivo
        if len(chunk) > max_length:
            smaller = recursive_splitter.split_text(chunk)
            final_chunks.extend([c.strip() for c in smaller if len(c) <= max_length])
        else:
            final_chunks.append(chunk)

    # Normalización final: eliminar duplicados y espacios redundantes
    cleaned_chunks = [c for c in {chunk.strip() for chunk in final_chunks} if c]

    return cleaned_chunks
