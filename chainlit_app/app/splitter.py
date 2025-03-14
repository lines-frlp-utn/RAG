import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def split_text_with_langchain(texts, chunk_size=1000, chunk_overlap=200):
    """
    Divide el texto en fragmentos utilizando RecursiveCharacterTextSplitter de LangChain.

    Args:
        texts (list): Lista de textos a dividir.
        chunk_size (int): Tamaño máximo de cada fragmento.
        chunk_overlap (int): Cantidad de superposición entre fragmentos.

    Returns:
        list: Lista de fragmentos de texto.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".", "!", "?", "\n\n", "\n", "\t", ",", ";", ":"],
    )

    # Usar comprensión de listas para dividir texto
    return [chunk for text in texts for chunk in text_splitter.split_text(text)]


def split_markdown_text(text) -> list[str]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        # ("######", "Header 6")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    chunks = markdown_splitter.split_text(text)
    return [chunk.page_content for chunk in chunks]


def refine_split_by_similarity(chunks, embeddings, threshold=0.9):
    """
    Refina y combina fragmentos de texto basándose en su similitud semántica utilizando
    las embeddings y un umbral específico.

    Args:
        chunks (list): Lista de fragmentos de texto a refinar.
        embeddings (ndarray): Matriz de embeddings correspondiente a los fragmentos de texto.
        threshold (float): Umbral de similitud para combinar fragmentos similares.
                           Por defecto es 0.7.

    Returns:
        list: Lista de fragmentos de texto refinados y combinados.
    """

    similarities = cosine_similarity(embeddings)
    used = np.zeros(len(chunks), dtype=bool)
    refined_chunks = []

    precomputed_similarities = {}
    # llenar el diccionario con las similitudes. indice : similitud
    for i in range(len(chunks)):
        precomputed_similarities[i] = np.where(similarities[i] >= threshold)[0]

    for i in range(len(chunks)):
        if not used[i]:
            temp_chunk = [chunks[i]]
            used[i] = True

            # Encuentra índices similares que no han sido utilizados
            similar_indices = precomputed_similarities[i][~used[precomputed_similarities[i]]]

            # Combina los chunks similares
            for j in similar_indices:
                if not used[j]:  # Solo combina si aún no se ha utilizado
                    temp_chunk.append(chunks[j])
                    used[j] = True

            # Agrega el nuevo chunk combinado
            refined_chunks.append(" \n\n".join(temp_chunk))

    return refined_chunks
