import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

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
        separators=[".", "!", "?", "\n\n", "\n", " ", "", "\t"]
    )
    
    # Usar comprensión de listas para dividir texto
    return [chunk for text in texts for chunk in text_splitter.split_text(text)]

def refine_split_by_similarity(chunks, embeddings, threshold=0.85):
    """
    Refina la separación de los chunks basado en la similitud coseno entre los embeddings.
    
    Args:
        chunks (list): Lista de fragmentos de texto.
        embeddings (array): Matriz de embeddings.
        threshold (float): Umbral de similitud para combinar fragmentos.

    Returns:
        list: Lista de fragmentos refinados.
    """
    similarities = cosine_similarity(embeddings)
    used = np.zeros(len(chunks), dtype=bool)
    refined_chunks = []

    for i in range(len(chunks)):
        if not used[i]:
            temp_chunk = [chunks[i]]
            used[i] = True
            
            # Usar boolean indexing para encontrar índices que cumplan la condición
            similar_indices = np.where((similarities[i] >= threshold) & ~used)[0]
            temp_chunk.extend(chunks[j] for j in similar_indices)
            used[similar_indices] = True
            
            refined_chunks.append(" ".join(temp_chunk))
    
    return refined_chunks
