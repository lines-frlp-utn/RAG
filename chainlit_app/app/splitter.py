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
        separators=[".", "!", "?", "\n\n", "\n", "\t", ",", ";", ":"]
    )
    
    # Usar comprensión de listas para dividir texto
    return [chunk for text in texts for chunk in text_splitter.split_text(text)]

def refine_split_by_similarity(chunks, embeddings, threshold=0.7):
    similarities = cosine_similarity(embeddings)
    used = np.zeros(len(chunks), dtype=bool)
    refined_chunks = []

    for i in range(len(chunks)):
        if not used[i]:
            temp_chunk = [chunks[i]]
            used[i] = True
            
            # Encuentra índices similares que no han sido utilizados
            similar_indices = np.where((similarities[i] >= threshold) & ~used)[0]
            
            # Combina los chunks similares
            for j in similar_indices:
                if not used[j]:  # Solo combina si aún no se ha utilizado
                    temp_chunk.append(chunks[j])
                    used[j] = True
            
            # Agrega el nuevo chunk combinado
            refined_chunks.append(" ".join(temp_chunk))
    
    return refined_chunks

