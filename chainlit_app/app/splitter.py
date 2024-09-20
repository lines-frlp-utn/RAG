from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


def split_text_with_langchain(texts, chunk_size=1000, chunk_overlap=100):
    """
    Divide el texto en fragmentos utilizando RecursiveCharacterTextSplitter de LangChain.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=[".", "!", "?", "\n\n", "\n", " ", "", "\t"]
    )
    
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    
    return chunks

def refine_split_by_similarity(chunks, embeddings, threshold=0.85):
    """
    Refina la separación de los chunks basado en la similitud coseno entre los embeddings.
    """
    refined_chunks = []
    similarities = cosine_similarity(embeddings)

    used = [False] * len(chunks)

    for i in range(len(chunks)):
        if not used[i]:
            temp_chunk = chunks[i]
            used[i] = True
            for j in range(len(chunks)):
                if not used[j] and similarities[i, j] >= threshold:
                    temp_chunk += " " + chunks[j]
                    used[j] = True
            refined_chunks.append(temp_chunk)
    
    return refined_chunks

