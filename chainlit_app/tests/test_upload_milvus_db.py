import hashlib


def test_upload(): 
    from vectordbs.Milvus.main import upload_pdf_to_vector_db
    from sentence_transformers import SentenceTransformer
    
    # Text strings to search from.
    embedding_fn = SentenceTransformer('all-MiniLM-L6-V2', device='cpu')
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    
    vectors = embedding_fn.encode(docs)
    
    def generate_id(text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 8)
    
    data = [
        {"id": generate_id(docs[i]), "vector": vectors[i], "text": docs[i]}
        for i in range(len(vectors))
    ]

    print("Data has", len(data), "entities, each with fields: ", data[0].keys())
    print("Vector dim:", len(data[0]["vector"]))

    collection_name = 'Prueba'
    upload_pdf_to_vector_db(dataWithEmbeddings=data, collection_name=collection_name)

def test_retrive():
    from vectordbs.Milvus.main import get_context_with_filters
    from sentence_transformers import SentenceTransformer
    collection_name = 'Prueba'
    embedding_fn = SentenceTransformer('all-MiniLM-L6-V2', device='cpu')
    query = embedding_fn.encode(["who was Alan Turing?"])
    response = get_context_with_filters(collection_name=collection_name, query=query)
    print(response)