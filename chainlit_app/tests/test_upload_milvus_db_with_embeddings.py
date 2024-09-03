def test_upload(): 
    from vectordbs.Milvus.main import upload_pdf_to_vector_db
    from app.embeddingGenerator import EmbeddingGenerator, extract_text_from_pdf
    
    embedding_generator = EmbeddingGenerator()
    pdf_path = "./tests/pdfs_prueba/algoritmos.pdf"  # Reemplazar con la ruta del archivo PDF
    texts = extract_text_from_pdf(pdf_path)
    data = embedding_generator.format_for_database(texts)
    collection_name = 'Prueba'
    upload_pdf_to_vector_db(dataWithEmbeddings=data, collection_name=collection_name)

def test_retrive():
    from vectordbs.Milvus.main import get_context_with_filters
    from app.embeddingGenerator import EmbeddingGenerator, extract_text_from_pdf
    embedding_generator = EmbeddingGenerator()
    collection_name = 'Prueba'
    question = ["who was Alan Turing?"]
    query = embedding_generator.get_embeddings(question).tolist()
    response = get_context_with_filters(collection_name=collection_name, query=query)
    print(response)