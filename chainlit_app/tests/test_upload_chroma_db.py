def test_upload():
    from app.embedding_generator import EmbeddingGenerator, extract_text_from_pdf

    from vectordbs.Chroma.main import upload_pdf_to_vector_db

    embedding_generator = EmbeddingGenerator()
    pdf_path = "./tests/pdfs_prueba/algoritmos.pdf"  # Reemplazar con la ruta del archivo PDF
    texts = extract_text_from_pdf(pdf_path)
    data = embedding_generator.format_for_database(texts)
    collection_name = "algo"
    upload_pdf_to_vector_db(dataWithEmbeddings=data, collection_name=collection_name)


def test_retrive():
    from app.embedding_generator import EmbeddingGenerator

    from vectordbs.Chroma.main import get_context_with_filters

    embedding_generator = EmbeddingGenerator()
    collection_name = "Prueba"
    question = ["who was Alan Turing?"]
    query = embedding_generator.get_embeddings(question).tolist()
    response = get_context_with_filters(collection_name=collection_name, query=query)
    print(response)
