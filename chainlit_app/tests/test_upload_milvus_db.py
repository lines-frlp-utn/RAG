def test_upload(): 
    from vectordbs.Milvus.main import upload_pdf_to_vector_db
    from sentence_transformers import SentenceTransformer
    import fitz 
    
    def leer_pdf(ruta_pdf):
        documento = fitz.open(ruta_pdf)
        texto_completo = ""
        
        for pagina_num in range(documento.page_count):
            pagina = documento.load_page(pagina_num)
            texto_completo += pagina.get_text("text")
    
        documento.close()
        return texto_completo

    def dividir_texto_por_puntos(texto):
        return texto.split('.')

    ruta_pdf = "./chainlit_app/tests/pdfs_prueba/algoritmos.pdf"

    texto = leer_pdf(ruta_pdf)
    fragmentos = dividir_texto_por_puntos(texto)

    modelo = SentenceTransformer('all-MiniLM-L6-V2', device='cpu')  
    input_milvus = []  
    for i in range(len(fragmentos)):
        chunk = fragmentos[i]
        vector = modelo.encode(fragmentos[i])
        diccionario = {
            "embedding": vector.tolist(), 
            "texto": chunk
        }
        input_milvus.append(diccionario)

    collection_name = 'Prueba'
    upload_pdf_to_vector_db(dataWithEmbeddings=input_milvus, collection_name=collection_name)