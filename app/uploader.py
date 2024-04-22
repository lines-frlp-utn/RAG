from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from app.models import embedding_model
from app.splitter import text_splitter

def upload_pdf_to_database(text_file, theme, subtheme, collection_name):

    loader = PyPDFLoader(text_file)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata ={
            "source": text_file,             
            "theme": theme,
            "subtheme": subtheme,
        }
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory="./database/",
    )

    print(text_file + " cargado correctamente...")

def get_context_with_filters(collection_name, theme, subtheme, query):

    collection = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="./database/"
    )
    if theme != "-":
        retriever = collection.as_retriever(search_kwargs={"filter":{"theme":theme}}, search_type="similarity")
    else:
        retriever = collection.as_retriever(search_type="similarity")
    
    response = retriever.get_relevant_documents(query)
    
    return response
    # response = retriever.get_relevant_documents(query)

    # if response is None:
    #     response = collection.similarity_search(query)

    # return response

# upload_pdf_to_database("../tests/pdfs_prueba/bitcoin_es.pdf", theme="Bitcoin", subtheme="", collection_name="CryptoCurrency")
# upload_pdf_to_database("../tests/pdfs_prueba/Ethereum.pdf", theme="Ethereum", subtheme="", collection_name="CryptoCurrency")
# get_context_with_filters(collection_name="CryptoCurrency", theme="Bitcoin", subtheme="", query="como funciona bitcoin")