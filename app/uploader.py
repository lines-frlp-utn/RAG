from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from models import embedding_model
import chromadb


client = chromadb.PersistentClient(path="/database")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=200, add_start_index=True
)

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
    
    collection = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory="/database",
    )

    response = collection.similarity_search(f"como funciona {theme}")
    print(response)

def get_from_db_with_filters(collection_name, theme, subtheme, query):

    collection = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="/database"
    )

    retriever = collection.as_retriever(search_kwargs={"filter":{"theme":theme}}, search_type="similarity")

    # chunks = collection.get(where={"theme":theme}, include=["metadatas", "documents", "embeddings"])
    response = retriever.get_relevant_documents(query)
    if response is None:
        response = collection.similarity_search(query)
    # if len(chunks['documents']) != 0:
    #     response = retriever.get_relevant_documents(query)
    # else:
    #     response = collection.similarity_search(query)

    return response

#upload_pdf_to_database("app/pdfs_prueba/bitcoin_es.pdf", theme="Bitcoin", subtheme="", collection_name="CryptoCurrency")
#upload_pdf_to_database("app/pdfs_prueba/Ethereum.pdf", theme="Ethereum", subtheme="", collection_name="CryptoCurrency")

get_from_db_with_filters(collection_name="CryptoCurrency", theme="Dot", subtheme="", query="como funciona Dot")