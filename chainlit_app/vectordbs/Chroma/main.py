import chromadb
# from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fastapi 
import uuid

app = fastapi.FastAPI()

client = chromadb.PersistentClient("./database/")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=200, 
    add_start_index=True, 
    separators=[
        "\n\n\n\n",
        "\n\n"
    ]
)

def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    vector_db = client.get_or_create_collection(collection_name)
    
    for doc in dataWithEmbeddings:
        vector_db.add(
            ids=[doc['id']],
            embeddings=[doc['vector'].tolist()],
            documents=[doc['text']],
        )
        print(f"{doc} cargado correctamente...")

def get_context_with_filters(collection_name, query: list):
    
    collection = client.get_collection(
        name=collection_name,
    )
    
    response = collection.query(query_embeddings=query)
    
    final_response = []
    for doc in response:
        if doc not in final_response:
            final_response.append(doc)
    print(final_response)
    return final_response


@app.post("/upload-embeddings")
def upload(dataWithEmbeddings: dict | list[dict], collection_name):
    upload_pdf_to_vector_db(dataWithEmbeddings, collection_name)


@app.post("/get-context")
def get_context(collection_name, theme, subtheme, query: list):
    return get_context_with_filters(collection_name, theme, subtheme, query)

