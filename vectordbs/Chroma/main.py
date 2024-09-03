import chromadb
# from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fastapi 
from pydantic import BaseModel

class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str

class QueryData(BaseModel):
    collection_name: str
    query: list

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
            ids=[str(doc['id'])],
            embeddings=[doc['vector']],
            documents=[doc['text']],
        )
        print(f"{doc} cargado correctamente...")

def get_context_with_filters(collection_name, query):
    
    collection = client.get_collection(
        name=collection_name,
    )
    
    response = collection.query(query_embeddings=query, n_results=1)
    
    # final_response = []
    # for doc in response:
    #     if doc not in final_response:
    #         final_response.append(doc)
    # print(final_response)
    print(response)
    return response["documents"]


@app.post("/upload-embeddings")
def upload(data: EmbeddingData):
    dataWithEmbeddings = data.dataWithEmbeddings
    collection_name = data.collection_name
    upload_pdf_to_vector_db(dataWithEmbeddings, collection_name)
    return {"status": "success"}

@app.post("/get-context")
def get_context(data: QueryData):
    query = data.query
    collection_name = data.collection_name
    return get_context_with_filters(collection_name, query)

