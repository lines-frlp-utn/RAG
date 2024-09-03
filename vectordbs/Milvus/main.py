import fastapi
import json
from pydantic import BaseModel

app = fastapi.FastAPI()

from pymilvus import MilvusClient
client = MilvusClient(uri="http://milvus-standalone:19530")

class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str

class QueryData(BaseModel):
    collection_name: str
    query: list

def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=384,  # The vectors we will use in this demo has 768 dimensions
    )
    result = client.insert(
        collection_name = collection_name,
        data = dataWithEmbeddings,
    )

    print("Docs uploaded to Milvus")
    print(result)


def get_context_with_filters(collection_name, query):
    respuesta = client.search(
        collection_name=collection_name,  # target collection
        data=query,  # query vectors
        limit=2,  # number of returned entities
        output_fields=["text"],  # specifies fields to be returned
    )
    print("Respuesta de Milvus:" + str(respuesta))
    response=[]
    for ans in respuesta:
        for item in ans:
            response.append(item['entity']['text'])
    return response


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
