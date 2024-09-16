import fastapi
import json
from pydantic import BaseModel
from pymilvus import DataType

app = fastapi.FastAPI()

from pymilvus import MilvusClient
client = MilvusClient(uri="http://milvus-standalone:19530")
index_params = client.prepare_index_params()
class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str

class QueryData(BaseModel):
    collection_name: str
    query: list

def create_schema():
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=384)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=5000) # cambiar el max_length cuando tengamos el recursive text splitter para los chunks
    
    return schema


def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    
    schema = create_schema()
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        dimension=384,  # The vectors we will use in this demo has 768 dimensions
    )
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index", 
        params={ "nlist": 128 }
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
        sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
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
