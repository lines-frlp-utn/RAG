import fastapi
# import json
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient, AnnSearchRequest, WeightedRanker

app = fastapi.FastAPI()

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
    schema.add_field(field_name="tokens", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=40000) # cambiar el max_length cuando tengamos el recursive text splitter para los chunks
    
    return schema


def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    print("Uploading data to Milvus collection " + collection_name + "vector dim: " + len(dataWithEmbeddings[0]["vector"]) + " tokens dim: " + len(dataWithEmbeddings[0]["tokens"]))
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    
    schema = create_schema()
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        dimension=1024,  # The vectors we will use in this demo has 768 dimensions
    )
    print("Collection created")
    print("Creating index")
    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index", 
        params={ "nlist": 128 }
    )
    print("vector Index created")
    index_params.add_index(
        field_name="tokens",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="tokens_index",
        params={ "nlist": 128 }
    )
    print("tokens Index created")
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
        sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
    )
    print("Index created")

    print(dataWithEmbeddings)

    result = client.insert(
        collection_name = collection_name,
        data = dataWithEmbeddings,
    )

    print("Docs uploaded to Milvus")
    print(result)


def get_context_with_filters(collection_name, query):
    client.load_collection(collection_name=collection_name)

    dense_request = AnnSearchRequest(
        data=query[0]["vector"],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3
    )

    sparse_request = AnnSearchRequest(
        data=query[0]["tokens"],
        anns_field="tokens",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3
    )

    reqs = [dense_request, sparse_request]
    rerank = WeightedRanker(0.5, 0.5)

    respuesta = client.hybrid_search(
        collection_name=collection_name,  # target collection
        reqs=reqs,  # search requests
        ranker=rerank,  # query vectors
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
def get_context(data: EmbeddingData):
    query = data.dataWithEmbeddings
    collection_name = data.collection_name
    print("collection_name: " + collection_name)
    return get_context_with_filters(collection_name, query)
