import fastapi
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient

app = fastapi.FastAPI()

# Conexión a Milvus
client = MilvusClient(uri="http://milvus-standalone:19530")

class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str

class QueryData(BaseModel):
    collection_name: str
    query: list[list[float]]  # Lista de embeddings para búsqueda

def create_schema():
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=40000)
    
    return schema

def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f'Colección "{collection_name}" eliminada.')

    uploadData = [
        {"id": item["id"], "text": item["text"], "vector": item["vector"]}
        for item in dataWithEmbeddings
    ]
    
    schema = create_schema()
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        consistency_level="Strong"
    )
    print("Colección creada.")

    index_params = {
        "field_name": "vector",
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "index_name": "vector_index",
        "params": {"nlist": 128}
    }

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
        sync=True  # Espera hasta que la creación del índice termine
    )
    print("Índice creado.")

    result = client.insert(collection_name=collection_name, data=uploadData)
    print("Documentos subidos a Milvus:", result)

def get_context_with_filters(collection_name, query):
    print("Iniciando búsqueda...")
    
    client.load_collection(collection_name=collection_name)
    print(f'Colección "{collection_name}" cargada.')

    results = client.search(
        collection_name=collection_name,
        anns_field="vector", 
        output_fields=["text"],
        data=query,
        limit=3,
        search_params={"metric_type": "COSINE"}
    )

    response = [doc["text"] for doc in results]
    return response

@app.post("/upload-embeddings")
def upload(data: EmbeddingData):
    upload_pdf_to_vector_db(data.dataWithEmbeddings, data.collection_name)
    return {"status": "success"}

@app.post("/get-context")
def get_context(data: QueryData):
    return get_context_with_filters(data.collection_name, data.query)
