import fastapi
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient, AnnSearchRequest, RRFRanker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = fastapi.FastAPI()

# Conexión a Milvus
client = MilvusClient(uri="http://milvus-standalone:19530")

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str

class QueryData(BaseModel):
    collection_name: str
    query: list[dict]  # Lista de embeddings para búsqueda

def create_schema():
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=40000)
    
    return schema

def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    print('Entro a la funcion upload to vector db')
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f'Colección "{collection_name}" eliminada.')

    texts = [item["text"] for item in dataWithEmbeddings]
    sparse_vectors = tfidf_vectorizer.fit_transform(texts)

    uploadData = []
    for i, item in enumerate(dataWithEmbeddings):
        sparse_vector = sparse_vectors[i]
        sparse_dict = {int(index): float(value) for index, value in zip(sparse_vector.indices, sparse_vector.data)}
        data = {
            "id": item["id"],
            "text": item["text"],
            "dense_vector": item["vector"],
            "sparse_vector": sparse_dict,
        }
        uploadData.append(data)
    
    print("Sparse Vectors added to data")
    
    schema = create_schema()
    print("Schema creado.")
    
    index_params = MilvusClient.prepare_index_params()
    
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
        metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
        params={"inverted_index_algo": "DAAT_MAXSCORE"},  # The ratio of small vector values to be dropped during indexing
    )
    
    client.create_collection(
        collection_name=collection_name,
        index_params=index_params,
        schema=schema,
    )
    print("Coleccion creada.")

    print(f"SPARSE_VECTORS", uploadData[0]['sparse_vector'])
    print(uploadData[0])
    result = client.insert(collection_name=collection_name, data=uploadData)
    print("Documentos subidos a Milvus:", result)

def get_context_with_filters(collection_name, query):
    print("Iniciando búsqueda...")  
      
    ##DENSE QUERY VECTOR
    dense_query_vector = query[0]['vector']
    print(f"Query dense_vector: {dense_query_vector}")
    
    search_param_1 = {
        "data": [dense_query_vector],
        "anns_field": "dense_vector",
        "param": {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        },
        "limit": 2
    }
    request_1 = AnnSearchRequest(**search_param_1)
    
    #SPARSE QUERY VECTOR
    sparse_query_vector = tfidf_vectorizer.transform([query[0]['text']])
    sparse_dict = int(index): float(value) for index, value in zip(sparse_query_vector.indices, sparse_query_vector.data)
    print(f"Query sparse_vector: sparse_dict}")
    
    search_param_2 = {
        "data": [sparse_dict],
        "anns_field": "sparse_vector",
        "param": {
            "metric_type": "IP",
            "params": {}
        },
        "limit": 2
    }
    request_2 = AnnSearchRequest(**search_param_2)
    
    #Requests
    reqs = [request_1, request_2]
    
    #ReRanker
    ranker = RRFRanker() #60 =>default
    
    #Busqueda Hibrida
    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=2
    )
    print(f"Resultados: {res}")
    
    client.load_collection(collection_name=collection_name)
    print(f'Colección "{collection_name}" cargada.')

@app.post("/upload-embeddings")
def upload(data: EmbeddingData):
    upload_pdf_to_vector_db(data.dataWithEmbeddings, data.collection_name)
    return {"status": "success"}

@app.post("/get-context")
def get_context(data: QueryData):
    return get_context_with_filters(data.collection_name, data.query)