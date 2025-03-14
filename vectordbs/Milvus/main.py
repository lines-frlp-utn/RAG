import fastapi
from pydantic import BaseModel
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
)
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

app = fastapi.FastAPI()

# Conexión a Milvus
client = MilvusClient(uri="http://milvus-standalone:19530")

# Crear el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()


class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str


class QueryData(BaseModel):
    collection_name: str
    query: str
    query_embedding: list[float]


def create_schema():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4000)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=768)
    return schema


def sparse_matrix_to_dict(matrix: csr_matrix) -> dict:
    return {int(index): float(value) for index, value in zip(matrix.indices, matrix.data)}


def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    print("ENTRANDO A LA FUNCION UPLOAD")

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"Colección borrada: {collection_name}")

    ## Parseamos los datos
    texts = [item["text"] for item in dataWithEmbeddings]

    # Aprende el vocabulario y genera embeddings TF-IDF (MATRIZ CSR)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    print("Embeddings TF-IDF generados")

    uploadData = []
    for i, item in enumerate(dataWithEmbeddings):
        data = {
            "id": item["id"],
            "text": item["text"],
            "dense": item["vector"],
            "sparse": sparse_matrix_to_dict(tfidf_matrix[i]),
        }
        uploadData.append(data)
    print(uploadData[0])

    ## Creación del esquema
    schema = create_schema()

    # Creación y mod de índices
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    ## Creamos la colección
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )

    ## Insertamos los datos
    res = client.insert(collection_name=collection_name, data=uploadData)
    print(f"Cargados con éxito: {res}")


def get_context_with_filters(query_data: QueryData):
    print("ENTRANDO A LA FUNCION GET CONTEXT")

    ## Campo dense
    dense_query_vector = query_data.query_embedding
    print(dense_query_vector)
    dense_param = {
        "data": [dense_query_vector],
        "anns_field": "dense",
        "param": {"metric_type": "IP", "params": {"nprobe": 10}},
        "limit": 2,
    }
    request_1 = AnnSearchRequest(**dense_param)

    ## Campo sparse
    # Convertir la query a embedding TF-IDF
    query_text = query_data.query
    query_tfidf = tfidf_vectorizer.transform([query_text])
    sparse_query_vector = sparse_matrix_to_dict(query_tfidf)
    print(sparse_query_vector)

    sparse_param = {
        "data": [sparse_query_vector],
        "anns_field": "sparse",
        "param": {"metric_type": "IP", "params": {}},
        "limit": 2,
    }
    request_2 = AnnSearchRequest(**sparse_param)

    ## Creamos la lista de requests
    reqs = [request_1, request_2]

    ## Creamos ReRanker
    ranker = RRFRanker()  # ==> Default en k=60

    ## Realizamos la búsqueda
    res = client.hybrid_search(
        collection_name=query_data.collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=2,
        output_fields=["text"],
    )
    print(f"Resultado: {res}")
    return res


@app.post("/upload-embeddings")
def upload(data: EmbeddingData):
    upload_pdf_to_vector_db(data.dataWithEmbeddings, data.collection_name)
    return {"status": "success"}


@app.post("/get-context")
def get_context(query_data: QueryData):
    return get_context_with_filters(query_data)
