import fastapi
from pydantic import BaseModel
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

app = fastapi.FastAPI()

# Conexión a Milvus
client = MilvusClient(uri="http://milvus-standalone:19530")


class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str


class QueryData(BaseModel):
    collection_name: str
    query: str
    query_embedding: list[float]


class RetrieveData(BaseModel):
    id: str
    text: str
    metadata: dict


def create_schema():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="text", datatype=DataType.VARCHAR, max_length=4000, enable_analyzer=True
    )
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=768)

    bm25_function = Function(
        name="text_bm25_emb",  # Function name
        input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
        output_field_names=[
            "sparse"
        ],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25,
    )

    schema.add_function(bm25_function)
    return schema


def upload_embedding_to_vector_db(dataWithEmbeddings, collection_name):
    print("ENTRANDO A LA FUNCION UPLOAD")

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"Colección borrada: {collection_name}")

    uploadData = []
    for i, item in enumerate(dataWithEmbeddings):
        data = {
            "id": item["id"],
            "text": item["text"],
            "dense": item["vector"],
            # "sparse" is done automatically by milvus
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
        field_name="sparse", index_name="sparse_index", index_type="AUTOINDEX", metric_type="BM25"
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
    dense_param = {
        "data": [dense_query_vector],
        "anns_field": "dense",
        "param": {"metric_type": "IP", "params": {"nprobe": 10}},
        "limit": 2,
    }
    request_1 = AnnSearchRequest(**dense_param)

    ## Campo sparse
    sparse_param = {
        "data": [query_data.query],
        "anns_field": "sparse",
        "param": {"params": {}},
        "limit": 2,
    }
    request_2 = AnnSearchRequest(**sparse_param)

    ## Lista de requests
    reqs = [request_1, request_2]

    ## ReRanker
    ranker = RRFRanker()

    ## Búsqueda híbrida
    res = client.hybrid_search(
        collection_name=query_data.collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=5,
        output_fields=["text"],
    )

    retrieve_data_list = []
    for hit in res[0]:
        hit_id = hit.get("id", "unknown")
        distance = hit.get("distance", -1)
        text = hit.get("entity", {}).get("text", "not text found")
        retrieve_data_list.append(
            RetrieveData(
                id=str(hit_id),
                text=str(text).strip(),
                metadata={
                    "search_metadata": {"distance": float(distance) if distance is not None else -1}
                },
            )
        )
    print(f"Resultados de búsqueda: {retrieve_data_list}")
    return retrieve_data_list


@app.post("/upload-embeddings")
def upload(data: EmbeddingData):
    upload_embedding_to_vector_db(data.dataWithEmbeddings, data.collection_name)
    return {"status": "success"}


@app.post("/get-context")
def get_context(query_data: QueryData) -> list[RetrieveData]:
    try:
        results = get_context_with_filters(query_data)
    except Exception as e:
        print(f"Error al obtener el contexto: {e}")
        return []
    if not results:
        return []
    return results
