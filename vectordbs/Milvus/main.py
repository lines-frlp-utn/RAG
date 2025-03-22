import fastapi
from pydantic import BaseModel
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
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


def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
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
    print(dense_query_vector)
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
    results = get_context_with_filters(query_data)
    result_context = ""
    for result in results[0]:
        result_context += result["entity"]["text"] + "\n"
    return result_context
