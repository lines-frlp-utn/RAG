import fastapi
# import json
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient, AnnSearchRequest, WeightedRanker
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction

# Create an analyzer for English language
analyzer = build_default_analyzer(language="sp")

# Create the BM25EmbeddingFunction
bm25_ef = BM25EmbeddingFunction(analyzer)

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
    schema.add_field(field_name="tokens", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=40000) # cambiar el max_length cuando tengamos el recursive text splitter para los chunks
    
    return schema


def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    
    texts = [item["text"] for item in dataWithEmbeddings]
    print("Texts: " + str(texts))
    # Fit the model on the corpus
    sparse_embedding_func = BM25SparseEmbedding(corpus=texts, language="sp")
    print("Sparse embedding function created")    

    uploadData=[]
    for item in dataWithEmbeddings:
        data= {
            "id": item["id"],
            "text": item["text"],
            "vector": item["vector"],
            "tokens": sparse_embedding_func.embed_documents([item["text"]])[0],
        }
        uploadData.append(data)

    print("Tokens added to data")

    schema = create_schema()
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
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
        metric_type="IP",
        index_type="SPARSE_INVERTED_INDEX",
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

    result = client.insert(
        collection_name = collection_name,
        data = uploadData,
    )

    print("Docs uploaded to Milvus")
    print(result)


def get_context_with_filters(collection_name, query):
    print("entro a la funcion")
    client.load_collection(collection_name=collection_name)
    print("collection loaded")
    dense_request = AnnSearchRequest(
        data=query[0]["vector"],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3
    )
    print("dense request created")

    results = client.query(
        collection_name=collection_name, 
        output_fields=["text"], 
        filter="",
        limit=100,
        offset=0
        )
    print("todos resultados")
    # Extract corpus from results
    corpus = [doc["text"] for doc in results]  # Extract the "text" field from the retrieved results
    print(f"se cargo el corpus: {corpus} ")
    # 4. Initialize BM25SparseEmbedding with the corpus
    bm25_ef.fit(corpus)
    
    print("se creo el bm25")
    print(f"query: {query[0]['text']}")
    query_embeddings = bm25_ef.encode_queries(query[0]["text"])
    print(f"se creo el query embeddings {query_embeddings}")
    sparse_request = AnnSearchRequest(
        data=[query_embeddings],
        anns_field="tokens",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3
    )
    print("se creo el sparse request")
    reqs = [dense_request, sparse_request]
    print("se crearon las reqs")
    rerank = WeightedRanker(0.5, 0.5)
    print("se creo el rerank")
    respuesta = client.hybrid_search(
        collection_name=collection_name,  # target collection
        reqs=reqs,  # search requests
        ranker=rerank,  # query vectors
        limit=2,  # number of returned entities
        output_fields=["text"],  # specifies fields to be returned
    )
    print("Respuesta hibrida de Milvus:" + str(respuesta))
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
    print("se hizo el post")
    query = data.dataWithEmbeddings
    collection_name = data.collection_name
    print("collection_name: " + collection_name)
    return get_context_with_filters(collection_name, query)
