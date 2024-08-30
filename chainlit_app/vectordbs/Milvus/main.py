import fastapi

app = fastapi.FastAPI()

from pymilvus import MilvusClient
client = MilvusClient()

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
        collection_name="demo_collection",  # target collection
        data=query,  # query vectors
        limit=2,  # number of returned entities
        output_fields=["text", "subject"],  # specifies fields to be returned
    )
    return respuesta


@app.post("/upload-embeddings")
def upload(dataWithEmbeddings: dict | list[dict], collection_name):
    upload_pdf_to_vector_db(dataWithEmbeddings, collection_name)


@app.post("/get-context")
def get_context(collection_name, theme, subtheme, query: list):
    return get_context_with_filters(collection_name, theme, subtheme, query)
