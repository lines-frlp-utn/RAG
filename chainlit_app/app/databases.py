import requests
from app.config import conf
from pydantic import BaseModel

class RetrieveData(BaseModel):
    id: str
    text: str
    metadata: dict

def post_embeddings(dataWithEmbeddings, collection_name):
    print("collection name: " + collection_name)
    response = requests.post(
        f"{conf.DB_URL}:{conf.DB_PORT}/upload-embeddings",
        json={"dataWithEmbeddings": dataWithEmbeddings, "collection_name": collection_name},
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        print("Request successful")
        return f"success: {response.status_code} - {response.text}"


def get_context_from_db(collection_name, query, query_embedding) -> list[RetrieveData]:
    response = requests.post(
        f"{conf.DB_URL}:{conf.DB_PORT}/get-context",
        json={
            "collection_name": collection_name,
            "query": query,
            "query_embedding": query_embedding,
        },
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        raise Exception(f"Error: {response.status_code} - {response.text}")
    else:
        print("Request successful")
        results=response.json()
        return [RetrieveData.model_validate(result) for result in results]
