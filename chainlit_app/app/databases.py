import requests
from app.config import conf


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


def get_context_from_db(collection_name, query):
    print("query comun: " + str(query))
    response = requests.post(
        f"{conf.DB_URL}:{conf.DB_PORT}/get-context",
        json={
            "collection_name": collection_name,
            "query": query,
        },
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        print("Request successful")
    return response.json()
