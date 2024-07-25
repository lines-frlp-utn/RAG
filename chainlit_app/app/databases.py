import requests
from app.config import conf

def post_embeddings(dataWithEmbeddings: dict | list[dict], collection_name):
    requests.post(
        f"{conf.MODEL_URL}:{conf.DB_PORT}/upload-embedding?dataWithEmbeddings={dataWithEmbeddings}&collection_name={collection_name}"
    )

def get_context_from_db(collection_name, theme, subtheme, query: list):
    answer = requests.post(
        f"{conf.MODEL_URL}:{conf.DB_PORT}/get-context?collection_name={collection_name}&theme={theme}&subtheme={subtheme}&query={query}"
    )
    return answer
