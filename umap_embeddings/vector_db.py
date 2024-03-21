import chromadb
from chromadb.config import Settings
from sqlmodel import Session

db = chromadb.PersistentClient(
    path="./database", settings=Settings(anonymized_telemetry=False)
)

collection = db.get_or_create_collection(name="chatbot_data")


def store_embedding_with_document(
    embedding, id: str, document: str, collection=collection
):
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[document],
        ids=[id],
    )


def retrieve_context(embedding, collection=collection) -> list:
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=3,
    )
    return results


def retrieve_context_ids(embedding, session: Session) -> list:
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=2,
    )
    return results["ids"][0]
