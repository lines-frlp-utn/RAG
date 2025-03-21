import chromadb
import fastapi
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class EmbeddingData(BaseModel):
    dataWithEmbeddings: list[dict]
    collection_name: str


class QueryData(BaseModel):
    collection_name: str
    query: list


app = fastapi.FastAPI()

client = chromadb.PersistentClient("./database/")


def upload_pdf_to_vector_db(dataWithEmbeddings, collection_name):
    vector_db = client.get_or_create_collection(collection_name)

    for doc in dataWithEmbeddings:
        vector_db.add(
            ids=[str(doc["id"])],
            embeddings=[doc["vector"]],
            documents=[doc["text"]],
        )
        print(f"{doc} cargado correctamente...")


def get_context_with_filters(collection_name, query):
    collection = client.get_collection(
        name=collection_name,
    )

    response = collection.query(query_embeddings=query, n_results=1)

    print(response)
    return response["documents"]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.post("/upload-embeddings")
def upload(data: EmbeddingData):
    dataWithEmbeddings = data.dataWithEmbeddings
    collection_name = data.collection_name
    upload_pdf_to_vector_db(dataWithEmbeddings, collection_name)
    return {"status": "success"}


@app.post("/get-context")
def get_context(data: QueryData):
    query = data.query
    collection_name = data.collection_name
    return get_context_with_filters(collection_name, query)
