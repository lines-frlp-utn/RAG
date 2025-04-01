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
    query: str
    query_embedding: list[float]


class RetrieveData(BaseModel):
    id: str
    text: str
    metadata: dict


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


def get_context_with_filters(collection_name, query_embedding):
    try:
        print(f"\n Buscando en colección: '{collection_name}'")  
        
        # 1. Validar que la colección existe
        collection = client.get_collection(name=collection_name)
        if collection.count() == 0:
            raise ValueError(f"La colección '{collection_name}' está vacía")
        
        # 2. Realizar consulta con manejo de errores
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        
        
        if not response or not isinstance(response, dict):
            raise ValueError("ChromaDB devolvió una respuesta inválida")
        
        ids = response.get("ids", [[]])[0] if response.get("ids") else []
        if not ids:
            raise ValueError("No se encontraron resultados (IDs vacíos)")
        
        # 3. Procesar resultados con metadatos
        retrieve_data_list = []
        for i, (doc_id, doc_text, meta, distance) in enumerate(zip(
            ids,
            response.get("documents", [[]])[0],
            response.get("metadatas", [[]])[0],
            response.get("distances", [[]])[0]
        )):
            
            doc_text = str(doc_text) if doc_text else "[Sin texto]"
            meta = meta if isinstance(meta, dict) else {}
            
            retrieve_data_list.append(
                RetrieveData(
                    id=str(doc_id),
                    text=doc_text.strip(),
                    metadata={
                        **meta,
                        "search_metadata": {  
                            "distance": float(distance) if distance else -1,
                        }
                    }
                )
            )
        
        return retrieve_data_list

    except Exception as e:
        print(f" Error crítico: {str(e)}")  
        return [
            RetrieveData(
                id="error",
                text="Error recuperando contexto",
                metadata={
                    "error": str(e),
                    "search_metadata": {  
                        "distance": -1,
                    }
                }
            )
        ]
    
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
    query_embedding = data.query_embedding
    collection_name = data.collection_name
    return get_context_with_filters(collection_name, query_embedding)
