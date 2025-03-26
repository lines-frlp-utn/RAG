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
    # print("\n--- ENTRANDO A get_context_with_filters ---")
    # print(f"Colección: {collection_name}")
    # print(f"Tipo de embedding: {type(query_embedding)}")
    # print(f"Longitud embedding: {len(query_embedding) if hasattr(query_embedding, '__len__') else 'N/A'}")
    try:
        collection = client.get_collection(name=collection_name)
        
        # Consulta con parámetros adicionales para debug
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"]  # Incluir distancias para debug
        )
        
        # Debug detallado
        # print("\n--- DEBUG ChromaDB Response ---")
        # print("Total results:", len(response['ids'][0]))
        # print("IDs:", response['ids'][0])
        # print("Documents:", response['documents'][0])
        # print("Metadatas:", response['metadatas'][0])
        # print("Distances:", response.get('distances', [[]])[0])
        # print("-----------------------------\n")
        
        # Procesamiento seguro de resultados
        documents = response.get("documents", [[]])[0] or ["[Documento sin texto]"]
        ids = response.get("ids", [[]])[0] or ["unknown"]
        metadatas = response.get("metadatas", [[]])[0] or [{}]
        
        retrieve_data_list = []
        for doc_id, doc_text, meta in zip(ids, documents, metadatas):
            # Validación exhaustiva del texto
            if not doc_text or not str(doc_text).strip():
                print(f"¡Documento vacío encontrado! ID: {doc_id}")
                doc_text = "[Contenido no disponible]"
            
            retrieve_data_list.append(
                RetrieveData(
                    id=str(doc_id),
                    text=str(doc_text).strip(),
                    metadata=meta if isinstance(meta, dict) else {}
                )
            )
        
        return retrieve_data_list
    
    except Exception as e:
        # print(f"\n--- ERROR en get_context_with_filters ---")
        # print(f"Collection: {collection_name}")
        # print(f"Query embedding: {query_embedding[:3]}...")  # Mostrar solo parte del embedding
        # print(f"Error: {str(e)}")
        # print("------------------------------------\n")
        
        # Retornar estructura vacía pero válida
        return [
            RetrieveData(
                id="error",
                text="Error recuperando contexto",
                metadata={"error": str(e)}
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
