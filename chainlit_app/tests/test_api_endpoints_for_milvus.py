import httpx
import pytest

from app.config import conf
from sentence_transformers import SentenceTransformer

# Inicializar modelo de embeddings con dimensión conocida y verificable
embedding_fn = SentenceTransformer('all-mpnet-base-v2', device='cpu')

@pytest.mark.asyncio
async def test_upload_embeddings_endpoint():
    """
    Prueba de integración asíncrona para el endpoint de carga de embeddings a Milvus.
    
    Este test verifica el endpoint REST API '/upload-embeddings' que recibe datos
    con embeddings precalculados y los almacena en la base de datos vectorial Milvus.

    Objetivos del test:
    1. Validar que el endpoint acepta y procesa correctamente datos con embeddings
    3. Comprobar que la estructura de datos es compatible con el formato esperado
    4. Asegurar que el servidor responde con status HTTP 200 (éxito)
    5. Testear la comunicación asíncrona usando httpx.AsyncClient

    Flujo detallado:
    1. Prepara textos de ejemplo sobre historia de la inteligencia artificial
    2. Genera embeddings vectoriales usando el modelo SentenceTransformer configurado
    4. Estructura los datos en el formato esperado por el endpoint:
       - IDs únicos generados mediante embedding_generator.generate_id()
       - Vectores convertidos a lista de Python con .tolist()
       - Textos originales preservados para contexto
    5. Realiza una petición HTTP POST asíncrona al endpoint configurado
    6. Valida que la respuesta tenga status code 200 indicando éxito

    Dependencias y configuraciones:
    - Requiere marcador pytest.mark.asyncio para tests asíncronos
    - Utiliza httpx.AsyncClient para comunicación HTTP asíncrona
    - Usa configuración desde app.config.conf para URL y puerto de la base de datos
    - Asume que embedding_fn está inicializado con modelo de 768 dimensiones

    Args:
        No recibe parámetros directos, pero depende de:
        - embedding_fn: Modelo SentenceTransformer previamente inicializado
        - embedding_generator: Módulo con función generate_id() para IDs únicos
        - conf.DB_URL: URL de la base de datos desde configuración
        - conf.DB_PORT: Puerto de la base de datos desde configuración

    Raises:
        AssertionError: Si el status code de la respuesta no es 200
        httpx.HTTPError: Si hay errores de conexión con el endpoint
        Exception: Para cualquier otro error durante la ejecución

    Notas importantes:
    - Los textos de prueba están relacionados con Alan Turing e historia de la IA
    - El endpoint espera JSON con estructura específica: dataWithEmbeddings y collection_name
    - El timeout de 30 segundos previene bloqueos en operaciones lentas
    - La verificación de dimensionalidad es crítica para compatibilidad con Milvus
    """
    from app.embedding_generator import embedding_generator

    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    
    # Generar embeddings vectoriales para cada texto
    vectors = embedding_fn.encode(docs)
    
    # Estructurar datos en el formato esperado por Milvus
    data = [
        {
            "id": embedding_generator.generate_id(docs[i]), 
            "vector": vectors[i].tolist(), 
            "text": docs[i]
        }
        for i in range(len(vectors))
    ]

    # Nombre de la colección de prueba en Milvus
    collection_name = 'Prueba'

    # Usar httpx.AsyncClient para requests asíncronos
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{conf.DB_URL}:{conf.DB_PORT}/upload-embeddings",
            json={"dataWithEmbeddings": data, "collection_name": collection_name},
            timeout=30.0  # Timeout opcional para evitar bloqueos
        )
    
        # Validación: el endpoint debe responder con éxito (200)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_context_endpoint():
    """
    Prueba de integración asíncrona para el endpoint de obtención de contexto.
    
    Este test verifica el endpoint REST API '/get-context' que realiza búsqueda semántica
    en la base de datos vectorial Milvus usando una consulta textual y su embedding.

    Objetivos del test:
    1. Validar que el endpoint '/get-context' procesa correctamente consultas semánticas
    2. Verificar la comunicación completa: desde la consulta hasta la respuesta contextual
    3. Comprobar que el endpoint responde con status HTTP 200 (éxito)
    4. Asegurar que los parámetros requeridos son aceptados correctamente
    5. Testear el flujo asíncrono de búsqueda semántica con embeddings

    Flujo detallado:
    1. Configura el nombre de la colección donde se realizó la inserción previa
    2. Prepara una consulta de prueba relevante para los datos insertados
    3. Genera el embedding vectorial para la consulta usando el modelo configurado
    4. Realiza una petición HTTP POST asíncrona al endpoint de obtención de contexto
    5. Incluye todos los parámetros requeridos: collection_name, query y query_embedding
    6. Valida que el servidor responda con status code 200 indicando éxito

    Este test asume que:
    - La colección 'Prueba' existe y contiene datos previamente insertados
    - Los datos incluyen información sobre Alan Turing e inteligencia artificial
    - El mismo modelo de embeddings se usó para inserción y búsqueda

    Args:
        No recibe parámetros directos, pero depende de:
        - embedding_fn: Modelo SentenceTransformer previamente inicializado
        - conf.DB_URL: URL de la base de datos desde configuración
        - conf.DB_PORT: Puerto de la base de datos desde configuración

    Raises:
        AssertionError: Si el status code de la respuesta no es 200
        httpx.HTTPError: Si hay errores de conexión con el endpoint
        Exception: Para cualquier otro error durante la ejecución

    Notas importantes:
    - La consulta "who was Alan Turing?" está diseñada para encontrar coincidencias
    - El embedding se genera con el mismo modelo usado para los datos insertados
    - El timeout de 30 segundos previene bloqueos en búsquedas lentas
    - El test valida el status code pero no el contenido de la respuesta
    """
    # Configuración de la colección de prueba (debe coincidir con la de inserción)
    collection_name = 'Prueba'
    
    # Consulta de prueba
    query = "who was Alan Turing?"

    # Generación del embedding vectorial para la consulta
    query_embedding = embedding_fn.encode(query) # Extraer el primer embedding de la lista

    # Usar httpx.AsyncClient para requests asíncronos
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{conf.DB_URL}:{conf.DB_PORT}/get-context",
            json={"collection_name": collection_name, "query": query, "query_embedding": query_embedding[0].tolist()},
            timeout=30.0  # Timeout opcional para evitar bloqueos
        )
    
        # Validación: el endpoint debe responder con éxito (200)
        assert response.status_code == 200