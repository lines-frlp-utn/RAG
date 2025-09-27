import pytest
from sentence_transformers import SentenceTransformer

# Inicializar modelo de embeddings con dimensión conocida y verificable
embedding_fn = SentenceTransformer('all-mpnet-base-v2', device='cpu')

@pytest.mark.integration
async def test_upload_embeddings_endpoint():
    """
    Prueba de integración para el endpoint de carga de embeddings a la base de datos vectorial.
    
    Este test verifica el flujo completo de carga de embeddings, desde la generación
    de vectores hasta la inserción en la base de datos Milvus mediante la función
    post_embeddings. Es una prueba de integración que valida la comunicación real
    entre los componentes del sistema.

    Objetivos del test:
    1. Validar que la generación de embeddings funciona correctamente
    2. Verificar que la estructura de datos es compatible con el endpoint
    3. Comprobar que la función post_embeddings comunica exitosamente con la base de datos
    4. Asegurar que el proceso completo de carga se ejecuta sin errores
    5. Confirmar que la respuesta indica éxito en la operación

    Flujo detallado:
    1. Prepara textos de ejemplo sobre historia de la inteligencia artificial
    2. Genera embeddings vectoriales usando el modelo SentenceTransformer configurado
    3. Estructura los datos en el formato esperado por Milvus:
       - IDs únicos generados mediante embedding_generator.generate_id()
       - Vectores convertidos a lista de Python con .tolist()
       - Textos originales preservados para contexto
    4. Ejecuta la función post_embeddings con los datos preparados
    5. Valida que la respuesta indica éxito y no contiene errores

    Características de la prueba de integración:
    - Requiere que la base de datos Milvus esté disponible y ejecutándose
    - Utiliza componentes reales del sistema (no mocks)
    - Valida la comunicación end-to-end entre los módulos
    - Está marcada con @pytest.mark.integration para ejecución selectiva

    Args:
        No recibe parámetros directos, pero depende de:
        - embedding_fn: Modelo SentenceTransformer previamente inicializado
        - embedding_generator: Módulo con función generate_id() para IDs únicos
        - post_embeddings: Función de la capa de datos para inserción en Milvus

    Raises:
        AssertionError: Si la respuesta no indica éxito o contiene errores
        Exception: Para cualquier error durante la generación o inserción de embeddings

    Notas importantes:
    - Los textos de prueba están relacionados con Alan Turing e historia de la IA
    - La colección 'Prueba' debe existir en Milvus o crearse automáticamente
    - Este test puede ser lento debido a la comunicación con la base de datos real
    - Debe ejecutarse solo en entornos con base de datos disponible
    """
    from app.embedding_generator import embedding_generator
    from app.databases import post_embeddings

    # Textos de ejemplo para generar embeddings
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

    # Llamada a la función de subida de embeddings
    response = post_embeddings(data, collection_name )
    
    # Validación: el endpoint debe responder con éxito (200)
    assert "success" in response, "La respuesta debería indicar éxito"
    assert "error" not in response, "La respuesta no debería indicar error"


@pytest.mark.integration
async def test_get_context_endpoint():
    """
    Prueba de integración para el endpoint de obtención de contexto desde la base de datos vectorial.
    
    Este test verifica el flujo completo de recuperación semántica, validando que la función
    get_context_from_db puede comunicarse correctamente con la base de datos Milvus y recuperar
    resultados relevantes para una consulta específica.

    Objetivos del test:
    1. Validar la comunicación end-to-end con la base de datos vectorial
    2. Verificar que se recuperan resultados relevantes para consultas semánticas
    3. Comprobar la estructura y calidad de los resultados devueltos
    4. Asegurar que el texto recuperado contiene información contextual pertinente

    Flujo detallado:
    1. Configura la colección de prueba donde previamente se insertaron datos
    2. Prepara una consulta de prueba relevante ("who was Alan Turing?")
    3. Genera el embedding vectorial para la consulta usando el modelo configurado
    4. Ejecuta la función get_context_from_db para obtener contexto semántico
    5. Valida exhaustivamente la estructura y contenido de la respuesta

    Args:
        No recibe parámetros directos, pero depende de:
        - get_context_from_db: Función de recuperación de contexto desde la base de datos
        - embedding_fn: Modelo de embeddings previamente inicializado
        - Colección 'Prueba' con datos previamente insertados en Milvus

    Raises:
        AssertionError: Si la respuesta es None, vacía o no cumple con la estructura esperada
        Exception: Para cualquier error durante la comunicación con la base de datos

    Notas importantes:
    - Requiere que la colección 'Prueba' exista y contenga datos relevantes
    - Asume que los datos insertados incluyen información sobre Alan Turing
    - Valida tanto la existencia como la relevancia semántica de los resultados
    - Es una prueba de integración que requiere servicios externos ejecutándose
    """
    from app.databases import get_context_from_db
    # Configuración de la colección de prueba (debe coincidir con la de inserción)
    collection_name = 'Prueba'
    
    # Consulta de prueba
    query = "who was Alan Turing?"

    # Generación del embedding vectorial para la consulta
    query_embeddings = embedding_fn.encode(query)

    query_embedding = query_embeddings.tolist()  # Convertir a lista de Python
    
    # Llamada a la función de obtención de contexto
    response = get_context_from_db( collection_name, query, query_embedding)
    
    
    print(f"Response: {response}")

    # Validacion de respuesta no debe ser nula
    assert response is not None, "La respuesta no debería ser None"
    
    assert isinstance(response, list), f"La respuesta debería ser una lista, no {type(response)}"
    assert len(response) > 0, "Debería haber al menos un resultado para esta consulta"
    
    # Validar la estructura del primer resultado
    first_result = response[0]
    print(f"Primer resultado: {first_result}")
    
    # Validar los valores específicos
    assert first_result.id is not None, "El ID no debería ser None"
    assert isinstance(first_result.id, (str, int)), f"ID debería ser str o int, no {type(first_result.id)}"
    
    assert first_result.text is not None, "El texto no debería ser None"
    assert isinstance(first_result.text, str), f"Text debería ser str, no {type(first_result.text)}"
    assert len(first_result.text) > 0, "El texto no debería estar vacío"
    
    # Validar que el texto contiene información relevante
    assert "Alan Turing" in first_result.text, "El texto debería mencionar a Alan Turing"
    assert "research" in first_result.text or "AI" in first_result.text, "El texto debería contener términos relevantes"