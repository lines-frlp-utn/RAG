import hashlib
import sys
import os

from sentence_transformers import SentenceTransformer

# Agregar el directorio raíz al path para importar el módulo Milvus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Inicializar modelo de embeddings con dimensión conocida y verificable
embedding_fn = SentenceTransformer('all-mpnet-base-v2', device='cpu')


# ______________________________________ TESTS ______________________________________

def test_upload_pdf_to_milvus_db():
    """
    Prueba de integración para la carga de embeddings a Milvus con datos simulados.
    
    Este test verifica el flujo completo de carga de datos vectoriales a Milvus,
    simulando el proceso que normalmente se haría con un PDF pero usando textos
    predefinidos para mayor control y reproducibilidad.

    Objetivos del test:
    1. Validar que la función upload_pdf_to_vector_db puede procesar datos con embeddings
    2. Verificar que los embeddings generados tienen la dimensión correcta (768)
    3. Comprobar que la estructura de datos es compatible con Milvus
    4. Detectar errores de dimensionalidad o formato durante la inserción
    5. Confirmar que el proceso completo se ejecuta sin excepciones

    Flujo detallado:
    1. Inicializa el modelo de embeddings 'all-mpnet-base-v2' (768 dimensiones)
    2. Genera embeddings para textos de prueba sobre inteligencia artificial
    3. Verifica críticamente que la dimensionalidad sea 768 para prevenir errores
    4. Genera IDs únicos usando hash MD5 para consistencia en las pruebas
    5. Estructura los datos en formato compatible con Milvus
    6. Ejecuta la función de carga manejando posibles excepciones
    7. Proporciona feedback claro sobre el éxito o error de la operación

    Args:
        No recibe parámetros directamente, pero depende de:
        - SentenceTransformer: Modelo de embeddings 'all-mpnet-base-v2'
        - upload_pdf_to_vector_db: Función del módulo Milvus a testear
        - hashlib: Para generación de IDs consistentes

    Raises:
        AssertionError: Si la dimensión de los embeddings no es 768
        Exception: Si falla la conexión, inserción o hay error en Milvus
        ImportError: Si no se encuentran las dependencias necesarias

    Notas importantes:
    - El modelo 'all-mpnet-base-v2' genera embeddings de 768 dimensiones, por lo que si se modifica en el futuro, deberá cambiarse el modelo para la verificación.
    - Los IDs se generan mediante hash MD5 para garantizar consistencia entre ejecuciones
    - La colección 'Prueba' debe existir en Milvus y estar configurada con dim=768
    - El test incluye manejo de excepciones para debugging detallado
    - Los textos de prueba son históricamente relevantes para IA
    """
    from vectordbs.Milvus.main import upload_pdf_to_vector_db
    
    # Textos de ejemplo cuidadosamente seleccionados sobre historia de la IA
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    
    # Generar embeddings vectoriales para cada texto
    vectors = embedding_fn.encode(docs)

    # Verificación de dimensionalidad
    assert len(vectors[0]) == 768, f"Expected 768, got {len(vectors[0])}"
    
    def generate_id(text):
        """
        Genera un ID único y determinístico basado en el hash MD5 del texto.
        
        Args:
            text (str): Texto original para generar el ID
            
        Returns:
            int: ID numérico único derivado del hash MD5, módulo 10^8
        """
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 8)
    
    # Estructurar datos en el formato esperado por Milvus
    data = [
        {
            "id": generate_id(docs[i]), 
            "vector": vectors[i], 
            "text": docs[i]
        }
        for i in range(len(vectors))
    ]

    # print("Data has", len(data), "entities, each with fields: ", data[0].keys())
    # print("Vector dim:", len(data[0]["vector"]))

    # Nombre de la colección de prueba en Milvus
    collection_name = 'Prueba'

    # Ejecución con manejo robusto de excepciones
    try:
        result = upload_pdf_to_vector_db(dataWithEmbeddings=data, collection_name=collection_name)
        print("Upload successful:", result)
    except Exception as e:
        print("Error:", str(e))
        raise  # Relanza la excepción para que pytest la capture


def test_get_context_with_filters():
    """
    Prueba de integración para la función de recuperación de contexto con filtros en Milvus.
    
    Este test verifica que el sistema de búsqueda semántica puede recuperar información
    relevante basada en una consulta textual y su embedding correspondiente, aplicando
    filtros sobre la colección especificada.

    Objetivos del test:
    1. Validar que get_context_with_filters procesa correctamente objetos QueryData
    2. Verificar que se obtienen resultados relevantes para consultas específicas
    3. Comprobar que el sistema devuelve respuestas no nulas para consultas válidas
    4. Asegurar que se recupera al menos un resultado para consultas con coincidencias esperadas

    Flujo detallado:
    1. Define una consulta de prueba sobre Alan Turing (relevante para los datos insertados)
    2. Genera el embedding vectorial para la consulta usando el modelo configurado
    3. Construye el objeto QueryData con los parámetros requeridos
    4. Ejecuta la función de búsqueda con filtros
    5. Valida que la respuesta sea estructurada y contenga resultados relevantes

    Args:
        No recibe parámetros directos, pero depende de:
        - embedding_fn: Modelo de embeddings previamente inicializado (all-mpnet-base-v2)
        - QueryData: Esquema Pydantic para estructuración de consultas
        - get_context_with_filters: Función bajo prueba del módulo Milvus

    Raises:
        AssertionError: Si la respuesta es None o vacía cuando se esperaban resultados
        ImportError: Si las dependencias necesarias no están disponibles
        Exception: Para cualquier otro error durante la ejecución de la búsqueda

    Notas importantes:
    - Asume que la colección 'Prueba' existe y contiene datos sobre Alan Turing
    - Utiliza el mismo modelo de embeddings que se usó para la inserción (all-mpnet-base-v2)
    - La consulta "who was Alan Turing?" debería encontrar coincidencias en los datos de prueba
    - El test valida tanto la existencia como la relevancia de los resultados
    """
    from vectordbs.Milvus.main import get_context_with_filters, QueryData

    # Configuración de la colección de prueba (debe coincidir con la de inserción)
    collection_name = 'Prueba'
    
    # Consulta de prueba
    query = "who was Alan Turing?"

    # Generación del embedding vectorial para la consulta
    query_embedding = embedding_fn.encode(query) # Extraer el primer embedding de la lista
    
    # Construcción del objeto de consulta con la estructura esperada por el sistema
    query_data = QueryData(
        collection_name=collection_name,
        query=query,  # Texto original de la consulta
        query_embedding=query_embedding.tolist()  # Embedding como lista de floats
    )
    
    # Ejecución de la búsqueda semántica con filtros aplicados
    response = get_context_with_filters(query_data)

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