import hashlib
import sys
import os

from sentence_transformers import SentenceTransformer

# Agregar el directorio raíz al path para importar el módulo Milvus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Inicializar modelo de embeddings con dimensión conocida y verificable
embedding_fn = SentenceTransformer('all-mpnet-base-v2', device='cpu')

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


# def test_retrive():
#     from vectordbs.Milvus.main import get_context_with_filters
#     collection_name = 'Prueba'
#     embedding_fn = SentenceTransformer('all-MiniLM-L6-V2', device='cpu')
#     query = embedding_fn.encode(["who was Alan Turing?"])
#     response = get_context_with_filters(collection_name=collection_name, query=query)
#    print(response)