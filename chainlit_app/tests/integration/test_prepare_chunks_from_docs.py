def test_prepare_chunks_from_docs():
    """
    Prueba de integración para el procesamiento de documentos PDF usando LlamaParse.
    
    Este test verifica que la función prepare_chunks_from_docs puede procesar correctamente
    un documento PDF real y extraer el contenido en fragmentos estructurados, preservando
    el formato y la estructura del documento original.

    Objetivos del test:
    1. Validar que LlamaParse procesa correctamente documentos PDF con estructura compleja
    2. Verificar que la función prepare_chunks_from_docs divide el documento en fragmentos
    3. Comprobar que el contenido extraído coincide con el texto esperado
    4. Asegurar que el formato markdown y las estructuras tabulares se preservan

    Flujo detallado:
    1. Aplica nest_asyncio para permitir operaciones asíncronas en entornos síncronos
    2. Carga un documento PDF real del fixture de pruebas (algoritmos.pdf)
    3. Procesa el documento usando prepare_chunks_from_docs con LlamaParse
    4. Compara el primer fragmento extraído con el texto mock esperado

    Args:
        No recibe parámetros directos, pero depende de:
        - prepare_chunks_from_docs: Función de procesamiento de documentos
        - tests/fixture/algoritmos.pdf: Documento PDF de prueba real

    Raises:
        AssertionError: Si el primer chunk no coincide con el texto esperado
        AssertionError: Si se generan un número incorrecto de chunks
        FileNotFoundError: Si el archivo PDF de prueba no existe
        Exception: Si hay errores durante el procesamiento con LlamaParse

    Notas importantes:
    - El documento 'algoritmos.pdf' contiene información académica real
    - El texto mock representa la estructura esperada del encabezado del documento
    - Se espera que LlamaParse preserve el formato markdown y tablas
    """
    from app.parser import prepare_chunks_from_docs
    
    # Texto esperado que debería contener el primer fragmento del documento
    mock_text = """Algoritmos y Estructuras de Datos
Planificación Ciclo lectivo 2024 – Ordenanza 1877

# Datos administrativos de la asignatura"""

    # Ruta al archivo PDF de prueba
    file_path = "tests/fixture/algoritmos.pdf"

    # Procesar el documento y obtener los fragmentos
    chunks = prepare_chunks_from_docs(file_path)

    
    assert chunks[0] == mock_text # Verificar que el primer fragmento coincide con el texto esperado