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
        - ./tests/fixture/algoritmos.pdf: Documento PDF de prueba real

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
    file_path = "./tests/fixture/algoritmos.pdf"

    # Procesar el documento y obtener los fragmentos
    chunks = prepare_chunks_from_docs(file_path)

    
    assert chunks[0] == mock_text # Verificar que el primer fragmento coincide con el texto esperado
    assert len(chunks) == 95  # Verificar que se generaron 95 fragmentos en total



def test_extract_text_from_pdf():
    """
    Prueba unitaria para la extracción de texto plano desde documentos PDF.
    
    Este test verifica que la función extract_text_from_pdf puede extraer correctamente
    el contenido textual de un documento PDF, incluyendo formato markdown y estructuras
    tabulares, y que el contenido tiene la longitud esperada.

    Objetivos del test:
    1. Validar que la extracción de texto preserva el formato markdown
    2. Verificar que las estructuras tabulares se convierten correctamente a texto
    3. Comprobar que el texto extraído comienza con el contenido esperado
    4. Asegurar que la longitud total del texto extraído es correcta

    Flujo detallado:
    1. Define el texto esperado que debe aparecer al inicio del documento
    2. Carga el documento PDF real del fixture de pruebas
    3. Extrae el texto usando extract_text_from_pdf
    4. Valida que el inicio del texto coincide con lo esperado
    5. Verifica la longitud total del texto extraído

    Args:
        No recibe parámetros directos, pero depende de:
        - extract_text_from_pdf: Función de extracción de texto
        - ./tests/fixture/algoritmos.pdf: Documento PDF de prueba real

    Raises:
        AssertionError: Si el texto no comienza con el contenido esperado
        AssertionError: Si la longitud del texto no es la esperada
        FileNotFoundError: Si el archivo PDF de prueba no existe

    Notas importantes:
    - El texto mock incluye markdown (# para encabezados) y estructura de tabla
    - La longitud exacta (43939 caracteres) valida la extracción completa del documento
    - La tabla se convierte a formato markdown con pipes y separadores
    - Este test valida la extracción cruda vs el procesamiento inteligente del test anterior
    """
    from app.parser import extract_text_from_pdf

    # Texto esperado que debería aparecer al inicio del documento
    mock_text = """# Algoritmos y Estructuras de Datos
 Planificación Ciclo lectivo 2024 – Ordenanza 1877

|Datos administrativos de la asignatura|Col2|Col3|Col4|
|---|---|---|---|
|Departamento:|Ingeniería en Sistemas de Información|Carrera|Ingeniería en Sistemas de Información|
|Asignatura:|Algoritmos y Estructuras de Datos|||
|Nivel de la carrera|1er año|Duración|Anual|"""

    # Ruta al archivo PDF de prueba
    file_path = "./tests/fixture/algoritmos.pdf"
    
    # Extraer el texto del documento PDF
    text = extract_text_from_pdf(file_path)
    
    assert text.startswith(mock_text) # Verificar que el texto comienza con el contenido esperado
    assert len(text) == 43939 # Verificar que la longitud del texto es la esperada (43939 caracteres)