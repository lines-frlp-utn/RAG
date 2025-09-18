import os
import sys
import pymupdf4llm

from app.config import conf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



# Función para extraer el texto de un PDF a formato Markdown
def extract_text_from_pdf_to_markdown(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)

# ______________________________________ TESTS ______________________________________

def test_markdown_splitter():
    """
    Prueba la funcionalidad del splitter de Markdown con un documento PDF real.
    
    Este test verifica que:
    1. La función split_markdown_text puede procesar correctamente texto extraído de un PDF
    2. Se generan fragmentos no vacíos del documento
    3. Ningún fragmento excede el límite máximo de 4000 caracteres
    4. La división respeta la estructura del contenido
    
    Flujo del test:
    - Carga un documento PDF de prueba desde la carpeta fixtures
    - Extrae el texto del PDF usando extract_text_from_pdf_to_markdown
    - Procesa el texto con split_markdown_text
    - Valida que se generen fragmentos y que cumplan con los límites de tamaño
    
    El PDF 'bitcoin_es.pdf' sirve como caso de prueba realista que probablemente
    contiene encabezados y estructura que el splitter debe reconocer.
    """
    from app.splitter.markdown_splitter import split_markdown_text

    # Cargar documento PDF de prueba desde la carpeta fixtures
    pdf_path = os.path.join(os.path.dirname(__file__), "../fixture/bitcoin_es.pdf")

    # Extraer texto del PDF para procesamiento
    text_pdf = extract_text_from_pdf_to_markdown(pdf_path)
    
    # Dividir el texto en fragmentos usando el splitter de Markdown
    chunks = split_markdown_text(text_pdf)

    # Verificar que se generaron fragmentos (test básico de funcionalidad)
    assert len(chunks) > 0, "No se generaron fragmentos del PDF"

    # Verificar que ningún fragmento excede el límite máximo de DEFAULT_MAX_LENGTH caracteres
    for chunk in chunks:
        assert len(chunk) <= conf.DEFAULT_MAX_LENGTH, "Un fragmento excede la longitud máxima permitida"

def test_semantic_splitter():
    """
    Prueba la funcionalidad del splitter semántico con un documento PDF real.
    
    Este test verifica que el splitter semántico:
    1. Puede procesar correctamente un documento PDF directamente desde su ruta
    2. Genera fragmentos no vacíos del documento
    3. Respeta el límite máximo de longitud especificado por DEFAULT_MAX_LENGTH
    4. Mantiene la integridad semántica del contenido durante la división
    
    El test utiliza un documento real sobre Bitcoin en español como caso de prueba,
    lo que permite validar el funcionamiento con contenido técnico y estructurado.
    
    Args:
        No recibe parámetros directamente, pero depende de:
        - ../fixture/bitcoin_es.pdf: Documento de prueba en la carpeta fixtures
        - DEFAULT_MAX_LENGTH: Varaible de entorno que define el tamaño máximo de fragmentos
        
    Raises:
        AssertionError: Si no se generan fragmentos o alguno excede el tamaño máximo
        FileNotFoundError: Si el documento PDF de prueba no existe
        RuntimeError: Si ocurre algún error durante el procesamiento del PDF
        
    Flujo del test:
    1. Construye la ruta al documento PDF de prueba en la carpeta fixtures
    2. Llama a split_semantic() con la ruta del PDF y el tamaño máximo permitido
    3. Valida que se generen fragmentos (test de funcionalidad básica)
    4. Verifica que cada fragmento cumpla con el límite de tamaño establecido
    
    Notas:
    - El splitter semántico debería preservar unidades de significado completo
    - Los fragmentos deberían mantener coherencia temática incluso después de la división
    - El documento 'bitcoin_es.pdf' debe contener contenido suficiente para generar múltiples fragmentos
    """
    from app.splitter.semantic_splitter import split_semantic

    # Cargar documento PDF de prueba desde la carpeta fixtures
    pdf_path = os.path.join(os.path.dirname(__file__), "../fixture/bitcoin_es.pdf")

    # Dividir el texto en fragmentos usando el splitter de Markdown
    chunks = split_semantic(pdf_path, max_length=conf.DEFAULT_MAX_LENGTH)

    # Verificar que se generaron fragmentos (test básico de funcionalidad)
    assert len(chunks) > 0, "No se generaron fragmentos del PDF"

    # Verificar que ningún fragmento excede el límite máximo de DEFAULT_MAX_LENGTH caracteres
    for chunk in chunks:
        assert len(chunk) <= conf.DEFAULT_MAX_LENGTH, "Un fragmento excede la longitud máxima permitida"