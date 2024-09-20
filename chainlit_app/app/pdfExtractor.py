import fitz

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")  # Especifica el formato si es necesario
            texts.append(text)
        return texts
    except Exception as e:
        print(f"Error al extraer texto del PDF: {e}")
        return []

