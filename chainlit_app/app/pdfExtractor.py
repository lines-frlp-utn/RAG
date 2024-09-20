import pymupdf  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        texts.append(text)
    return texts
