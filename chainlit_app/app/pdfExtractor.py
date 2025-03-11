import pymupdf4llm  

def extract_text_from_pdf(pdf_path):
    texts = pymupdf4llm.to_markdown(pdf_path)
    
    return texts
