import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,  # 100 characters
    length_function=len,
)


def split_text(text: str):
    return text_splitter.split_text(text)


def split_pdf(pdf: bytes):
    file_like_object = io.BytesIO(pdf)
    pdf_reader = PdfReader(file_like_object)
    pdf_content = pdf_reader.pages[0].extract_text()
    chunks = text_splitter.split_text(pdf_content)
    return chunks
