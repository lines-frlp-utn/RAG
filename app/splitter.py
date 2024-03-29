from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=200, add_start_index=True
)


def pdf_to_chunks(text_file):
    loader = PyPDFLoader(text_file.path)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)
    return chunks
