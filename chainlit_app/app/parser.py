import nest_asyncio
import pymupdf4llm
from app.config import conf
from app.splitter.markdown_splitter import split_text_with_langchain as text_splitter
from llama_parse import LlamaParse

nest_asyncio.apply()

__parser = LlamaParse(
    api_key=conf.LLAMA_PARSE_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="es",  # Optionally you can define a language, default=en
)


def prepare_chunks_from_docs(file_path: str, theme: str, subtheme: str):
    documents = __parser.load_data(file_path=file_path)

    doc_text = []
    for doc in documents:
        doc_text.append(doc.text)

    doc_text = text_splitter.create_documents(
        doc_text, metadatas=[{"source": file_path, "theme": theme, "subtheme": subtheme}]
    )
    chunks = text_splitter.split_documents(documents=doc_text)

    return chunks


def extract_text_from_pdf(pdf_path):
    text = pymupdf4llm.to_markdown(pdf_path)

    return text
