import nest_asyncio
import pymupdf4llm
from app.config import conf
from app.splitter.markdown_splitter import split_text_with_langchain
from llama_parse import LlamaParse

nest_asyncio.apply()

__parser = LlamaParse(
    api_key=conf.LLAMA_PARSE_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=4,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="es",  # Optionally you can define a language, default=en
)


def prepare_chunks_from_docs(file_path: str):
    documents = __parser.load_data(file_path=file_path)

    doc_text = []
    for doc in documents:
        doc_text.append(doc.text)
    
    chunks = split_text_with_langchain(doc_text)

    return chunks


def extract_text_from_pdf(pdf_path):
    text = pymupdf4llm.to_markdown(pdf_path)

    return text
