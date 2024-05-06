from app.models import embedding_model
from app.splitter import text_splitter
from chromadb import PersistentClient

# from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import HTMLHeaderTextSplitter
from llmsherpa.readers import LayoutPDFReader

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

client = PersistentClient("./database/")


def upload_pdf_to_database(text_file, theme, subtheme, collection_name):
    doc = pdf_reader.read_pdf(text_file)
    content = doc.to_html()

    chunks = html_splitter.split_text(content)
    # html_splited_text = html_splitter.split_text(content)
    # chunks = text_splitter.split_documents(html_splited_text)
    for chunk in chunks:
        chunk.metadata.update(
            {
                "source": text_file,
                "theme": theme,
                "subtheme": subtheme,
            }
        )
        print(chunk)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        client=client,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(text_file + " cargado correctamente...")


def get_context_with_filters(collection_name, theme, subtheme, query):
    if collection_name != " ":
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
        )

        if theme != "-":
            retriever = collection.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "filter": {"theme": theme},
                    "score_threshold": 0.5,
                    "k": 4,
                },
            )

        else:
            retriever = collection.as_retriever(
                search_kwargs={"score_threshold": 0.2, "k": 4},
                search_type="similarity_score_threshold",
            )
    else:
        return []
    response = retriever.invoke(query)
    final_response = []
    for doc in response:
        if doc not in final_response:
            final_response.append(doc)
    print(final_response)
    return final_response


def get_db(collection_name):
    if collection_name != " ":
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            client=client,
        )
