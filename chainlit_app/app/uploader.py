from app.models import embedding_model
from chromadb import PersistentClient
from app.parser import prepare_chunks_from_docs
from langchain_community.vectorstores import Chroma

client = PersistentClient("./database/")

def upload_pdf_to_database(text_file, theme, subtheme, collection_name):
    
    chunks = prepare_chunks_from_docs(
        file_path=text_file, 
        theme=theme, 
        subtheme=subtheme,
    )

    for chunk in chunks:
        print(chunk)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        client=client,
    )

    print(text_file + " cargado correctamente...")


def get_context_with_filters(collection_name, theme, subtheme, query):
    if collection_name != " ":
        collection = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            client=client,
        )

        if theme != "-":
            retriever = collection.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "filter": {"theme": theme},
                    "k": 6,
                },
            )

        else:
            retriever = collection.as_retriever(
                search_kwargs={"k": 6},
                search_type="mmr",
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
