import chainlit as cl
from langchain.memory import ConversationBufferMemory

from app.models import embedding_model, get_conversational_answer
from app.splitter import pdf_to_chunks
from app.umap import make_umap
from app.vector_db import vector_db


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


@cl.on_chat_start
async def start():
    cl.user_session.set("session_number", 1)

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))


@cl.step
async def vectordb_results_step(vector_db, query):
    results = vector_db.similarity_search(query)
    cl.context.current_step.output = results
    context = await context_step(results)
    return context


@cl.step
async def context_step(results):
    context = ""
    for doc in results:
        context = f"{context} {doc.page_content}"
    # context = f"{results[0].page_content} {results[1].page_content}"
    cl.context.current_step.output = context
    return context


@cl.step
async def llm_step(query, context):
    respuesta = await cl.make_async(get_conversational_answer)(query, context)
    return respuesta


@cl.on_message
async def main(message: cl.Message):
    session_number = cl.user_session.get("session_number")

    msg = cl.Message(content="")  # Muestra un loader mientras carga el mensaje
    await msg.send()

    if message.elements:
        chunks = pdf_to_chunks(message.elements[0])
        vector_db.add_documents(chunks)

    if message.content.startswith("umap"):
        query = message.content[5:]
        query_embedding = embedding_model.embed_query(query)
        results = vector_db.similarity_search(query)
        retrieved_embeddings = []
        for doc in results:
            retrieved_embeddings.append(embedding_model.embed_query(doc.page_content))
        umap_path = await cl.make_async(make_umap)(
            vector_db, retrieved_embeddings, query_embedding, query, session_number
        )
        msg.elements = [cl.Image(path=umap_path, name="umap", display="inline")]
    else:
        query = message.content
        context = await vectordb_results_step(vector_db, query)
        respuesta = await llm_step(query, context)
        msg.content = f"{respuesta}"

    await msg.update()  # actualizamos el mensaje con los nuevos datos


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
