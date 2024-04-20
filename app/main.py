import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

from app.models import embedding_model, get_conversational_answer
from app.splitter import pdf_to_chunks
from app.umap import make_umap


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


@cl.on_chat_start
async def start():
    cl.user_session.set("session_number", 1)

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    files = None

    # Se envia dos botones de accion al comienzo del chat
    # tener encuenta que todos los "Ask" tienen un timeout en segundos, si no se realiza nada antes del timeout, tira un error de timeout.
    # res = await cl.AskActionMessage(
    #     content="Pick an action!",
    #     actions=[
    #         cl.Action(name="Chat", value="chat", label="✅ ChatBot"),
    #         cl.Action(name="Cargar PDF", value="pdf", label="🔥 Cargar PDF"),
    #     ],
    # ).send()

    # # Chequeando la opcion elegida
    # if res and res.get("value") == "chat":
    #     # esta opcion aun no funciona, deberiamos tener una base de datos con archivos para que la inteligencia tenga un contexto con el cual responder
    #     # esperando la respuesta del usuario
    #     name = await cl.AskUserMessage(
    #         content="Bienvenido! ¿Cual es tu nombre?",
    #     ).send()
    #     if res:
    #         await cl.Message(
    #             content=f"Hola {name['output']}! ¿de que querias hablar hoy? ",
    #         ).send()
    # if res and res.get("value") == "pdf":
    #     while files is None:
    #         # Esperando que el usuario cargue un archivo pdf
    #         files = await cl.AskFileMessage(
    #             content="Please upload a text file to begin!",
    #             accept=["text/csv", "application/pdf"],
    #             max_size_mb=20,
    #             timeout=180,
    #         ).send()
    #     # Si el usuario carga varios archivos, en esta ocasion se lee solo el primero
    #     text_file = files[0]
    #     if text_file.type == "application/pdf":
    #         chunks = pdf_to_chunks(text_file)
    #         vector_db = Chroma.from_documents(
    #             documents=chunks, embedding=embedding_model
    #         )

    #         # results = vector_db.similarity_search(question['output'])

    #         cl.user_session.set("vector_db", vector_db)

    #     # Mostramos un mensaje donde simplemente decimos el nombre del archivo y la longitud de caractaeres
    #     await cl.Message(
    #         content=f"archivo '{text_file.name}' type: '{text_file.type}', size: {text_file.size}, N° chunks: {len(chunks)}, subido correctamente \n ya puedes hacer tu pregunta"
    #     ).send()


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


# decorator que define lo que sucede cuando el usuario envia un mensaje
@cl.on_message
async def main(message: cl.Message):
    vector_db = cl.user_session.get("vector_db")
    session_number = cl.user_session.get("session_number")

    msg = cl.Message(content="")  # Muestra un loader mientras carga el mensaje
    await msg.send()

    if message.elements:
        chunks = pdf_to_chunks(message.elements[0])
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./app/data/chroma_db",
        )
        cl.user_session.set("vector_db", vector_db)

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
