import chainlit as cl
from chromadb.errors import InvalidDimensionException
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from app.models import embedding_model
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
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="Chat", value="chat", label="✅ ChatBot"),
            cl.Action(name="Cargar PDF", value="pdf", label="🔥 Cargar PDF"),
        ],
    ).send()

    # Chequeando la opcion elegida
    if res and res.get("value") == "chat":
        # esta opcion aun no funciona, deberiamos tener una base de datos con archivos para que la inteligencia tenga un contexto con el cual responder
        # esperando la respuesta del usuario
        name = await cl.AskUserMessage(
            content="Bienvenido! ¿Cual es tu nombre?",
        ).send()
        if res:
            await cl.Message(
                content=f"Hola {name['output']}! ¿de que querias hablar hoy? ",
            ).send()
    if res and res.get("value") == "pdf":
        while files is None:
            # Esperando que el usuario cargue un archivo pdf
            files = await cl.AskFileMessage(
                content="Please upload a text file to begin!",
                accept=["text/csv", "application/pdf"],
                max_size_mb=20,
                timeout=180,
            ).send()
        # Si el usuario carga varios archivos, en esta ocasion se lee solo el primero
        text_file = files[0]
        if text_file.type == "application/pdf":
            chunks = pdf_to_chunks(text_file)

            # logger.info("storing in chroma")
            try:
                vector_db = Chroma.from_documents(
                    documents=chunks, embedding=embedding_model
                )
            except InvalidDimensionException:
                # logger.info(f"invalid dimension exception")
                vector_db.delete_collection()
                vector_db = Chroma.from_documents(
                    documents=chunks, embedding=embedding_model
                )

            # generar retriever
            # logger.info("retriever")

            retriever = vector_db.as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            )

            ##### backend

            question = await cl.AskUserMessage(
                content=f"archivo '{text_file.name}' type: '{text_file.type}', size: {text_file.size}, N° chunks: {len(chunks)}, subido correctamente \n ya puedes hacer tu pregunta",
                timeout=500,
            ).send()

            # results = vector_db.similarity_search(question['output'])

            retrieved_docs = retriever.invoke(question["output"])

            # prompt = prompt_template.format(context= results, question=question['output'])

            cl.user_session.set("retriever", retriever)
            cl.user_session.set("vector_db", vector_db)

        else:
            with open(text_file.path, "r", encoding="utf-8") as f:
                text = f.read()

        # Mostramos un mensaje donde simplemente decimos el nombre del archivo y la longitud de caractaeres
        await cl.Message(
            content=f"pregunta: {question['output']}, cantidad de chunks encontrados: {len(retrieved_docs)}\nchunk1: {retrieved_docs[0].page_content} \nchunk2: {retrieved_docs[1].page_content}"
        ).send()


# decorator que define lo que sucede cuando el usuario envia un mensaje
@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    vector_db = cl.user_session.get("vector_db")
    session_number = cl.user_session.get("session_number")

    actions = [
        cl.Action(name="action_button", value="example_value", description="Click me!")
    ]

    msg = cl.Message(content="")  # Muestra un loader mientras carga el mensaje
    await msg.send()

    await cl.sleep(2)  # aca iria la logica del back-end (supongo)
    if message.elements:
        images = [
            file for file in message.elements if "image" in file.mime
        ]  # preguntando se se ingreso una imagen en el mensaje.
        with open(images[0].path, "r") as f:
            pass
        msg.elements = [cl.Image(path=images[0].path, name="image", display="inline")]

    query = message.content
    results = vector_db.similarity_search(query)

    if message.content.startswith("umap"):
        query_embedding = embedding_model.embed_query(query)
        umap_path = make_umap(
            vector_db, results, query_embedding, query, session_number
        )
        msg.elements = [cl.Image(path=umap_path, name="umap", display="inline")]

    msg.content = f"resultado: {results[0].page_content}, {results[1].page_content}"
    msg.actions = (
        actions  # cargamos las acciones del mensaje (boton de accion de demostracion)
    )

    await msg.update()  # actualizamos el mensaje con los nuevos datos


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
