import chainlit as cl
from app.chunk_visualization import make_umap

# from app.uploader import get_context_with_filters, get_db, upload_pdf_to_database
from app.databases import get_context_from_db, post_embeddings
from app.embeddingGenerator import EmbeddingGenerator, extract_text_from_pdf
from app.models import embedding_model, get_conversational_answer
from app.splitter import pdf_to_chunks
from chainlit.input_widget import Select, Slider, Switch
from langchain.memory import ConversationBufferMemory

embedding_generator = EmbeddingGenerator()
collection_name = "prueba_lines"


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


@cl.on_chat_start
async def start():
    cl.user_session.set("session_number", 1)

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    settings = await cl.ChatSettings(
        [
            Select(
                id="collection",
                label="collection",
                values=[
                    "prueba_lines",
                ],
                initial_index=0,
            ),
            Select(
                id="theme",
                label="theme",
                values=[
                    "-",
                ],
                initial_index=0,
            ),
            Select(
                id="subtheme",
                label="subtheme",
                values=["-"],
                initial_index=0,
            ),
        ]
    ).send()
    await update_settings(settings)

    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Por favor suba un archivo PDF para continuar!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Procesando archivo `{file.name}`...")
    await msg.send()
    texts = extract_text_from_pdf(file.path)
    embeddings = await cl.make_async(embedding_generator.format_for_database)(texts)

    result = await cl.make_async(post_embeddings)(
        collection_name=collection_name, dataWithEmbeddings=embeddings
    )

    msg.content = f"Archivo `{file.name}` cargado exitosamente, `{result}`"
    await msg.update()


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", settings)


@cl.step
async def vectordb_results_step(query):
    settings = cl.user_session.get("settings")
    query_embedding = await cl.make_async(embedding_generator.get_embeddings)(query)
    results = await cl.make_async(get_context_from_db)(
        collection_name=settings["collection"],
        query=query_embedding,
    )
    cl.context.current_step.output = results
    context = await context_step(results)
    return context


@cl.step
async def context_step(results):
    context = ""
    for doc in results:
        context = f"{context} {doc}"

    cl.context.current_step.output = context
    return context


@cl.step
async def llm_step(query, context):
    respuesta = await cl.make_async(get_conversational_answer)(query, context)
    return respuesta


@cl.on_message
async def main(message: cl.Message):
    session_number = cl.user_session.get("session_number")
    settings = cl.user_session.get("settings")

    msg = cl.Message(content="")  # Muestra un loader mientras carga el mensaje
    await msg.send()

    # if message.elements:
    #     file = pdf_to_chunks(message.elements[0])
    #     upload_pdf_to_database(
    #         text_file=file.path,
    #         theme="-",
    #         subtheme="-",
    #         collection_name="prueba_lines",
    #     )

    if message.content.startswith("umap"):
        a = 0
        # v_db = get_db(settings["collection"]) if settings["collection"] != " " else vector_db
        # query = message.content[5:]
        # query_embedding = embedding_model.embed_query(query)
        # results = (
        #     get_context_with_filters(
        #         settings["collection"], settings["theme"], settings["subtheme"], query
        #     )
        #     if settings["collection"] != " "
        #     else v_db.similarity_search(query)
        # )
        # retrieved_embeddings = []
        # for doc in results:
        #     retrieved_embeddings.append(embedding_model.embed_query(doc.page_content))
        # umap_path = await cl.make_async(
        #     make_umap
        # )(
        #     v_db, retrieved_embeddings, query_embedding, query, session_number
        # )  # TODO: Arreglar porque ya no tenemos vector_db, tenemos retriever, capaz en uploader habria que hacer otra funcion que traiga la coleccion
        # msg.elements = [cl.Image(path=umap_path, name="umap", display="inline")]
    else:
        query = message.content
        context = await vectordb_results_step(query)
        respuesta = await llm_step(query, context)
        msg.content = f"{respuesta}"

    await msg.update()  # actualizamos el mensaje con los nuevos datos


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
