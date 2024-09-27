import chainlit as cl
from app.databases import get_context_from_db, post_embeddings
from app.embeddingGenerator import EmbeddingGenerator
from app.pdfExtractor import extract_text_from_pdf
from app.splitter import split_text_with_langchain, refine_split_by_similarity
from app.models import get_conversational_answer
from chainlit.input_widget import Select, Slider
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
                id="model",
                label="model",
                values=[
                    "llama3.1",
                    "gemma2:2b",
                ],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="temperature",
                min=0,
                max=1,
                step=0.1,
                initial=0,
            ),
            Slider(
                id="frequency_penalty",
                label="frequency penalty",
                min=0,
                max=1,
                step=0.1,
                initial=0,
            ),
        ]
    ).send()
    
    await update_settings(settings)


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", settings)

@cl.step
async def vectordb_results_step(query):
    settings = cl.user_session.get("settings")
    query_embedding = await cl.make_async(embedding_generator.get_embeddings)([query])
    results = await cl.make_async(get_context_from_db)(
        collection_name=collection_name,
        query=query_embedding,
    )
    cl.context.current_step.output = results
    context = await context_step(results)
    return context

@cl.step
async def context_step(results):
    context = ""
    for doc in results:
        context = f"{context} \n {doc}"

    cl.context.current_step.output = context
    return context

@cl.step
async def llm_step(query, context, **kwargs):
    chat_context = cl.chat_context.to_openai()
    print(f"Chat context: {chat_context}")
    respuesta = await cl.make_async(get_conversational_answer)(query, context, chat_context, **kwargs)
    return respuesta

@cl.on_message
async def main(message: cl.Message):
    session_number = cl.user_session.get("session_number")
    settings = cl.user_session.get("settings")
    
    if message.elements:
        file = message.elements[0]
        # msg = cl.Message(content=f"Procesando archivo `{file.name}`...")
        # await msg.send()
        try:
            # Extraer el texto del PDF
            texts = extract_text_from_pdf(file.path)
            
            # Splittear el texto en chunks semánticos
            chunks = split_text_with_langchain(texts)
            
            # Generar los embeddings de los chunks
            embeddings = await cl.make_async(embedding_generator.get_embeddings)(chunks)

            # Refinar los chunks según la similitud coseno
            refined_chunks = refine_split_by_similarity(chunks, embeddings)

            # Formatear y cargar los embeddings en la base de datos
            embeddings_data = embedding_generator.format_for_database(refined_chunks)
            result = await cl.make_async(post_embeddings)(
                collection_name=collection_name, dataWithEmbeddings=embeddings_data
            )
            print(f"Archivo `{file.name}` cargado exitosamente, `{result}`")
            # msg.content = f"Archivo `{file.name}` cargado exitosamente, `{result}`"
        except Exception as e:
            # msg.content = f"Error procesando el archivo `{file.name}`: {str(e)}"
            print(f"Error procesando el archivo `{file.name}`: {str(e)}")

    msg = cl.Message(content="")  # Solo muestra el loader si no se envió otro mensaje
    await msg.send()

    query = message.content
    context = await vectordb_results_step(query)
    kwargs = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "frequency_penalty": settings["frequency_penalty"]
    }
    respuesta = await llm_step(
        query=query, context=context, 
        **kwargs
    )
    msg.content = f"{respuesta}"

    await msg.update()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
