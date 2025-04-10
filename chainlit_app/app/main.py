import chainlit as cl
from pydantic import BaseModel
from app.aim_tracker import end_aim_run, start_aim_run
from app.databases import get_context_from_db, post_embeddings, RetrieveData
from app.embeddingGenerator import EmbeddingGenerator
from app.models import get_conversational_answer
from app.pdfExtractor import extract_text_from_pdf
from app.splitter import split_markdown_text
from chainlit.input_widget import Select, Slider


# from langchain.memory import ConversationBufferMemory

embedding_generator = EmbeddingGenerator()
collection_name = "prueba_lines"

@cl.on_chat_start
async def start():
    cl.user_session.set("session_number", 1)
    # cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("aim_run", start_aim_run())
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="model",
                values=[
                    "llama3.1",
                    "gemma3:1b",
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
async def vectordb_results_step(query: str):
    settings = cl.user_session.get("settings")
    query_embedding = await cl.make_async(embedding_generator.get_embeddings)([query])
    query_embedding = query_embedding[0]
    print(f"Query embedding: {query_embedding}")
    results = await cl.make_async(get_context_from_db)(
        collection_name=collection_name,
        query=query,
        query_embedding=query_embedding,
    )
    context = await context_step(results)
    return context

async def context_step(results: list[RetrieveData]) -> str:
    """Procesa resultados de bases vectoriales (siempre lista)"""
    
    context_sections = []
    context_texts = []
    for result in results:
        # Convertir el diccionario en una instancia de RetrieveData
        section = [
            f"🏷️ ID: {result.id}",
            *[f"📋 {param}: {value}" for param, value in result.metadata.items()],
            f"\n{'━'*40}",
            result.text,
            f"{'━'*40}"
        ]
        context_sections.append("\n".join(section))
        context_texts.append(result.text)
    
    full_output = "\n\n".join(context_sections) if context_sections else "Sin coincidencias"
    context_texts = "\n\n".join(context_texts) if context_texts else "Sin coincidencias"
    cl.context.current_step.output = full_output
    return context_texts

@cl.step
async def llm_step(query, context, **kwargs):
    chat_context = cl.chat_context.to_openai()
    print(f"Chat context: {chat_context}")
    aim_run = cl.user_session.get("aim_run")
    respuesta = await cl.make_async(get_conversational_answer)(
        query, context, chat_context, aim_run, **kwargs
    )
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
            print(f"Extrayendo texto de `{file.name}`...")
            text = extract_text_from_pdf(file.path)

            # Splittear el texto en chunks semánticos
            print(f"Splitteando texto de `{file.name}`...")
            chunks = split_markdown_text(text)

            # Generar los embeddings de los chunks
            print(f"Generando embeddings de `{file.name}`...")
            embeddings = await cl.make_async(embedding_generator.get_embeddings)(chunks)

            # Refinar los chunks según la similitud coseno
            # print(f"Refinando chunks de `{file.name}`...")
            # refined_chunks = refine_split_by_similarity(chunks, embeddings)

            # Formatear y cargar los embeddings en la base de datos
            print(f"Formateando embeddings de `{file.name}`...")
            embeddings_data = await cl.make_async(embedding_generator.format_for_database)(
                embeddings, chunks
            )
            print("Embeddings formateados")
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
        "frequency_penalty": settings["frequency_penalty"],
    }
    respuesta = await llm_step(query=query, context=context, **kwargs)
    msg.content = f"{respuesta}"

    await msg.update()


@cl.on_chat_end
async def close():
    aim_run = cl.user_session.get("aim_run")
    end_aim_run(aim_run)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
