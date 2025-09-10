from typing import Dict

import chainlit as cl
from app.aim_tracker import end_aim_run, start_aim_run
from app.auth import Role, authenticate
from app.databases import RetrieveData, get_context_from_db, post_embeddings
from app.embedding_generator import embedding_generator
from app.langgraph_flow import app as langgraph_app
from app.models import get_conversational_answer
from app.parser import extract_text_from_pdf
from app.splitter.markdown_splitter import split_markdown_text as markdown_split
from app.splitter.semantic_splitter import split_semantic as semantic_split
from chainlit.input_widget import Select, Slider
from chainlit.types import ThreadDict
from langchain.memory import ConversationBufferMemory

collection_name = "prueba_lines"


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Callback de autenticación con sistema robusto
@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    return await authenticate(username, password)


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
):
    email = raw_user_data.get("email")
    display_name = raw_user_data.get("name", "")

    if not email:
        print("OAuth callback: Email no proporcionado")
        return None

    return cl.User(
        identifier=email,
        metadata={
            "role": Role.CLIENT.value,
            "provider": provider_id,
            "display_name": display_name,
            "email": email,
        },
    )


@cl.on_chat_start
async def start():
    cl.user_session.set("session_number", 1)
    app_user = cl.user_session.get("user")
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("aim_run", start_aim_run())

    if app_user:
        if not app_user.metadata.get("role"):
            app_user.metadata["role"] = Role.CLIENT.value

        cl.user_session.set("user", app_user)

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="model",
                values=[
                    "llama3.1",
                    "qwen2.5vl",
                ],
                initial_index=1,
            ),
            Select(
                id="splitter",
                label="Tipo de splitter",
                values=[
                    "Markdown",
                    "Semantico",
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

    if app_user:
        display_name = app_user.metadata.get("display_name", app_user.identifier)
        msg = cl.Message(content=f"¡Hola, {display_name}! ¿En qué puedo ayudarte hoy?")
        await msg.send()
    else:
        cl.Message(
            content="Ha habido un error de autenticación. Por favor, vuelve a intentar iniciar sesión."
        ).send()

    await update_settings(settings)


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("settings", settings)


@cl.on_chat_resume
async def resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    settings = cl.user_session.get("settings")
    cl.user_session.set("aim_run", start_aim_run())
    await update_settings(settings)

    chainlit_thread_id = thread.get("id")
    cl.user_session.set("chainlit_thread_id", chainlit_thread_id)

    history_steps = thread.get("steps") or []

    if history_steps:
        for step in history_steps:
            if step.get("type") == "user_message":
                content = step.get("output", "")
                if content:
                    memory.chat_memory.add_user_message(content)
            elif step.get("type") == "assistant_message":
                content = step.get("output", "")
                if content:
                    memory.chat_memory.add_ai_message(content)

    cl.user_session.set("memory", memory)


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
            f"\n{'━' * 40}",
            result.text,
            f"{'━' * 40}",
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
async def on_message(message: cl.Message):
    thread_id = message.thread_id

    user = cl.user_session.get("user")

    if message.elements and user.metadata["role"] == Role.CLIENT.value:
        file = message.elements[0]
        settings = cl.user_session.get("settings")
        try:
            # Extraer el texto del PDF
            print(f"Extrayendo texto de `{file.name}`...")
            text = extract_text_from_pdf(file.path)

            # Splittear el texto en chunks semánticos
            print(f"Splitteando texto de `{file.name}`...")
            splitter_type = settings.get("splitter", "Markdown").lower()

            if splitter_type == "markdown":
                chunks = markdown_split(text)
            elif splitter_type == "semantico":
                chunks = semantic_split(text)
            else:
                raise ValueError(f"Splitter desconocido: {splitter_type}")

            print(f"usando splitter `{splitter_type}`")

            # Generar los embeddings de los chunks
            print(f"Generando embeddings de `{file.name}`...")
            embeddings = await cl.make_async(embedding_generator.get_embeddings)(chunks)

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

            if not message.content or not message.content.strip():
                return

        except Exception as e:
            error_msg = f"Error procesando el archivo `{file.name}`: {str(e)}"
            print(error_msg)

            if not message.content or not message.content.strip():
                return

    if message.content and message.content.strip():
        msg = cl.Message(content="")
        await msg.send()

    query = message.content
    settings = cl.user_session.get("settings")
    state = {
        "question": query,
        "context": None,
        "answer": None,
        "grounded": None,
        "settings": {
            "model": settings["model"],
            "temperature": settings["temperature"],
            "frequency_penalty": settings["frequency_penalty"],
        },
    }
    result = await langgraph_app.ainvoke(state)
    msg.content = result["answer"]
    await msg.update()

    if user and thread_id:
        try:
            await api.create_message(thread_id, "assistant", result["answer"])
        except Exception as e:
            print(f"Error saving assistant message: {e}")

    memory = cl.user_session.get("memory")
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)


@cl.on_chat_end
async def close():
    aim_run = cl.user_session.get("aim_run")
    if aim_run:
        end_aim_run(aim_run)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
