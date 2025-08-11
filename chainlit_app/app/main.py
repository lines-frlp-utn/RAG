from typing import Dict

import chainlit as cl
from app.aim_tracker import end_aim_run, start_aim_run
from app.auth import Role, create_user, user_exists
from app.databases import RetrieveData, get_context_from_db, post_embeddings
from app.embedding_generator import embedding_generator
from app.models import get_conversational_answer
from app.parser import extract_text_from_pdf
from app.splitter.markdown_splitter import split_markdown_text as markdown_split
from app.splitter.semantic_splitter import split_semantic as semantic_split
from chainlit.input_widget import Select, Slider
from chainlit.types import ThreadDict
from langchain.memory import ConversationBufferMemory
import httpx
from app.config import conf

collection_name = "prueba_lines"


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Callback de autenticación
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username and password:
        user = user_exists(username, password)
        if user.exists is False:
            user = create_user(username, Role.CLIENTE, password, name=username)
            if user:
                print(f"User created: {username}")
                return cl.User(
                    identifier=username,
                    metadata={
                        "role": Role.CLIENTE,
                        "provider": "credentials",
                        "display_name": username,
                    },
                )
            else:
                print(f"Error creating user: {username}")
                return None
        else:
            print(f"User exists: {user}")
            return cl.User(
                identifier=username,
                metadata={
                    "role": user.role_name,
                    "provider": "credentials",
                    "display_name": username,
                },
            )
    else:
        return None


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
):
    email = raw_user_data.get("email")
    display_name = raw_user_data.get("name", "")
    picture = raw_user_data.get("picture", "")

    if not email:
        print("OAuth callback: Email no proporcionado")
        return None
    try:
        user = user_exists(email, "")
        if not user or user.exists is False:
            print("Usuario no existe, creando con OAuth")
            created = create_user(
                email, Role.CLIENTE, "", provider_id, email, picture, name=display_name
            )
            if created:
                role = Role.CLIENTE
        else:
            print("Usuario encontrado")
            role = user.role_name

    except Exception as e:
        print(f"Error durante verificación/creación de usuario: {e}")
        return None

    return cl.User(
        identifier=email,
        metadata={"role": role, "provider": provider_id, "display_name": display_name},
    )


@cl.on_chat_start
async def start():
    cl.user_session.set("session_number", 1)
    app_user = cl.user_session.get("user")
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("aim_run", start_aim_run())
    
    if app_user:
        # Crear nuevo thread usando directamente el endpoint de Iván
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{conf.USERS_API_FULL_URL}/threads/create/{app_user.identifier}")
                response.raise_for_status()
                thread_data = response.json()  # {"thread_id": int, "user_id": int}
                cl.user_session.set("thread_id", thread_data["thread_id"])
                print(f"Created new thread {thread_data['thread_id']} for user {app_user.identifier}")
            except Exception as e:
                print(f"Error creating thread: {e}")
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="model",
                values=[
                    "llama3.1",
                    "qwen2.5vl",
                ],
                initial_index=0,
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
    
    app_user = cl.user_session.get("user")
    if app_user:
        try:            
            # Obtener threads del usuario usando llamada HTTP directa
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{conf.USERS_API_FULL_URL}/threads/{app_user.identifier}")
                response.raise_for_status()
                user_threads = response.json()
                
                if user_threads:
                    last_thread_id = user_threads[-1]["thread_id"]  # Obtenemos el último thread
                    cl.user_session.set("thread_id", last_thread_id)
                    
                    # Obtener mensajes del thread
                    response = await client.get(f"{conf.USERS_API_FULL_URL}/threads/messages/{last_thread_id}")
                    response.raise_for_status()
                    messages = response.json()
                                    
                    for msg in messages:
                        if msg["sender"] == "user":
                            memory.chat_memory.add_user_message(msg["content"])
                        elif msg["sender"] == "assistant":
                            memory.chat_memory.add_ai_message(msg["content"])
                            
                    print(f"Cargados {len(messages)} mensajes del thread {last_thread_id}")
                else:
                    # Crear nuevo thread
                    response = await client.post(f"{conf.USERS_API_FULL_URL}/threads/create/{app_user.identifier}")
                    response.raise_for_status()
                    thread_data = response.json()
                    cl.user_session.set("thread_id", thread_data["thread_id"])
                    print(f"Nuevo thread creado en resume: {thread_data['thread_id']}")
                
        except Exception as e:
            print(f"Error: {e}")
            root_messages = [m for m in thread["steps"] if m["parentId"] is None]
            for message in root_messages:
                if message["type"] == "user_message":
                    memory.chat_memory.add_user_message(message["output"])
                else:
                    memory.chat_memory.add_ai_message(message["output"])
    
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
    user = cl.user_session.get("user")
    thread_id = cl.user_session.get("thread_id")
    
    if user and thread_id:
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{conf.USERS_API_FULL_URL}/threads/message/{thread_id}",
                    params={"sender": "user"},
                    json={"message": message.content}
                )
            except Exception as e:
                print(f"Error saving user message: {e}")

    if (
        message.elements and user.metadata["role"] == Role.CLIENTE
    ):  # Esto requiere modificarse por Role.CLIENTE para utilisar la funcion de subir pdfs...
        file = message.elements[0]
        settings = cl.user_session.get("settings")
        # msg = cl.Message(content=f"Procesando archivo `{file.name}`...")
        # await msg.send()
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
        
            # Guardar en thread
            if user and thread_id:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(
                            f"{conf.USERS_API_FULL_URL}/threads/message/{thread_id}",
                            params={"sender": "assistant"},
                            json={"message": success_message}
                        )
                    except Exception as e:
                        print(f"Error saving success message: {e}")
            
            # NO hacer return aquí si hay contenido de texto para procesar
            if not message.content or not message.content.strip():
                return
                    
        except Exception as e:
            error_msg = f"Error procesando el archivo `{file.name}`: {str(e)}"
            print(error_msg)

            # Guardar mensaje de error en thread
            if user and thread_id:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(
                            f"{conf.USERS_API_FULL_URL}/threads/message/{thread_id}",
                            params={"sender": "assistant"},
                            json={"message": error_msg}
                        )
                    except Exception as e:
                        print(f"Error saving error message: {e}")
            # NO hacer return aquí si hay contenido de texto para procesar
            if not message.content or not message.content.strip():
                return

    # Procesar contenido de texto (si existe y no es solo archivo)        
    if message.content and message.content.strip():
        msg = cl.Message(content="")  # Solo muestra el loader si no se envió otro mensaje
    await msg.send()

    query = message.content
    context = await vectordb_results_step(query)
    settings = cl.user_session.get("settings")
    kwargs = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "frequency_penalty": settings["frequency_penalty"],
    }
    respuesta = await llm_step(query=query, context=context, **kwargs)
    msg.content = f"{respuesta}"

    await msg.update()  
    
    # Guardamos respuesta del asistente
    if user and thread_id and 'respuesta' in locals():
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{conf.USERS_API_FULL_URL}/threads/message/{thread_id}",
                    params={"sender": "assistant"},
                    json={"message": respuesta}
                )
            except Exception as e:
                print(f"Error: {e}")

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
