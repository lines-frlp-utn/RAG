from app.aim_tracker import aim_callback, callbacks
from app.config import conf
from langchain_openai import ChatOpenAI

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""


llm = ChatOpenAI(
    model="llama3.1",
    base_url=f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
    temperature=0,
    api_key="none",
    callbacks=callbacks,
)


def get_conversational_answer(query, db_context, chat_history, **kwargs):
    # Formatear el contexto de la base de datos
    system_prompt = f"""Eres un asistente llamado lines-bot. Siempre vas a responder en español.
    El usuario no sabe que se te proporciona un contexto, no lo menciones.
    Para responder la consulta podes ayudarte con la informacion de contexto y el historial de conversacion:
    {db_context}
    """

    # Insertar el system prompt al inicio de la conversación
    chat_history.insert(0, {"role": "system", "content": system_prompt})

    # Imprimir el prompt para depuración
    print(f"system prompt: {system_prompt}")

    # Generar la respuesta usando el LLM
    answer = llm.invoke(chat_history, **kwargs)
    aim_callback.flush_tracker(langchain_asset=llm)

    return answer.content
