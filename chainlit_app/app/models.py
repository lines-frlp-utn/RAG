from app.aim_tracker import track_param, track_text
from app.config import conf
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama3.1",
    base_url=f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
    temperature=0,
    api_key="none",
)


def get_conversational_answer(query, db_context, chat_history, aim_run, **kwargs):
    track_param(
        aim_run,
        "llm_config",
        {
            "model": "llama3.1",
            "base_url": f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
            "temperature": 0,
            "api_key": "none",
        },
    )
    # Formatear el contexto de la base de datos
    system_prompt = f"""Eres un asistente llamado lines-bot. Siempre vas a responder en español.
    El usuario no sabe que se te proporciona un contexto, no lo menciones.
    Para responder la consulta podes ayudarte con la informacion de contexto y el historial de conversacion:
    {db_context}
    """

    # Reemplazar el system prompt al inicio de la conversación con context actualizado
    chat_history = [msg for msg in chat_history if msg["role"] != "system"]
    chat_history.insert(0, {"role": "system", "content": system_prompt})
    # Eliminar el último elemento del historial de chat, mensaje de asistente vacio
    if chat_history[-1]["role"] == "assistant":
        chat_history.pop()
    # Imprimir el prompt para depuración
    print(f"system prompt: {system_prompt}")
    track_text(aim_run, "system_prompt", system_prompt)
    track_text(aim_run, "db_context_section", db_context)
    track_text(aim_run, "user_prompt", query)

    # Generar la respuesta usando el LLM
    answer = llm.invoke(chat_history, **kwargs)
    track_text(aim_run, "answer", answer.content)
    return answer.content
