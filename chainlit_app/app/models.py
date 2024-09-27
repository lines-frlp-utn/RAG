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
    context_section = f"El usuario ha hecho la siguiente pregunta: \"{query}\".\n"
    
    # Formatear el contexto recuperado
    db_context_section = f"Además, para responder la consulta podes ayudarte con la informacion de contexto y el historial de conversacion:\n{db_context}\n"

    # Instrucción de respuesta
    instruction_section = (
        "Independientemente del idioma de la pregunta, responde siempre en español. "
    )

    chat_history_section = "Este es el historial de conversacion con el usuario que puede ser relevante para entender mejor su pregunta actual:\n"

    for message in chat_history:
        role = "Usuario" if message["role"] == "user" else "Asistente"
        chat_history_section += f"{role}: {message['content']}\n"
    # Construir el prompt completo
    full_prompt = f"{context_section}{instruction_section}{db_context_section}{chat_history_section}"
    
    # Imprimir el prompt para depuración
    print(f"Full prompt: {full_prompt}")
    
    # Generar la respuesta usando el LLM
    answer = llm.invoke(full_prompt, **kwargs)
    aim_callback.flush_tracker(langchain_asset=llm)
    
    return answer.content
