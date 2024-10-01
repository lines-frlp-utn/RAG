from app.aim_tracker import track_param, track_text
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
)


def get_conversational_answer(query, db_context, chat_history, aim_run, **kwargs):

    track_param(aim_run, "llm_config", {
        "model": "llama3.1",
        "base_url": f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
        "temperature": 0,
        "api_key": "none",
    })
    # Formatear el contexto de la base de datos
    context_section = f"El usuario ha hecho la siguiente pregunta: \"{query}\".\n"

    track_text(aim_run, "context_section", context_section)
    
    # Formatear el contexto recuperado
    db_context_section = f"Además, para responder la consulta podes ayudarte con la informacion de contexto y el historial de conversacion:\n{db_context}\n"

    track_text(aim_run, "db_context_section", db_context_section)

    # Instrucción de respuesta
    instruction_section = (
        "Independientemente del idioma de la pregunta, responde siempre en español. "
    )

    track_text(aim_run, "instruction_section", instruction_section)

    chat_history_section = "Este es el historial de conversacion con el usuario que puede ser relevante para entender mejor su pregunta actual:\n"

    track_text(aim_run, "chat_history_section", chat_history_section)

    for message in chat_history:
        role = "Usuario" if message["role"] == "user" else "Asistente"
        chat_history_section += f"{role}: {message['content']}\n"
    # Construir el prompt completo
    full_prompt = f"{context_section}{instruction_section}{db_context_section}{chat_history_section}"
    
    # Imprimir el prompt para depuración
    print(f"Full prompt: {full_prompt}")

    track_text(aim_run, "full_prompt", full_prompt)
    
    # Generar la respuesta usando el LLM
    answer = llm.invoke(full_prompt, **kwargs)
   
    track_text(aim_run, "answer", answer)
    
    return answer.content
