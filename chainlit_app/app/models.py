from app.aim_tracker import track_param, track_text
from app.config import conf
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama3.1",
    base_url=f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
    temperature=0,
    api_key="none",
)

def get_conversational_answer(query, db_context, chat_history, aim_run, safe_context=None, **kwargs):
    # Usar safe_context si está disponible, de lo contrario formatear db_context
    tracking_context = safe_context if safe_context is not None else format_db_context(db_context)
    
    track_param(
        aim_run,
        "llm_config",
        {
            "model": kwargs.get("model", "llama3.1"),
            "base_url": f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
            "temperature": kwargs.get("temperature", 0),
            "api_key": "none",
        },
    )
    
    system_prompt = f"""Eres un asistente llamado lines-bot. Siempre vas a responder en español.
    El usuario no sabe que se te proporciona un contexto, no lo menciones.
    Para responder la consulta podes ayudarte con la informacion de contexto:
    {format_db_context(db_context)}
    """
    
    track_text(aim_run, "system_prompt", system_prompt)
    track_text(aim_run, "db_context_section", tracking_context)  # Usar el contexto seguro para tracking
    track_text(aim_run, "user_prompt", query)

    chat_history.insert(0, {"role": "system", "content": system_prompt})
    print(f"Prompt completo: {chat_history}")

    answer = llm.invoke(chat_history, **kwargs)
    track_text(aim_run, "answer", answer.content)
    return answer.content


def format_db_context(db_context):
    """Formatea el contexto de la base de datos para asegurar que sea un string válido."""
    if isinstance(db_context, str):
        return db_context
    elif isinstance(db_context, list):
        context_lines = []
        for item in db_context:
            if hasattr(item, 'text'):
                context_lines.append(str(item.text))
            elif isinstance(item, dict) and 'text' in item:
                context_lines.append(str(item['text']))
            elif isinstance(item, str):
                context_lines.append(item)
        return "\n\n".join(context_lines) if context_lines else "No hay contexto disponible"
    elif db_context is None:
        return "No hay contexto disponible"
    else:
        return str(db_context)