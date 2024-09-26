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


def get_conversational_answer(db_context, chat_history, **kwargs):
    # Formatear el contexto de la base de datos
    context_section = f"Added Context: {db_context}\n"
    
    # Formatear el historial de la conversación
    # Puedes elegir el formato que mejor se ajuste a tu caso, aquí usamos un formato tipo diálogo
    chat_history_section = "Chat History:\n"
    for message in chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        chat_history_section += f"{role}: {message['content']}\n"
    
    # Construir el prompt completo
    full_prompt = f"{context_section}{chat_history_section}User Query: "
    
    # Imprimir el prompt para depuración
    print(f"Full prompt: {full_prompt}")
    answer = llm.invoke(full_prompt, **kwargs)
    aim_callback.flush_tracker(langchain_asset=llm)
    return answer.content
