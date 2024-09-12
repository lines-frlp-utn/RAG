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


def get_conversational_answer(prompt, context, **kwargs):
    # TODO use context
    prompt = f"Context: {context}. user: {prompt}"
    answer = llm.invoke(prompt, **kwargs)
    aim_callback.flush_tracker(langchain_asset=llm)
    return answer.content
