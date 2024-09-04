import requests
from app.config import conf
from langchain_community.embeddings import SentenceTransformerEmbeddings

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""

EMB_MULTI_MINILM = "sentence-transformers/distiluse-base-multilingual-cased-v2"
model_bertMini = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformerEmbeddings(model_name=model_bertMini)


def get_conversational_answer(prompt, context):
    answer = requests.post(
        url=f"{conf.MODEL_URL}:{conf.MODEL_PORT}/submit-prompt?prompt={prompt}&context={context}", headers={"Content-Type": "application/json"}
    )
    return answer.json()
