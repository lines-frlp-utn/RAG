import requests
from langchain_community.embeddings import SentenceTransformerEmbeddings

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""

EMB_MULTI_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = SentenceTransformerEmbeddings(model_name=EMB_MULTI_MINILM)


def get_conversational_answer(prompt, context):
    answer = requests.post(
        f"http://localhost:8007/submit-prompt?prompt={prompt}&context={context}"
    )
    return answer.json()
