from langchain_community.vectorstores import Chroma

from app.models import embedding_model

vector_db = Chroma(persist_directory="./database/", embedding_function=embedding_model)
