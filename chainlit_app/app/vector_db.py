from langchain_community.vectorstores import Chroma

from app.models import embedding_model

vector_db = Chroma(embedding_function=embedding_model) 
#le saque el persist_directory para que los archivos del "alumno" no se guarden
