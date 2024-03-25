# from typing import Optional
import chainlit as cl
from chainlit.types import ThreadDict


from operator import itemgetter
from loggerQA import logger

#para memoria necesito langchain
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain.memory import ConversationBufferMemory


#BACKEND
########################################################################
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings
from chromadb.errors import InvalidDimensionException
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

import sys
sys.path.append("..")
from umap_embeddings.auto_umap import make_umap

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

EMB_SBERT_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MISTRAL7B="TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF"
#C:\Users\usuario\.cache\huggingface\hub
LLM_LLAMA_BLOKE = "TheBloke/Llama-2-13B-Ensemble-v5-GGUF"

def create_embeding():
    """This model maps sentences & paragraphs to a 384 dimensional dense vector space 
        and can be used for tasks like clustering or semantic search.
    """
    # logger.info(f"creating model embedding: {EMB_SBERT_MINILM}")
    return SentenceTransformerEmbeddings(model_name=EMB_SBERT_MINILM)

def create_llm_model(load_in_8bit=False):
    """Creates the LLMS model that will generate contextualized embeddings"""
    # logger.info(f"creating llm model: {LLM_MISTRAL7B}")
    config = {'context_length':1700 ,'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.0, 'top_k': 10, 'context_length':2300}
    return CTransformers(model= LLM_MISTRAL7B, model_file="mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf", model_type="llama", config=config)

embedding_model = create_embeding()
llm=create_llm_model()

prompt_template = """Conteste la siguiente pregunta basandose solamente en el contexto provisto:
Si no sabes la respuesta, sólo di que no sabes, no trates de crearla. Siempre di "gracias por preguntar!".
Si la pregunta está fuera de contexto, amablemente informa que no fuiste entrenado para esa pregunta.
Si el contexto no es relevante para contestar la pregunta, por favor no contestes la pregunta usando tu propio conocimiento.

<contexto>
{context}
</contexto>

Pregunta: {question}"""
########################################################################

def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])


#decorator que sirve para definir lo que va a suceder apenas arranque el chat
@cl.on_chat_start
async def start():
    # logger.info(f"Chat started!")
    cl.user_session.set("session_number", 0)

    #necesito el langchain para esto
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    files = None

    # Se envia dos botones de accion al comienzo del chat
    #tener encuenta que todos los "Ask" tienen un timeout en segundos, si no se realiza nada antes del timeout, tira un error de timeout.
    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="Chat", value="chat", label="✅ ChatBot"),
            cl.Action(name="Cargar PDF", value="pdf", label="🔥 Cargar PDF"),
        ],
    ).send()

    #Chequeando la opcion elegida
    if res and res.get("value") == "chat":
        #esta opcion aun no funciona, deberiamos tener una base de datos con archivos para que la inteligencia tenga un contexto con el cual responder
        #esperando la respuesta del usuario
        name = await cl.AskUserMessage(
            content="Bienvenido! ¿Cual es tu nombre?",
        ).send()
        if res:
            await cl.Message(
                content=f"Hola {name['output']}! ¿de que querias hablar hoy? ",
            ).send()
    if res and res.get("value") == "pdf":
        while files == None:
            #Esperando que el usuario cargue un archivo pdf
            files = await cl.AskFileMessage(
                content="Please upload a text file to begin!", 
                accept=["text/csv", "application/pdf"], 
                max_size_mb=20, 
                timeout=180
            ).send()
        #Si el usuario carga varios archivos, en esta ocasion se lee solo el primero
        text_file = files[0]
        if text_file.type == "application/pdf":
            ##### backend
            
            #Apertura del archivo
            # logger.info(f"cargando pdf")
            loader = PyPDFLoader(text_file.path)
            pages = loader.load()
            
            #splitter
            # logger.info(f"splitting pdf")
            chunks = text_splitter.split_documents(pages)
            
            #guardo los chunks en chroma
            # logger.info("storing in chroma")
            try:
                vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model)
            except InvalidDimensionException:
                # logger.info(f"invalid dimension exception")
                vector_db.delete_collection()
                vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model)

            #generar retriever
            # logger.info("retriever")

            
            retriever = vector_db.as_retriever(search_type="similarity",search_kwargs={"k":2})

            ##### backend

            question = await cl.AskUserMessage(
                content=f"archivo '{text_file.name}' type: '{text_file.type}', size: {text_file.size}, N° chunks: {len(chunks)}, subido correctamente \n ya puedes hacer tu pregunta",
                timeout=500,
            ).send()

            #results = vector_db.similarity_search(question['output'])

            retrieved_docs = retriever.invoke(question['output'])

            #prompt = prompt_template.format(context= results, question=question['output'])

            cl.user_session.set("retriever", retriever)
            cl.user_session.set("vector_db", vector_db)
            



        else:
            with open(text_file.path, "r", encoding="utf-8") as f:
                text = f.read()

        #Mostramos un mensaje donde simplemente decimos el nombre del archivo y la longitud de caractaeres
        await cl.Message(
            content=f"pregunta: {question['output']}, cantidad de chunks encontrados: {len(retrieved_docs)}\nchunk1: {retrieved_docs[0].page_content} \nchunk2: {retrieved_docs[1].page_content}"
        ).send()


#decorator que define lo que sucede cuando el usuario envia un mensaje
@cl.on_message
async def main(message: cl.Message):

    retriever = cl.user_session.get("retriever")
    vector_db = cl.user_session.get("vector_db")
    session_number = cl.user_session.get("session_number")
    
    
    actions = [
        cl.Action(name="action_button", value="example_value", description="Click me!")
    ]    

    msg = cl.Message(content="") #Muestra un loader mientras carga el mensaje
    await msg.send()

    await cl.sleep(2) #aca iria la logica del back-end (supongo)
    if message.elements:
        images = [file for file in message.elements if "image" in file.mime] #preguntando se se ingreso una imagen en el mensaje.
        with open(images[0].path, "r") as f:
            pass
        msg.elements = [cl.Image(path=images[0].path, name="image", display="inline")]

    query = message.content
    results = vector_db.similarity_search(query)


    if message.content.startswith("umap"):
        query_embedding = embedding_model.embed_query(query)
        umap_path = make_umap(vector_db, results, query_embedding, query, session_number)
        msg.elements = [cl.Image(path=umap_path, name="umap", display="inline")]

    msg.content = f"resultado: {results[0].page_content}, {results[1].page_content}"
    msg.actions = actions #cargamos las acciones del mensaje (boton de accion de demostracion)

    await msg.update() #actualizamos el mensaje con los nuevos datos


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)