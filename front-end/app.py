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
########################################################################

def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


#autenticacion de usuario (proximamente podriamos usarlo con el servidor local del lines)
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    return cl.User(identifier="test")
#     #identificacion de usuario
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None


#decorator que sirve para definir lo que va a suceder apenas arranque el chat
@cl.on_chat_start
async def start():
    # logger.info(f"Chat started!")

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
                vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
            except InvalidDimensionException:
                # logger.info(f"invalid dimension exception")
                vectorstore.delete_collection()
                vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

            #generar retriever
            # logger.info("retriever")
            retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":2})

            #cargar prompt
            # logger.info(f"prompt template")
            prompt_template = """Conteste la siguiente pregunta basandose solamente en el contexto provisto:
            Si no sabes la respuesta, sólo di que no sabes, no trates de crearla. Siempre di "gracias por preguntar!".
            Si la pregunta está fuera de contexto, amablemente informa que no fuiste entrenado para esa pregunta.
            Si el contexto no es relevante para contestar la pregunta, por favor no contestes la pregunta usando tu propio conocimiento.

            <contexto>
            {context}
            </contexto>

            Pregunta: {question}"""
            prompt = ChatPromptTemplate.from_template(prompt_template)

            #runable
            # logger.info(f"runnable")
            runnable = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            # logger.info(f"set runnable")
            cl.user_session.set("runnable", runnable) #user_sesion.set nos permite guardar informacion a traves del ciclo de vida del chat

            ##### backend

            question = await cl.AskUserMessage(
                content=f"archivo '{text_file.name}' type: '{text_file.type}', size: {text_file.size}, N° chunks: {len(chunks)}, subido correctamente \n ya puedes hacer tu pregunta",
                timeout=500,
            ).send()

            retrieved_docs = retriever.invoke(question['output'])

        else:
            with open(text_file.path, "r", encoding="utf-8") as f:
                text = f.read()

        #Mostramos un mensaje donde simplemente decimos el nombre del archivo y la longitud de caractaeres
        await cl.Message(
            content=f"pregunta: {question['output']}, cantidad de chunks encontrados: {len(retrieved_docs)}\nchunk1: {retrieved_docs[0].page_content} \nchunk2: {retrieved_docs[1].page_content}"
        ).send()
    #setup_runnable()

@cl.step
async def tool():
    # Simulate a running task
    await cl.sleep(2)

    await child_step() #llamo a una tarea hija
    return 'Response 1'
@cl.step
async def child_step():
    current_step = cl.context.current_step

    # Override the input of the step
    current_step.input = "My custom input"

    # Override the output of the step
    current_step.output = "Response 2"

#decorator que define lo que sucede cuando el usuario envia un mensaje
@cl.on_message
async def main(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")
    msg=message
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

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            logger.info(f"MSG received _init_: {message.content}")
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                logger.info(f"MSG Received on_llm_end: {self.msg.content}")
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async with cl.Step(type="run", name="Asistente PDF  QA"):
        msg.content = f"Procesando la pregunta.Por favor espere!"
        await msg.send()

        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(message)
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()
    msg.actions = actions #cargamos las acciones del mensaje (boton de accion de demostracion)

    await msg.update() #actualizamos el mensaje con los nuevos datos

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)

#el return devuelve un mensaje flotante en la pantalla
@cl.action_callback("action_button")
async def on_action(action: cl.Action):
    print("The user clicked on the action button!")

    return "Thank you for clicking on the action button!"

#cuando el usuario clickea el boton para detener la tarea que se estaba ejecutando
#hay que ver si se puede mostrar un mensaje flotante en la pantalla
@cl.on_stop
async def on_stop():
    print("The user wants to stop the task!")

#Cuando termina la sesion del usuario
#hay que ver si se puede mostrar un mensaje flotante en la pantalla o si la IA puede mandar un saludo como ultimo mensaje
@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")
    return "thanks"



# esto va con el ThreadDict -> proximamente aprenderemos a usarlo, necesitamos autenticacion
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable()
