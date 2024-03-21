from langchain.chains import RetrievalQAWithSourcesChain
import chainlit as cl
from loggerQA import logger
from pathlib import Path
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from constants import *
from langchain.llms import CTransformers
import os
from langchain.vectorstores import Chroma
from chromadb.errors import InvalidDimensionException
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter,CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.schema import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
)
from tempfile import NamedTemporaryFile
#for upload a file
from chainlit.types import AskFileResponse
from langchain.docstore.document import Document
#for chainlit playground configuration
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from chainlit.input_widget import Select, Slider


#Para cuestiones de config, nombre,habilitar caracteristicas, como playground
#modificarlo en /chainlit/config.toml

load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_ENDPOINT=os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_PROJECT=os.getenv('LANGCHAIN_PROJECT')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY')
UPLOAD_PATH = os.getenv('UPLOAD_PATH')
PDF_STORAGE_PATH = os.getenv('PDF_STORAGE_PATH')

chunk_size = 1024
chunk_overlap = 50


def create_sbert_minilm():
    """This model maps sentences & paragraphs to a 384 dimensional dense vector space 
        and can be used for tasks like clustering or semantic search.
    """
    logger.info(f"creating model embedding: {EMB_SBERT_MINILM}")
    #self.embedding=SentenceTransformerEmbeddings(model_name=EMB_SBERT_MINILM , model_kwargs={"device": self._set_device()})
    return SentenceTransformerEmbeddings(model_name=EMB_SBERT_MINILM)

def create_mistral7b( load_in_8bit=False):
    logger.info(f"creating llm model: {LLM_MISTRAL7B}")
    #TODO probar de bajar primero el modelo y levantarlo de un path, xq sino usa conexión....
    config = {'context_length':1700 ,'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.0, 'top_k': 10, 'context_length':2300}
    #config = {'context_length':1000 ,'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.0, 'top_k': 10, 'context_length':2030}
    return CTransformers(model=LLM_MISTRAL7B, model_file="mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf", model_type="llama", config=config) 

def vector_db_pdf():
    """
    creates vector db for the embeddings and persists them or loads a vector db from the persist directory
    """
    logger.info(f"creating Chroma vector db")
    pdf_directory = Path(PDF_STORAGE_PATH)
    logger.info(f"file_path:{pdf_directory} y: {os.path.exists(pdf_directory)}")    
    logger.info(f"PERSIST_DIRECTORY:{PERSIST_DIRECTORY}")     
    #file_path=os.path.join(UPLOAD_PATH, filename)
    if PERSIST_DIRECTORY and os.path.exists(PERSIST_DIRECTORY):
        ## Load from the persist db
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
        return vectordb
    elif pdf_directory: # and os.path.exists(file_path):
        docs = []  # type: List[Document]
        texts=""
        ## 1. Extract the documents
        logger.info(f"antes:{texts}")
        for pdf_path in pdf_directory.glob("*.pdf"):
            logger.info(f"en:{texts}")
            loader = PDFPlumberLoader(str(pdf_path))
            documents = loader.load()
            logger.info(f"load:{texts}")
            #loader = PyMuPDFLoader(str(pdf_path))
            #documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)            
            texts = text_splitter.split_documents(documents)
            ## 2. Split the texts       
            text_splitter = TokenTextSplitter(chunk_size=1500, chunk_overlap=10)  # This the encoding for text-embedding-ada-002
            texts = text_splitter.split_documents(texts)
            logger.info(f"texts:{texts}")
            #docs +=texts            

        ## 3. Create Embeddings and add to chroma store
        try:
            vectordb = Chroma.from_documents(documents=texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY)
            return vectordb
        except InvalidDimensionException:
            vectordb.delete_collection()
            vectordb = Chroma.from_documents(documents=texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY)               
            return vectordb
    else:
        raise ValueError("NO PDF found")   
          

embedding_model = create_sbert_minilm()  
# Instantiate the LLM
llm=create_mistral7b() 
# Add the LLM provider
#https://docs.chainlit.io/advanced-features/prompt-playground/llm-providers
add_llm_provider(
    LangchainGenericProvider(
        # It is important that the id of the provider matches the _llm_type
        id=llm._llm_type,
        # The name is not important. It will be displayed in the UI.
        name=LLM_MISTRAL7B,
        # This should always be a Langchain llm instance (correctly configured)
        llm=llm,
        inputs=[
            Select(
                id=LLM_MISTRAL7B,
                label="Model",
                values=[LLM_MISTRAL7B],
                initial_value=LLM_MISTRAL7B,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                min=0.0,
                max=1.0,
                step=0.01,
                initial=0.0,
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                min=0,
                max=2000,
                step=1,
                initial=256,
            )            
        ],        
        # If the LLM works with messages, set this to True
        is_chat=False
    )
)
#vectordb = vector_db_pdf()    

def process_file(file: AskFileResponse):
    logger.info(f"on process_file")
    """if file.type == "text/plain":
        Loader = TextLoader
    elif """
    if file.type == "application/pdf":
        #Loader = PyPDFLoader
        #loader = Loader(file.path)
        loader = PDFPlumberLoader(file.path)
        documents = loader.load()


        """docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs"""
        logger.info(f"end process_file")        
        return documents

@cl.on_chat_start
async def on_chat_start():
    logger.info(f"on_chat_start")
    welcome_message = cl.Message(content="Inicializando el bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hola, Bienvenidos al bot de QA sobre PDF usando Langchain y Mistral."
    )
    await welcome_message.update()

    #upload de pdf file
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Por favor suba un archivo PDF para continuar!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]    
    msg = cl.Message(content=f"Procesando archivo `{file.name}`...", disable_feedback=True)
    await msg.send()    

    documents = process_file(file=file)
    logger.info(f"start splitt")
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)            
    texts = text_splitter.split_documents(documents)
    ## 2. Split the texts       
    text_splitter = TokenTextSplitter(chunk_size=1500, chunk_overlap=100)  # This the encoding for text-embedding-ada-002
    texts = text_splitter.split_documents(texts)     
    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]          
    #cl.user_sessionprocess_file.set("metadatas", metadatas) 
    logger.info(f"end splitt")

    #cl.user_sessionprocess_file.set("docs", texts)   
    #logger.info(f"end session set docs")



    
    ## 3. Create Embeddings and add to chroma store
    logger.info(f"start vectordb")    
        ## 3. Create Embeddings and add to chroma store
    try:
        vectordb = await cl.make_async(Chroma.from_documents)(documents=texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY)
    except InvalidDimensionException:
        vectordb.delete_collection()
        vectordb = await cl.make_async(Chroma.from_documents)(documents=texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY)               
    
    logger.info(f"end vectordb")  
    logger.info(f"prompt_template")
    prompt_template = """Conteste la siguiente pregunta basandose solamente en el contexto provisto:
    Si no sabes la respuesta, sólo di que no sabes, no trates de crearla. Siempre di "gracias por preguntar!".
    Si la pregunta está fuera de contexto, amablemente informa que no fuiste entrenado para esa pregunta.
    Si el contexto no es relevante para contestar la pregunta, por favor no contestes la pregunta usando tu propio conocimiento.

    <contexto>
    {context}
    </contexto>

    Pregunta: {question}"""
    PROMPT = ChatPromptTemplate.from_template(prompt_template)
  
    """PROMPT = ChatPromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )"""    

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = vectordb.as_retriever(search_kwargs={"k":2})
    logger.info(f"armo retriever")
    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    logger.info(f"armo runnable")
    cl.user_session.set("runnable", runnable)    

    msg = cl.Message(content=f"Subida archivo `{file.name}` exitosa", disable_feedback=True)
    await msg.send()     

@cl.on_message
async def on_message(message: cl.Message):
    logger.info(f"MSG Received: {message.content}")
    runnable = cl.user_session.get("runnable")  # type: Runnable
    #msg = cl.Message(content="")
    msg=message
    logger.info(f"msg:{msg.content}")

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
        logger.info(f"MSG Received step: {message.content}")
        msg.content = f"Procesando la pregunta.Por favor espere!"
        await msg.send()
        
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(msg)
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()    
    
 
