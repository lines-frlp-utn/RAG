from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import CTransformers

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""
EMB_SBERT_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MISTRAL7B = "TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF"
# C:\Users\usuario\.cache\huggingface\hub
LLM_LLAMA_BLOKE = "TheBloke/Llama-2-13B-Ensemble-v5-GGUF"

embedding_model = SentenceTransformerEmbeddings(model_name=EMB_SBERT_MINILM)

"""Creates the LLMS model that will generate contextualized embeddings"""
config = {
    "context_length": 1700,
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": 0.0,
    "top_k": 10,
}
llm = CTransformers(
    model=LLM_MISTRAL7B,
    model_file="mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf",
    model_type="llama",
    config=config,
)

prompt_template = """Conteste la siguiente pregunta basandose solamente en el contexto provisto:
Si no sabes la respuesta, sólo di que no sabes, no trates de crearla. Siempre di "gracias por preguntar!".
Si la pregunta está fuera de contexto, amablemente informa que no fuiste entrenado para esa pregunta.
Si el contexto no es relevante para contestar la pregunta, por favor no contestes la pregunta usando tu propio conocimiento.

<contexto>
{context}
</contexto>

Pregunta: {question}"""
