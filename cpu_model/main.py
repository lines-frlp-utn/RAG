import fastapi
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers

app = fastapi.FastAPI()

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""

LLM_REYNA_MINI = "aloobun/Reyna-Mini-1.8B-v0.2"
# C:\Users\usuario\.cache\huggingface\hub

"""Creates the LLMS model that will generate contextualized embeddings"""
config = {
    "context_length": 1700,
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": 0.0,
    "top_k": 10,
}
llm = CTransformers(
    model=LLM_REYNA_MINI,
    model_file="./Reyna-Mini-1.8B-v0.2",
    model_type="llama",
    config=config,
    gpu_layers=50,
)

prompt_template = """  <|im_start|>system\n Conteste la siguiente pregunta basandose solamente en el contexto provisto,
Si no sabes la respuesta, sólo di que no sabes, no trates de crearla. Siempre di "gracias por preguntar!".
Si la pregunta está fuera de contexto, amablemente informa que no fuiste entrenado para esa pregunta.
Si el contexto no es relevante para contestar la pregunta, por favor no contestes la pregunta usando tu propio conocimiento. contexto:
{context}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

def get_response(prompt, context):
    output = llm(
        prompt_template.format(context=context, prompt=prompt),
        stop=["<|im_end|>"],
    )
    return output


@app.post("/submit-prompt")
def generate_answer(prompt: str, context: str = None):
    return get_response(prompt, context)