import fastapi
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = fastapi.FastAPI()

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""

LLM_REYNA_MINI = "."
# C:\Users\usuario\.cache\huggingface\hub

"""Creates the LLMS model that will generate contextualized embeddings"""
config = {
    "context_length": 1700,
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": 0.0,
    "top_k": 10,
}

modelpath = "."

model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    modelpath,
    trust_remote_code=True,
    use_fast=False,
)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
