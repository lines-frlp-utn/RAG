import fastapi
from transformers import AutoTokenizer, Qwen2ForCausalLM, pipeline

app = fastapi.FastAPI()

# C:\Users\usuario\.cache\huggingface\hub

"""Creates the LLMS model that will generate contextualized embeddings"""
# config = {
#     "context_length": 1700,
#     "max_new_tokens": 256,
#     "repetition_penalty": 1.1,
#     "temperature": 0.0,
#     "top_k": 10,
#     "stop": ["<|im_end|>"],
# }

modelpath = "aloobun/Reyna-Mini-1.8B-v0.2"

model = Qwen2ForCausalLM.from_pretrained(
    modelpath,
)

tokenizer = AutoTokenizer.from_pretrained(
    modelpath,
    trust_remote_code=True,
)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_template = """<|im_start|>system\n Conteste la siguiente pregunta basandose solamente en el contexto provisto,
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
        max_new_tokens=50,
        return_full_text=False,
        top_k=10,
    )
    return output[0]["generated_text"]


@app.post("/submit-prompt")
def generate_answer(prompt: str, context: str = None):
    return get_response(prompt, context)
