import fastapi
from llama_cpp import Llama

app = fastapi.FastAPI()

# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF
llm = Llama(
    model_path="./mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf",  # Download the model file first
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=-1,  # The number of layers to offload to GPU, if GPU available
)

# Simple inference example
prompt_template = """<|im_start|>system
sos un chatbot. contexto: {context}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
config = {
    "repeat_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 10,
    "top_p": 0.95,
    "echo": False,
    "max_tokens": 256,
}


def get_response(prompt, context):
    output = llm(
        prompt_template.format(context=context, prompt=prompt),
        stop=["<|im_end|>"],
        **config,
    )
    return output["choices"][0]["text"]


@app.post("/submit-prompt")
def generate_answer(prompt: str, context: str = None):
    return get_response(prompt, context)
