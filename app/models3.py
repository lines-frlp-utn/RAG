from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GPTQ
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, device_map="cuda", trust_remote_code=False, revision="main"
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "cuales son las leyes de la fisica?"
system_message = "Sos un chatbot virtual"
prompt_template = """<|im_start|>sistema sos un chatbot virtual. contexto:
{context}<|im_end|>
<|im_start|>usuario
{prompt}<|im_end|>
<|im_start|>asistente
"""

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    return_full_text=False,
)
