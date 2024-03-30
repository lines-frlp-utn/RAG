from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

"""This model maps sentences & paragraphs to a 384 dimensional dense vector space
and can be used for tasks like clustering or semantic search.
"""

EMB_MULTI_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GPTQ
LLM_MISTRAL7B = "TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GPTQ"

embedding_model = SentenceTransformerEmbeddings(model_name=EMB_MULTI_MINILM)

"""Creates the LLMS model that will generate contextualized embeddings"""
config = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 10,
    "return_full_text": False,
    "do_sample": False,
    "top_p": 0.95,
}

# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(
    LLM_MISTRAL7B, device_map="cuda", trust_remote_code=False, revision="main"
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MISTRAL7B, use_fast=True)

prompt = "cuales son las leyes de la fisica?"

prompt_template = """<|im_start|>sistem sos un chatbot virtual en espanol. Conteste la siguiente pregunta basandose solamente en el contexto provisto:
Si no sabes la respuesta, sólo di que no sabes, no trates de crearla. contexto:
{context}<|im_end|>
<|im_start|>usuario
{prompt}<|im_end|>
<|im_start|>asistente
"""

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **config)
