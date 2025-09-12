from app.config import conf
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI

# LLM
llm = ChatOpenAI(
    model="llama3.1",
    base_url=f"{conf.MODEL_URL}:{conf.MODEL_PORT}/v1",
    temperature=0,
    api_key="none",
)


# Prompt
hallucination_grader_prompt = PromptTemplate(
    template="""You are a strict grader assessing whether an answer is fully grounded in and supported by the provided context. Follow these rules precisely:

- **Grounded (yes)**: The answer must be directly supported by explicit information in the context. It should not introduce new facts, assumptions, or details not present in the context. The answer must reference or paraphrase the context accurately without adding external knowledge.
- **Not Grounded (no)**: Respond with "no" if:
  - The answer contradicts any information in the context.
  - The answer includes facts or details not mentioned in the context.
  - The context is irrelevant to the answer (e.g., context about one topic but answer about another unrelated topic).
  - The answer relies on general knowledge, speculation, or assumptions not tied to the context.
- Only evaluate based on the provided context—ignore any external knowledge or common sense.
- If the context is empty, irrelevant, or does not address the answer's content, the answer is not grounded.

Here are the facts (context):
\n ------- \n
{context} 
\n ------- \n
Here is the answer: {answer}

Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by the context. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["answer", "context"],
)

hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

classification_prompt = PromptTemplate(
    template="""You are a classifier that determines if a question is a casual conversation or requires specific knowledge to answer. \n 
    Here is the question: {question}
    Classify the question as either 'conversation' or 'knowledge'. \n
    Examples:
    - 'Hola, ¿cómo estás?' is 'conversation'.
    - '¿Qué dice el documento sobre X?' is 'knowledge'.
    - 'Cuéntame un chiste' is 'conversation'.
    - 'Explica el teorema de Pitágoras' is 'knowledge'.
    - '¿Cuál es la capital de Francia?' is 'knowledge'.
    - '¿Qué opinas del clima hoy?' is 'conversation'.
    - '¿Quién ganó la Copa del Mundo en 2018?' is 'knowledge'.
    - '¿Quieres charlar un rato?' is 'conversation'.
    - '¿Qué es la fotosíntesis?' is 'knowledge'.
    - '¿Cómo estuvo tu día?' is 'conversation'.
    - Cualquier cosa relacionada con saludos, despedidas, charlas informales o preguntas personales es 'conversation'.
    - Cualquier cosa que busque información factual, explicaciones, datos o conocimiento específico es 'knowledge'.
    - cualquier cosa que sea una consulta sobre la UTN, la facultad, universidad, FRlP es 'knowledge'.
    Provide the classification as a JSON with a single key 'classification' and no preamble or explanation.""",
    input_variables=["question"],
)

classification_grader = classification_prompt | llm | JsonOutputParser()

# prompt
answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve or appropriately respond to a question. This includes factual questions requiring knowledge and casual or conversational questions.

- **Useful (yes)**: The answer directly addresses the question, provides relevant information, or engages appropriately in conversation. For casual greetings or chit-chat, a friendly response is considered useful.
- **Not Useful (no)**: The answer is off-topic, irrelevant, incomplete, or fails to engage with the question's intent (e.g., ignoring a greeting or providing incorrect information).

Here is the answer:
\n ------- \n
{answer} 
\n ------- \n
Here is the question: {question}

Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve or respond to the question. \n
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["answer", "question"],
)

answer_grader = answer_grader_prompt | llm | JsonOutputParser()

re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Give the improved question with no preamble and no explanation. 
     Mantain the original language of the question \n """,
    input_variables=["answer", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

context_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether a context is useful to resolve a question. \n 
    Here is the context:
    \n ------- \n
    {context} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the context is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["context", "question"],
)
context_grader = context_grader_prompt | llm | JsonOutputParser()
