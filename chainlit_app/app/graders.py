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
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {context} 
    \n ------- \n
    Here is the answer: {answer}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["answer", "context"],
)

hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()


answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {answer} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
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
