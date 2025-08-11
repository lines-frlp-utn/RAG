from typing import Optional, TypedDict

from app.graders import answer_grader, hallucination_grader, question_rewriter
from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]
    grounded: Optional[bool]
    settings: Optional[dict]


# Node: Retrieve context
async def retrieve(state: GraphState):
    from app.main import vectordb_results_step

    context = await vectordb_results_step(state["question"])
    return {**state, "context": context}


# Node: Generate answer
async def generate(state: GraphState):
    from app.main import llm_step

    settings = state.get("settings", {})
    answer = await llm_step(query=state["question"], context=state["context"], **settings)
    return {**state, "answer": answer}


# async def grade_context(state: GraphState):
# print("---CHECK CONTEXT RELEVANCE TO QUESTION---")
# question = state["question"]
# context = state["context"]


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    context = state["context"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"context": context, "question": better_question}


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    context = state["context"]
    answer = state["answer"]

    score = hallucination_grader.invoke({"context": context, "answer": answer})
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "answer": answer})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
# workflow.add_node("grade_context", grade_context)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)


workflow.add_edge(START, "retrieve")

# workflow.add_edge("retrieve", "grade_context")

# workflow.add_conditional_edges(
#     "grade_context",
#     decide_to_generate,
#     {
#         "transform_query": "transform_query",
#         "generate": "generate",
#     },
# )

workflow.add_edge("retrieve", "generate")

# workflow.add_edge("transform_query", "retrieve")


def handle_not_supported(state):
    return {
        **state,
        "answer": "Error: La generación no está fundamentada en los documentos. Por favor, revise la pregunta o el contexto.",
    }


def handle_not_useful(state):
    return {
        **state,
        "answer": "Error: La respuesta generada no responde adecuadamente a la pregunta. Intente reformular la pregunta.",
    }


workflow.add_node("not_supported_error", handle_not_supported)
workflow.add_node("not_useful_error", handle_not_useful)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "not_supported_error",
        "useful": END,
        "not useful": "not_useful_error",
    },
)

# Compile
app = workflow.compile()
