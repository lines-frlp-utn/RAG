from typing import Optional, TypedDict

from app.graders import answer_grader, context_grader, hallucination_grader, question_rewriter
from langgraph.graph import END, START, StateGraph

MAX_RETRIES = 3  # Maximum number of retries for context retrieval


class GraphState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]
    grounded: Optional[bool]
    settings: Optional[dict]
    context_grade: Optional[str]
    retry_count: Optional[int]


# Node: Retrieve context from vector database
async def retrieve(state: GraphState):
    retries = state.get("retry_count", 0) + 1
    from app.main import vectordb_results_step

    # Ejemplo: agrego el número de intento a la consulta para que se modifique
    question_for_retrieval = f"{state['question']} (intent {retries})"

    print(f"---RETRIEVING CONTEXT (attempt {retries})---")
    context = await vectordb_results_step(question_for_retrieval)
    return {**state, "context": context, "retry_count": retries}


# Node: Generate answer from context
async def generate(state: GraphState):
    from app.main import llm_step

    settings = state.get("settings", {})
    # Usar el contexto solo si la calificación fue "yes"
    context_relevant = state.get("context_grade") == "yes"
    context = state["context"] if context_relevant else None

    if not context_relevant:
        # Devolver mensaje directo sin llamar a llm_step
        return {**state, "answer": "No hay contexto relevante para responder esta pregunta."}

    answer = await llm_step(query=state["question"], context=context, **settings)
    return {**state, "answer": answer}


# Node: Generate answer without context
async def generate_no_context(state: GraphState):
    from app.main import llm_step

    settings = state.get("settings", {})
    answer = await llm_step(query=state["question"], context=None, **settings)
    return {**state, "answer": answer}


# Node: Grade context
async def grade_context(state: GraphState):
    print("---CHECK CONTEXT RELEVANCE TO QUESTION---")
    question = state["question"]
    context = state["context"] or ""

    score = context_grader.invoke({"question": question, "context": context})
    grade = score["score"]
    print(f"Context relevance score: {grade}")

    if grade == "yes":
        print("---DECISION: CONTEXT IS RELEVANT TO QUESTION---")
    else:
        print("---DECISION: CONTEXT IS NOT RELEVANT TO QUESTION---")

    return {**state, "context_grade": grade}


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

    # Re-write question
    better_question = question_rewriter.invoke({"question": question, "answer": ""})
    return {**state, "question": better_question}


def decide_to_generate_from_context(state: GraphState):
    print("---ASSESS GRADED CONTEXT WITH RETRY LIMIT---")
    grade = state.get("context_grade")
    retries = state.get("retry_count", 0)

    if grade != "yes":
        if retries >= MAX_RETRIES:
            print(
                f"---MAX RETRIES ({MAX_RETRIES}) ALCANZADO, NO SE PUEDE OBTENER CONTEXTO RELEVANTE -> GENERATE WITHOUT CONTEXT---"
            )
            return "generate_no_context"
        else:
            print(
                f"---DECISION: CONTEXT NO RELEVANTE -> TRANSFORM QUERY (INTENTO {retries + 1})---"
            )
            return "transform_query"
    else:
        print("---DECISION: CONTEXT RELEVANTE -> GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    question = state["question"]
    context = state.get("context")
    answer = state["answer"]

    # Evaluar grounding solo si hay contexto
    if context:
        score = hallucination_grader.invoke({"context": context, "answer": answer})
        grounded = score["score"] == "yes"
    else:
        grounded = None  # No hay contexto

    if grounded is True:
        # Contexto presente y fundamentado
        score = answer_grader.invoke({"question": question, "answer": answer})
        return "useful" if score["score"] == "yes" else "not useful"

    elif grounded is False:
        # Contexto presente pero no fundamentado
        return "not supported"

    else:  # grounded is None → fallback sin contexto
        # evaluamos si responde la pregunta
        score = answer_grader.invoke({"question": question, "answer": answer})
        return "useful" if score["score"] == "yes" else "not useful"


# Error handlers
async def handle_not_supported(state: GraphState):
    # Add warning prefix to the existing answer
    warning = "**⚠️ ADVERTENCIA:** La respuesta generada puede no estar completamente fundamentada en los documentos disponibles.\n"

    existing_answer = state.get("answer")
    final_answer = warning + existing_answer

    return {
        **state,
        "answer": final_answer,
    }


async def handle_not_useful(state: GraphState):
    # Add warning prefix to the existing answer
    warning = (
        "**⚠️ ADVERTENCIA:** La respuesta inicial puede no abordar completamente su pregunta.\n"
    )

    existing_answer = state.get("answer") or ""
    final_answer = warning + existing_answer

    return {
        **state,
        "answer": final_answer,
    }


# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_context", grade_context)
workflow.add_node("generate", generate)
workflow.add_node("generate_no_context", generate_no_context)  # New node
workflow.add_node("transform_query", transform_query)
workflow.add_node("not_supported_error", handle_not_supported)
workflow.add_node("not_useful_error", handle_not_useful)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_context")

workflow.add_conditional_edges(
    "grade_context",
    decide_to_generate_from_context,
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "transform_query": "transform_query",
        "generate_no_context": "generate_no_context",  # New edge
        "not_supported_error": "not_supported_error",
    },
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "not_supported_error",
        "useful": END,
        "not useful": "not_useful_error",
    },
)

workflow.add_conditional_edges(
    "generate_no_context",
    grade_generation_v_documents_and_question,  # Same evaluation as generate
    {
        "not supported": "not_supported_error",
        "useful": END,
        "not useful": "not_useful_error",
    },
)

workflow.add_edge("not_supported_error", END)
workflow.add_edge("not_useful_error", END)

app = workflow.compile()
