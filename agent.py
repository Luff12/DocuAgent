import re
import requests

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from vector_store import retrieve_chunks

# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

ABSTENTION = (
    "I couldn't find sufficient information in the "
    "uploaded documents to answer this question reliably."
)


# ==================================================
# LANGGRAPH STATE
# ==================================================


class AgentState(TypedDict, total=False):

    question: str
    original_question: str
    intent: str
    document_ids: Optional[List[str]]
    chunks: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]


# ==================================================
# OLLAMA LLM
# ==================================================


def call_llm(
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 180,
) -> str:

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running.")

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama took too long to respond.")

    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Ollama request failed: {error}")


# ==================================================
# FOLLOW-UP QUESTION REWRITING
# ==================================================


def rewrite_follow_up_question(
    question: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:

    question = question.strip()

    if not chat_history:
        return question

    follow_up_starters = (
        "what about",
        "how about",
        "and what",
        "and in",
        "what was it",
        "what were they",
        "how much",
        "how many",
        "why is that",
        "explain that",
        "tell me more",
        "compare it",
        "compare them",
    )

    lower_question = question.lower()

    is_follow_up = lower_question.startswith(follow_up_starters) or lower_question in {
        "why?",
        "how?",
        "when?",
        "where?",
        "who?",
        "which one?",
        "and?",
    }

    if not is_follow_up:
        return question

    history_text = []

    for message in chat_history[-6:]:

        role = message.get("role", "unknown")
        content = message.get("content", "")

        if content:
            history_text.append(f"{role}: {content}")

    if not history_text:
        return question

    prompt = f"""
You rewrite follow-up questions using conversation history.

Rewrite the latest user question so it can be understood
independently.

RULES:

1. Preserve the user's original meaning.
2. Use the conversation to resolve references such as "it",
   "they", "that", or "those".
3. Do not answer the question.
4. Do not invent facts.
5. Return only the rewritten question.
6. If the question is already understandable, return it unchanged.

CONVERSATION:

{chr(10).join(history_text)}

LATEST QUESTION:

{question}

REWRITTEN QUESTION:
"""

    try:

        rewritten = call_llm(
            prompt,
            temperature=0,
        )

        rewritten = rewritten.strip().strip('"').strip("'")

        if not rewritten or len(rewritten) > 500:
            return question

        return rewritten

    except Exception:
        return question


# ==================================================
# INTENT CLASSIFICATION
# ==================================================


def classify_intent(state: AgentState) -> AgentState:

    question = state["question"]

    prompt = f"""
Classify the user's document-related question.

Choose exactly one category:

SEARCH
SUMMARIZE
COMPARE

SEARCH:
The user asks for a specific fact, name, number, date,
definition, or piece of information.

SUMMARIZE:
The user asks for a summary, overview, or key points.

COMPARE:
The user asks to compare two or more subjects, documents,
values, periods, or items.

Return only one category.

QUESTION:

{question}
"""

    try:

        result = call_llm(
            prompt,
            temperature=0,
        ).upper()

        if "COMPARE" in result:
            intent = "compare"

        elif "SUMMARIZE" in result:
            intent = "summarize"

        else:
            intent = "search"

    except Exception:
        intent = "search"

    state["intent"] = intent

    return state


# ==================================================
# DOCUMENT RETRIEVAL
# ==================================================


def search_documents(state: AgentState) -> AgentState:

    question = state["question"]
    document_ids = state.get("document_ids")

    try:

        chunks = retrieve_chunks(
            question=question,
            top_k=8,
            document_ids=document_ids,
            candidate_k=40,
        )

        state["chunks"] = chunks

    except Exception as error:

        state["chunks"] = []
        state["answer"] = f"Document retrieval failed: {error}"

    return state


# ==================================================
# SOURCE PREPARATION
# ==================================================


def prepare_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    sources = []

    for index, chunk in enumerate(chunks, start=1):

        sources.append(
            {
                "id": f"S{index}",
                "filename": chunk.get("filename", "Document"),
                "page": chunk.get("page", 0),
                "text": chunk.get("text", ""),
                "document_id": chunk.get("document_id", ""),
            }
        )

    return sources


# ==================================================
# CONTEXT PREPARATION
# ==================================================


def build_context(sources: List[Dict[str, Any]]) -> str:

    context_parts = []

    for source in sources:

        page = source["page"]

        page_label = f"Page {page}" if page else "Page not available"

        context_parts.append(f"""
SOURCE ID: [{source['id']}]

DOCUMENT: {source['filename']}

LOCATION: {page_label}

CONTENT:

{source['text']}
""")

    return "\n".join(context_parts)


# ==================================================
# CITATION UTILITIES
# ==================================================


def extract_citations(answer: str) -> set:
    """
    Extract source citations such as [S1] and [S2].
    """

    return set(re.findall(r"\[(S\d+)\]", answer))


def validate_citations(
    answer: str,
    sources: List[Dict[str, Any]],
) -> str:
    """
    Remove citations that do not exist in the current
    retrieved source list.

    If the model did not provide citations, attach a
    source reference section so the user can inspect
    the retrieved passages.
    """

    valid_ids = {source["id"] for source in sources}

    cited_ids = extract_citations(answer)

    invalid_ids = cited_ids - valid_ids

    # Remove citations that refer to nonexistent sources.
    for invalid_id in invalid_ids:

        answer = re.sub(
            rf"\[{re.escape(invalid_id)}\]",
            "",
            answer,
        )

    cited_ids = extract_citations(answer)

    if not cited_ids and sources:

        reference_ids = [f"[{source['id']}]" for source in sources[:3]]

        answer = answer.rstrip() + "\n\nSources: " + ", ".join(reference_ids)

    return answer.strip()


# ==================================================
# ABSTENTION DETECTION
# ==================================================


def is_abstention_response(answer: str) -> bool:

    text = answer.strip().lower()

    if not text:
        return True

    if text == ABSTENTION.lower():
        return True

    abstention_starters = (
        "i couldn't find sufficient information",
        "i could not find sufficient information",
        "i cannot answer this",
        "i can't answer this",
        "the provided context does not contain",
        "the documents do not contain",
        "the document does not mention",
        "not mentioned in the document",
        "not provided in the document",
        "insufficient information is available",
        "this cannot be determined from the context",
    )

    return text.startswith(abstention_starters)


# ==================================================
# ANSWER RECOVERY
# ==================================================


def recover_answer(
    question: str,
    context: str,
) -> str:
    """
    Retry answer generation when the model unnecessarily
    abstains despite having retrieved document passages.
    """

    recovery_prompt = f"""
You are DocuAgent, an AI document analysis assistant.

The previous answer attempt may have overlooked relevant
information in the supplied document passages.

Carefully review every passage before answering.

IMPORTANT RULES:

1. Search all passages for information relevant to the question.
2. Pay attention to names, headings, dates, numbers, tables,
   descriptions, and relationships.
3. Use only the supplied document passages.
4. Do not use outside knowledge.
5. Do not invent or assume facts.
6. If the requested information is present, answer directly.
7. Cite supporting information using the exact source IDs.
8. Use citations like [S1] or [S2].
9. If the information is genuinely absent, explain that it
   is not available in the supplied passages.

QUESTION:

{question}

DOCUMENT PASSAGES:

{context}

ANSWER:
"""

    try:

        return call_llm(
            recovery_prompt,
            temperature=0,
        ).strip()

    except Exception:
        return ""


# ==================================================
# ANSWER GENERATION
# ==================================================


def generate_answer(state: AgentState) -> AgentState:

    question = state["question"]
    chunks = state.get("chunks", [])

    # Retrieval failure or no matching passages.
    if not chunks:

        if state.get("answer"):
            return state

        state["answer"] = ABSTENTION
        state["sources"] = []

        return state

    # Prepare sources and document context.
    sources = prepare_sources(chunks)

    context = build_context(sources)

    intent = state.get("intent", "search")

    # Intent-specific instructions.
    if intent == "summarize":

        instruction = """
Provide a clear and organized summary of the relevant
document content.

Include important topics, facts, figures, dates,
and conclusions when available.

Do not add information that is not in the document.
"""

    elif intent == "compare":

        instruction = """
Compare the requested subjects using the supplied passages.

Clearly explain similarities and differences.

For numerical comparisons:
- Identify each value and its corresponding year or period.
- Calculate differences only when the required values exist.
- Do not assume missing values.
- Clearly distinguish values from different documents.
"""

    else:

        instruction = """
Answer the user's question directly.

If the question asks for a name, number, date, title,
definition, or other specific detail, provide it clearly
when it is available in the supplied passages.

Do not give an unnecessary general summary.
"""

    prompt = f"""
You are DocuAgent, an intelligent AI document analysis assistant.

Your task is to answer questions using the supplied document
passages.

You must carefully examine the complete supplied context
before deciding whether the answer is available.

IMPORTANT RULES:

1. Use ONLY the supplied document context.
2. Do not use outside knowledge.
3. Never invent names, dates, numbers, or facts.
4. Search all passages for information relevant to the question.
5. Information may appear in headings, tables, descriptions,
   lists, or ordinary paragraphs.
6. A short answer is acceptable when it answers the question.
7. Do not refuse when the requested information is explicitly
   available in the supplied passages.
8. Do not confuse information from different documents.
9. Pay attention to dates, years, units, and numerical values.
10. Cite supporting factual claims using the source IDs.
11. Use only the exact citation IDs provided in the context.
12. Citation format: [S1], [S2], etc.
13. If information is genuinely missing, say so clearly.
14. Do not claim that information is missing without checking
    all relevant passages.

TASK:

{instruction}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # First answer attempt.
    try:

        answer = call_llm(
            prompt,
            temperature=0.1,
        ).strip()

    except Exception as error:

        state["answer"] = f"Unable to generate an answer: {error}"

        state["sources"] = sources

        return state

    # Retry if the model unnecessarily abstains.
    if is_abstention_response(answer):

        recovered_answer = recover_answer(
            question=question,
            context=context,
        )

        if recovered_answer:
            answer = recovered_answer

    # Final abstention check.
    if is_abstention_response(answer):

        state["answer"] = ABSTENTION
        state["sources"] = sources

        return state

    # Validate citations without rejecting a useful answer
    # simply because the model omitted citations.
    answer = validate_citations(
        answer=answer,
        sources=sources,
    )

    state["answer"] = answer
    state["sources"] = sources

    return state


# ==================================================
# LANGGRAPH WORKFLOW
# ==================================================


def build_agent():

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "classify",
        classify_intent,
    )

    workflow.add_node(
        "retrieve",
        search_documents,
    )

    workflow.add_node(
        "generate",
        generate_answer,
    )

    workflow.set_entry_point("classify")

    workflow.add_edge(
        "classify",
        "retrieve",
    )

    workflow.add_edge(
        "retrieve",
        "generate",
    )

    workflow.add_edge(
        "generate",
        END,
    )

    return workflow.compile()


# ==================================================
# INITIALIZE AGENT
# ==================================================

agent = build_agent()


# ==================================================
# MAIN FUNCTION USED BY STREAMLIT
# ==================================================


def ask_document(
    question: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    document_ids: Optional[List[str]] = None,
):

    original_question = question.strip()

    if not original_question:

        return {
            "answer": "Please enter a question.",
            "sources": [],
            "intent": "search",
            "rewritten_question": "",
        }

    # Rewrite follow-up questions.
    rewritten_question = rewrite_follow_up_question(
        original_question,
        chat_history or [],
    )

    # Initial LangGraph state.
    initial_state: AgentState = {
        "question": rewritten_question,
        "original_question": original_question,
        "intent": "search",
        "document_ids": document_ids,
        "chunks": [],
        "answer": "",
        "sources": [],
    }

    try:

        result = agent.invoke(initial_state)

        return {
            "answer": result.get("answer", ABSTENTION),
            "sources": result.get("sources", []),
            "intent": result.get("intent", "search"),
            "rewritten_question": rewritten_question,
        }

    except Exception as error:

        return {
            "answer": (f"An error occurred while processing your question: {error}"),
            "sources": [],
            "intent": "search",
            "rewritten_question": rewritten_question,
        }
