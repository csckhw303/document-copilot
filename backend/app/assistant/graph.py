"""LangGraph document agent graph."""

from __future__ import annotations

import re
import time
from pathlib import Path

import structlog
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Overwrite

from app.assistant.outputs import GroundedAnswer
from app.assistant.state import AgentState, registry_from_state
from app.assistant.tools import AGENT_TOOLS, get_retriever, prefetch_search
from app.config import settings
from app.grounding.validator import GroundingValidator, prune_unreferenced_citations
from app.retrieval.types import format_passages_for_agent

log = structlog.get_logger()

# Simple keyword-based filter extraction for the known pilot corpus.
# Imprecise matches are acceptable — the agent will re-search with corrected filters if needed.
_TICKER_KEYWORDS: dict[str, str] = {
    "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
}
_YEAR_RE = re.compile(r"\bfy\s*(\d{4})\b|fiscal\s+(?:year\s+)?(\d{4})\b", re.IGNORECASE)


def _extract_filters(query: str) -> tuple[str | None, str | None, str | None]:
    lower = query.lower()
    ticker = next((sym for kw, sym in _TICKER_KEYWORDS.items() if kw in lower), None)
    form: str | None = None
    if "10-k" in lower or "annual" in lower:
        form = "10-K"
    elif "10-q" in lower or "quarterly" in lower:
        form = "10-Q"
    matches = _YEAR_RE.findall(query)
    fiscal_years = ",".join(m[0] or m[1] for m in matches) if matches else None
    return ticker, form, fiscal_years

MAX_VALIDATION_ATTEMPTS = 2

_INSTRUCTIONS = (Path(__file__).with_name("instructions.md")).read_text(encoding="utf-8")

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
        )
    return _llm


async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = _get_llm().bind_tools(AGENT_TOOLS)
    messages = list(state["messages"])
    if state.get("validation_attempts", 0) > 0:
        messages.append(SystemMessage(
            content=(
                "Your previous answer had citation issues. "
                "Re-examine retrieved passages carefully and only cite chunks you have explicitly read."
            )
        ))
    response = await llm.ainvoke(messages, config)
    return {"messages": [response]}


async def extract_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = _get_llm().with_structured_output(GroundedAnswer)
    grounded = await llm.ainvoke(state["messages"], config)
    return {"grounded_answer": grounded}


async def validate_node(state: AgentState) -> dict:
    attempts = state.get("validation_attempts", 0) + 1
    grounded = state.get("grounded_answer")
    if grounded is None:
        return {"validation_attempts": attempts, "validation_ok": False}

    registry = registry_from_state(state)
    grounded = prune_unreferenced_citations(grounded)
    validation = await GroundingValidator().validate(grounded, registry)
    log.info(
        "citation validation",
        attempt=attempts,
        ok=validation.ok,
        citations=len(grounded.citations),
    )
    return {
        "grounded_answer": grounded,
        "validation_attempts": attempts,
        "validation_ok": validation.ok,
    }


def _should_retry(state: AgentState) -> str:
    if not state.get("validation_ok", False) and state.get("validation_attempts", 0) < MAX_VALIDATION_ATTEMPTS:
        return "agent_node"
    return END


_builder = StateGraph(AgentState)
_builder.add_node("agent_node", agent_node)
_builder.add_node("tools_node", ToolNode(AGENT_TOOLS))
_builder.add_node("extract_node", extract_node)
_builder.add_node("validate_node", validate_node)

_builder.add_edge(START, "agent_node")
_builder.add_conditional_edges("agent_node", tools_condition, {
    "tools": "tools_node",
    END: "extract_node",
})
_builder.add_edge("tools_node", "agent_node")
_builder.add_edge("extract_node", "validate_node")
_builder.add_conditional_edges("validate_node", _should_retry)

# MemorySaver keeps per-thread state in-process (POC only — lost on restart, not
# shared across workers). Swap for a PostgresSaver for durable, multi-worker memory.
_checkpointer = MemorySaver()
graph = _builder.compile(checkpointer=_checkpointer)


def make_initial_state(query: str) -> dict:
    """Build the full initial AgentState for the first turn of a thread."""
    return {
        "messages": [
            SystemMessage(content=_INSTRUCTIONS),
            {"role": "user", "content": query},
        ],
        "grounded_answer": None,
        "registry_passages": [],
        "validation_attempts": 0,
        "validation_ok": False,
    }


def make_followup_input(query: str) -> dict:
    """Build the state update for a continuing turn on an existing (checkpointed) thread.

    Only the new user message is appended — prior history lives in the checkpoint. The
    per-turn fields are reset so grounding stays scoped to this question: Overwrite(value=[])
    clears the citation allowlist (bypassing the append reducer), and the validation
    counters restart from zero.
    """
    return {
        "messages": [{"role": "user", "content": query}],
        "grounded_answer": None,
        "registry_passages": Overwrite(value=[]),
        "validation_attempts": 0,
        "validation_ok": False,
    }
