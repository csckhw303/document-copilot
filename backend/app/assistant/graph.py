"""LangGraph document agent graph."""

from __future__ import annotations

from pathlib import Path

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.assistant.outputs import GroundedAnswer
from app.assistant.state import AgentState, registry_from_state
from app.assistant.tools import AGENT_TOOLS
from app.config import settings
from app.grounding.validator import GroundingValidator, prune_unreferenced_citations
log = structlog.get_logger()

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

graph = _builder.compile()


def make_initial_state(query: str) -> dict:
    """Build the initial AgentState dict for a new turn."""
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
