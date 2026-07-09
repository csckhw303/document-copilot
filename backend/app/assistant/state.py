"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.assistant.deps import TurnRegistry
from app.assistant.outputs import GroundedAnswer
from app.retrieval.types import RetrievedPassage


class _ResetPassages:
    """Sentinel update that clears accumulated registry passages.

    When a checkpointer persists state across turns, the citation allowlist must be
    reset at the start of each turn so passages retrieved for an earlier question do
    not stay citable. Passing RESET_PASSAGES to the ``registry_passages`` channel
    empties it before this turn's tool calls append to it.
    """


RESET_PASSAGES = _ResetPassages()


def _merge_passages(
    left: list[RetrievedPassage], right: list[RetrievedPassage] | _ResetPassages
) -> list[RetrievedPassage]:
    """Reducer: reset on RESET_PASSAGES, else append new passages (dedup by chunk_id)."""
    if right is RESET_PASSAGES:
        return []
    seen = {p.chunk_id for p in left}
    return left + [p for p in right if p.chunk_id not in seen]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    grounded_answer: GroundedAnswer | None
    registry_passages: Annotated[list[RetrievedPassage], _merge_passages]
    validation_attempts: int
    validation_ok: bool


def registry_from_state(state: AgentState) -> TurnRegistry:
    registry = TurnRegistry()
    registry.register_many(state["registry_passages"])
    return registry
