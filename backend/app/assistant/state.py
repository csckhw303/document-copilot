"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.assistant.deps import TurnRegistry
from app.assistant.outputs import GroundedAnswer
from app.retrieval.types import RetrievedPassage


def _merge_passages(
    left: list[RetrievedPassage], right: list[RetrievedPassage]
) -> list[RetrievedPassage]:
    """Reducer: append new passages, deduped by chunk_id.

    To reset the accumulated list at the start of a turn, send
    ``Overwrite(value=[])`` (from ``langgraph.types``) instead of a plain list —
    that bypasses this reducer entirely rather than routing through it.
    """
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
