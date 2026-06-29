"""Coordinates one chat turn: graph → stream → persist."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

import structlog
from supabase import AsyncClient

from app.assistant.deps import TurnRegistry
from app.assistant.graph import graph, make_initial_state
from app.assistant.outputs import GroundedAnswer
from app.auth.dependencies import CurrentUser
from app.chat.messages import text_from_parts
from app.chat.streaming import (
    stream_error,
    stream_grounded_turn_and_persist,
    stream_status,
)
from app.grounding.validator import ValidationResult
from app.retrieval.types import RetrievedPassage
from app.schemas.chat import UIMessage

log = structlog.get_logger()


def _tool_status(tool_name: str) -> tuple[str, str]:
    if tool_name == "search_filings":
        return "searching", "Searching SEC filings…"
    if tool_name in {"read_chunk", "read_chunks", "read_surrounding_chunks"}:
        return "reading", "Reading source passages…"
    return "reading", "Reading source documents…"


async def run_turn(
    *,
    client: AsyncClient,
    thread_id: uuid.UUID,
    user: CurrentUser,
    user_message: UIMessage,
    thread_title: str,
) -> AsyncIterator[str]:
    query = text_from_parts(user_message.parts).strip()
    if not query:
        async for event in stream_error("User message is empty."):
            yield event
        return

    turn_log = log.bind(thread_id=str(thread_id), user_id=str(user.id))
    turn_log.info("turn start", query=query[:120])
    t0 = time.perf_counter()

    async for event in stream_status("analyzing", "Analyzing your question…"):
        yield event

    config = {
        "configurable": {
            "thread_id": str(thread_id),
            "user_id": str(user.id),
        }
    }

    grounded: GroundedAnswer | None = None
    validation_ok = False
    all_passages: list[RetrievedPassage] = []

    try:
        async for chunk in graph.astream(
            make_initial_state(query), config=config, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                if node_name == "agent_node":
                    messages = update.get("messages", [])
                    if messages:
                        tool_calls = getattr(messages[-1], "tool_calls", [])
                        if tool_calls:
                            stage, message = _tool_status(tool_calls[0]["name"])
                            async for event in stream_status(stage, message):
                                yield event

                elif node_name == "validate_node":
                    ok = update.get("validation_ok", False)
                    if not ok:
                        async for event in stream_status(
                            "retrying",
                            "Could not fully verify citations; retrying with stricter grounding…",
                        ):
                            yield event
                    else:
                        async for event in stream_status("verifying", "Verifying citations…"):
                            yield event

                if "grounded_answer" in update and update["grounded_answer"] is not None:
                    grounded = update["grounded_answer"]
                if "validation_ok" in update:
                    validation_ok = update["validation_ok"]
                if "registry_passages" in update:
                    all_passages.extend(update["registry_passages"])

    except Exception as exc:
        turn_log.error("graph failed", error=str(exc))
        async for event in stream_error(f"Assistant run failed: {exc}"):
            yield event
        return

    turn_log.info("turn done", elapsed=round(time.perf_counter() - t0, 2), ok=validation_ok)

    if grounded is None:
        async for event in stream_error("Assistant run failed before producing an answer."):
            yield event
        return

    async for event in stream_status("streaming", "Preparing answer…"):
        yield event

    registry = TurnRegistry()
    registry.register_many(all_passages)

    async for event in stream_grounded_turn_and_persist(
        client=client,
        thread_id=thread_id,
        user_message=user_message,
        thread_title=thread_title,
        answer=grounded,
        registry=registry,
        validation=ValidationResult(ok=validation_ok),
    ):
        yield event
