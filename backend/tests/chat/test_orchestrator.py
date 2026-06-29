import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.assistant.outputs import Citation, GroundedAnswer
from app.auth.dependencies import CurrentUser
from app.chat.orchestrator import run_turn
from app.retrieval.types import RetrievedPassage
from app.schemas.chat import TextPart, UIMessage


def _passage() -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        text="Azure revenue increased 29%.",
        page="5",
        section="MD&A",
        fusion_score=0.7,
        ticker="MSFT",
        company_name="Microsoft Corporation",
        form="10-K",
        filing_date=date(2024, 7, 30),
        fiscal_year=2024,
        accession_number="0000789019-24-000012",
    )


@pytest.mark.anyio
async def test_run_turn_streams_grounded_answer_and_persists() -> None:
    passage = _passage()
    grounded = GroundedAnswer(
        answer="Azure grew [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Azure revenue increased 29%.",
            )
        ],
    )
    user_message = UIMessage(role="user", parts=[TextPart(text="Azure growth?")])
    events: list[str] = []

    async def fake_astream(*args, **kwargs):
        yield {"agent_node": {"messages": [AIMessage(content="", tool_calls=[{"name": "search_filings", "id": "t1", "type": "tool_call", "args": {}}])]}}
        yield {"validate_node": {"grounded_answer": grounded, "validation_ok": True, "validation_attempts": 1, "registry_passages": [passage]}}

    with (
        patch("app.chat.orchestrator.graph.astream", fake_astream),
        patch("app.chat.streaming.append_grounded_turn", AsyncMock()) as mock_persist,
    ):
        async for event in run_turn(
            client=MagicMock(),
            thread_id=uuid.uuid4(),
            user=CurrentUser(id=uuid.uuid4(), email="a@example.com"),
            user_message=user_message,
            thread_title="New chat",
        ):
            events.append(event)

    assert events[0].startswith('data: {"type":"data-status"')
    assert any('"stage":"searching"' in event for event in events)
    assert any('"type":"text-delta"' in event for event in events)
    assert any('"type":"data-citation"' in event for event in events)
    mock_persist.assert_awaited_once()
    persisted = mock_persist.await_args.kwargs["assistant_message"]
    part_types = {part.type for part in persisted.parts}
    assert "data-status" not in part_types


@pytest.mark.anyio
async def test_run_turn_validation_failure_does_not_persist() -> None:
    grounded = GroundedAnswer(
        answer="Bad answer without markers.",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=uuid.uuid4(),
                excerpt="missing from registry",
            )
        ],
    )
    user_message = UIMessage(role="user", parts=[TextPart(text="Question")])

    async def fake_astream(*args, **kwargs):
        yield {"validate_node": {"grounded_answer": grounded, "validation_ok": False, "validation_attempts": 1}}

    with (
        patch("app.chat.orchestrator.graph.astream", fake_astream),
        patch("app.chat.streaming.append_grounded_turn", AsyncMock()) as mock_persist,
    ):
        events = [
            event
            async for event in run_turn(
                client=MagicMock(),
                thread_id=uuid.uuid4(),
                user=CurrentUser(id=uuid.uuid4(), email="a@example.com"),
                user_message=user_message,
                thread_title="New chat",
            )
        ]

    assert any('"type":"error"' in event for event in events)
    mock_persist.assert_not_awaited()


@pytest.mark.anyio
async def test_run_turn_retries_once_after_validation_failure() -> None:
    passage = _passage()
    invalid = GroundedAnswer(
        answer="Bad citation [1].",
        citations=[
            Citation(citation_index=1, chunk_id=uuid.uuid4(), excerpt="not registered")
        ],
    )
    valid = GroundedAnswer(
        answer="Azure grew [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Azure revenue increased 29%.",
            )
        ],
    )
    user_message = UIMessage(role="user", parts=[TextPart(text="Azure growth?")])

    async def fake_astream(*args, **kwargs):
        yield {"validate_node": {"grounded_answer": invalid, "validation_ok": False, "validation_attempts": 1}}
        yield {"agent_node": {"messages": [AIMessage(content="Let me try again.")]}}
        yield {"validate_node": {"grounded_answer": valid, "validation_ok": True, "validation_attempts": 2, "registry_passages": [passage]}}

    with (
        patch("app.chat.orchestrator.graph.astream", fake_astream),
        patch("app.chat.streaming.append_grounded_turn", AsyncMock()) as mock_persist,
    ):
        events = [
            event
            async for event in run_turn(
                client=MagicMock(),
                thread_id=uuid.uuid4(),
                user=CurrentUser(id=uuid.uuid4(), email="a@example.com"),
                user_message=user_message,
                thread_title="New chat",
            )
        ]

    assert any('"stage":"retrying"' in event for event in events)
    assert any('"type":"text-delta"' in event for event in events)
    mock_persist.assert_awaited_once()
