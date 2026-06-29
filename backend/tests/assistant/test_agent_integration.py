import asyncio

import pytest

from app.assistant.graph import graph, make_initial_state
from app.assistant.state import registry_from_state
from app.grounding.validator import GroundingValidator


@pytest.mark.integration
def test_agent_answers_apple_question_with_citations() -> None:
    state = asyncio.run(
        graph.ainvoke(
            make_initial_state(
                "How did Apple describe iPhone and Services revenue in its recent 10-K filings?"
            )
        )
    )
    answer = state["grounded_answer"]
    registry = registry_from_state(state)
    validation = asyncio.run(GroundingValidator().validate(answer, registry))

    assert validation.ok
    if not answer.insufficient_evidence:
        assert answer.citations
        tickers = {
            registry.passages_by_chunk_id[c.chunk_id].ticker
            for c in answer.citations
        }
        assert "AAPL" in tickers


@pytest.mark.integration
def test_agent_refuses_underspecified_stock_pick_question() -> None:
    state = asyncio.run(
        graph.ainvoke(make_initial_state("What is the best stock to buy right now?"))
    )
    answer = state["grounded_answer"]
    registry = registry_from_state(state)
    validation = asyncio.run(GroundingValidator().validate(answer, registry))

    assert validation.ok
    assert answer.insufficient_evidence or not answer.citations
