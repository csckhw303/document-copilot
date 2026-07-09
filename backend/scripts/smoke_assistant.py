"""Run one assistant smoke query. Edit QUERY_KEY, then: uv run python scripts/smoke_assistant.py"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid

from app.assistant.graph import graph, make_initial_state
from app.assistant.state import registry_from_state
from app.config import settings
from app.grounding.validator import GroundingValidator

QUERIES = {
    "apple-mix": "Across Apple's 2021–2025 10-Ks, how did the revenue mix between iPhone, Services, Mac, iPad, and Wearables change?",
    "nvda-datacenter": "How did NVIDIA describe demand drivers for its Data Center business from fiscal 2021 through fiscal 2025?",
    "q10-refusal": "Do the filings prove that generative AI improved margins for any of these companies?",
    "underspecified": "What is the best stock to buy right now?",
}

QUERY_KEY = "apple-mix"


async def _run(query: str) -> None:
    t0 = time.perf_counter()
    print(f"Model: {settings.openai_chat_model}", flush=True)
    print(f"Query ({QUERY_KEY}): {query}\n", flush=True)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = await graph.ainvoke(make_initial_state(query), config=config)
    elapsed = round(time.perf_counter() - t0, 2)

    answer = state["grounded_answer"]
    registry = registry_from_state(state)
    validation = await GroundingValidator().validate(answer, registry)

    print(
        f"[{elapsed}s] done — grounding ok={validation.ok} "
        f"insufficient_evidence={answer.insufficient_evidence} "
        f"citations={len(answer.citations)}",
        flush=True,
    )
    if validation.error:
        print(f"validation_error: {validation.error}", flush=True)

    print(f"\ninsufficient_evidence: {answer.insufficient_evidence}", flush=True)
    print(f"validation_ok: {validation.ok}", flush=True)
    print(f"\n{answer.answer}\n", flush=True)

    for c in answer.citations:
        p = registry.passages_by_chunk_id.get(c.chunk_id)
        meta = f"{p.ticker} {p.form} p.{p.page}" if p else ""
        print(f"[{c.citation_index}] {meta}\n  {c.excerpt[:200]}", flush=True)


def main() -> None:
    asyncio.run(_run(QUERIES[QUERY_KEY]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)
