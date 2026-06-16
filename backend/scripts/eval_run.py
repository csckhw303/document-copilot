"""Run evaluation experiments against the Langfuse dataset.

Usage:
    uv run python scripts/eval_run.py

Each run:
 1. Fetches all items from 'sec-filing-qa-v1' dataset in Langfuse
 2. Runs each question through the real document agent (same code as production)
 3. Scores each result with 4 evaluators (3 rule-based + 1 LLM-as-judge)
 4. Posts scores to Langfuse — view at http://localhost:3000 → Traces

Run again after changing the prompt or model to compare experiments side-by-side.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import re
import uuid

from app.tracing import init_tracing

init_tracing()

import structlog  # noqa: E402
from openai import OpenAI  # noqa: E402

import app.assistant.agent as _agent_module  # noqa: E402
from app.assistant.agent import run_document_agent  # noqa: E402
from app.assistant.deps import DocumentAgentDeps, TurnRegistry  # noqa: E402
from app.assistant.outputs import GroundedAnswer  # noqa: E402
from app.config import settings  # noqa: E402
from app.retrieval.retriever import DocumentRetriever  # noqa: E402
from langfuse import get_client  # noqa: E402

log = structlog.get_logger()

DATASET_NAME = "sec-filing-qa-v1"
_EVAL_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
ITEM_TIMEOUT_SEC = 180  # max seconds per item before giving up

_retriever = DocumentRetriever()


# ── Agent runner ──────────────────────────────────────────────────────────────

def _run_item(question: str) -> tuple[GroundedAnswer | None, str | None]:
    """Runs in its own thread so asyncio.run() inside gets a fresh event loop.
    Resets the agent singleton so the LangfuseAsyncOpenAI client is recreated
    in this thread's event loop — prevents hangs on reuse across threads.
    Returns (output, trace_id) — trace_id is captured at @observe entry so it
    survives even when the agent raises (e.g. UsageLimitExceeded)."""
    _agent_module._document_agent = None  # force fresh client in this thread's event loop
    deps = DocumentAgentDeps(
        retriever=_retriever,
        registry=TurnRegistry(),
        thread_id=uuid.uuid4(),
        user_id=_EVAL_USER_ID,
    )
    output: GroundedAnswer | None = None
    try:
        output = run_document_agent(question, deps)
    except Exception as exc:
        # Agent raised (e.g. UsageLimitExceeded) — deps.trace_id is still set
        # because @observe captured it before run_sync() was called.
        log.warning("agent raised exception", error=str(exc), question=question[:60])
    finally:
        # Explicitly close the httpx client before the thread exits so the GC
        # destructor doesn't fire in the wrong async context (silences the
        # AsyncHttpxClientWrapper._state AttributeError warning).
        agent = _agent_module._document_agent
        if agent is not None:
            try:
                import asyncio
                provider = agent.model.client  # type: ignore[attr-defined]
                asyncio.run(provider.close())
            except Exception:
                pass
        _agent_module._document_agent = None
    return output, deps.trace_id


# ── Evaluators ────────────────────────────────────────────────────────────────

def _eval_has_citations(output: GroundedAnswer | None) -> tuple[float, str]:
    if output is None:
        return 0.0, "agent failed"
    passed = bool(_CITATION_MARKER_RE.search(output.answer)) and len(output.citations) > 0
    return (1.0 if passed else 0.0), f"{len(output.citations)} citation(s)"


def _eval_citation_count(output: GroundedAnswer | None) -> tuple[float, str]:
    if output is None:
        return 0.0, "agent failed"
    return float(len(output.citations)), f"{len(output.citations)} citations"


def _eval_flagged_insufficient(output: GroundedAnswer | None) -> tuple[float, str]:
    if output is None:
        return 1.0, "agent failed"
    flagged = output.insufficient_evidence
    return (1.0 if flagged else 0.0), ("flagged no evidence" if flagged else "produced answer")


def _eval_correctness(
    question: str,
    output: GroundedAnswer | None,
    expected_answer: str,
) -> tuple[float, str]:
    if output is None:
        return 0.0, "agent failed"
    prompt = f"""\
You are evaluating an AI answer to a SEC filing question.

Question: {question}
Expected answer (gold standard): {expected_answer}
Actual answer: {output.answer}

Score 0.0–1.0:
- 1.0  All key facts match (numbers, company, fiscal year)
- 0.75 Most facts correct, minor omission
- 0.5  Partially correct
- 0.25 Mostly wrong but on-topic
- 0.0  Wrong, refused, or insufficient evidence flagged

Reply with JSON only: {{"score": <float>, "reason": "<one sentence>"}}"""
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_grounding_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0)), result.get("reason", "")
    except Exception as exc:
        return 0.0, f"judge failed: {exc}"


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    lf = get_client()
    dataset = lf.get_dataset(DATASET_NAME)
    items = dataset.items
    print(f"Loaded '{DATASET_NAME}' — {len(items)} items.\n")

    run_tag = f"eval-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    rows: list[dict] = []

    for i, item in enumerate(items, 1):
        question: str = item.input["question"]
        expected: str = (item.expected_output or {}).get("answer", "")
        print(f"[{i}/{len(items)}] {question[:70]}…")

        # Run agent in isolated thread — gives asyncio.run() a fresh event loop
        output: GroundedAnswer | None = None
        trace_id: str | None = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_item, question)
            try:
                output, trace_id = future.result(timeout=ITEM_TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                log.error("item timed out", question=question[:60])
            except Exception as exc:
                log.error("item failed", error=str(exc), question=question[:60])

        # Score
        has_cite, has_cite_comment     = _eval_has_citations(output)
        cite_n,   cite_n_comment       = _eval_citation_count(output)
        insuff,   insuff_comment       = _eval_flagged_insufficient(output)
        correct,  correct_comment      = _eval_correctness(question, output, expected)
        passed = has_cite == 1.0 and correct >= 0.5

        print(
            f"  citations={cite_n:.0f}  correct={correct:.2f}  "
            f"pass={'✓' if passed else '✗'}  trace={'ok' if trace_id else 'MISSING'}"
            f"  — {correct_comment}"
        )

        rows.append({
            "question": question,
            "has_citations": has_cite,
            "citation_count": cite_n,
            "flagged_insufficient": insuff,
            "correctness": correct,
            "passed": passed,
            "run": run_tag,
        })

        # Always link to the dataset run (item appears even without a trace)
        try:
            lf.api.dataset_run_items.create(
                run_name=run_tag,
                dataset_item_id=item.id,
                trace_id=trace_id,  # None is accepted — item still appears in run
            )
        except Exception as exc:
            log.warning("failed to link dataset run item", error=str(exc))

        # Attach scores to the trace (only when we have a trace_id)
        if trace_id:
            for name, value, comment in [
                ("has_citations",        has_cite,          has_cite_comment),
                ("citation_count",       cite_n,            cite_n_comment),
                ("flagged_insufficient", insuff,            insuff_comment),
                ("correctness",          correct,           correct_comment),
                ("passed",               1.0 if passed else 0.0, "pass" if passed else "fail"),
            ]:
                lf.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
        else:
            log.warning("trace_id missing — scores not posted to Langfuse", question=question[:60])

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(rows)
    passed_n    = sum(r["passed"] for r in rows)
    avg_correct = sum(r["correctness"] for r in rows) / n if n else 0
    avg_cite    = sum(r["citation_count"] for r in rows) / n if n else 0

    print(f"\n── {run_tag} ───────────────────────────────────────────")
    print(f"  Items run  : {n}/{len(items)}")
    print(f"  Pass rate  : {passed_n}/{n}  ({100*passed_n/n:.0f}%)" if n else "  No results")
    print(f"  Avg correct: {avg_correct:.2f}")
    print(f"  Avg cite_n : {avg_cite:.1f}")
    print(f"\nView traces in Langfuse: http://localhost:3000 → Traces (filter by tag '{run_tag}')")

    lf.flush()


if __name__ == "__main__":
    main()
