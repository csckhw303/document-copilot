"""PydanticAI document agent definition."""

from __future__ import annotations

from pathlib import Path

from langfuse import get_client, observe
from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import GroundedAnswer
from app.assistant.status import emit_agent_done, emit_agent_start
from app.assistant.tools import (
    read_chunks,
    read_surrounding_chunks,
    search_filings,
)
from app.config import settings

_INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.md")
INSTRUCTIONS = _INSTRUCTIONS_PATH.read_text(encoding="utf-8")

_document_agent: Agent[DocumentAgentDeps, GroundedAnswer] | None = None


def get_document_agent() -> Agent[DocumentAgentDeps, GroundedAnswer]:
    global _document_agent
    if _document_agent is None:
        # LangfuseAsyncOpenAI is a drop-in wrapper that auto-traces every LLM call
        openai_client = LangfuseAsyncOpenAI(api_key=settings.openai_api_key)
        model = OpenAIChatModel(
            settings.openai_chat_model,
            provider=OpenAIProvider(openai_client=openai_client),
        )
        _document_agent = Agent(
            model,
            deps_type=DocumentAgentDeps,
            output_type=GroundedAnswer,
            instructions=INSTRUCTIONS,
            tools=[search_filings, read_chunks, read_surrounding_chunks],
        )
    return _document_agent


@observe(name="document-agent", as_type="agent", capture_input=False, capture_output=False)
def run_document_agent(query: str, deps: DocumentAgentDeps) -> GroundedAnswer:
    lf = get_client()
    deps.trace_id = lf.get_current_trace_id()  # capture while @observe span is open
    lf.update_current_span(
        input={"query": query},
        metadata={
            "user_id": str(deps.user_id),
            "session_id": str(deps.thread_id),
            "model": settings.openai_chat_model,
        },
    )
    emit_agent_start(
        deps,
        model=settings.openai_chat_model,
        request_limit=settings.openai_agent_request_limit,
    )
    result = get_document_agent().run_sync(
        query,
        deps=deps,
        usage_limits=UsageLimits(request_limit=settings.openai_agent_request_limit),
    )
    usage = result.usage
    emit_agent_done(
        deps,
        requests=usage.requests or 0,
        tool_calls=usage.tool_calls or 0,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    lf.update_current_span(
        output={
            "answer": result.output.answer[:300],
            "citations": len(result.output.citations),
            "insufficient_evidence": result.output.insufficient_evidence,
        },
    )
    return result.output
