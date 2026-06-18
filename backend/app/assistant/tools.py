"""LangChain document agent tools."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Annotated
from uuid import UUID

import structlog
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from app.config import settings
from app.database.documents import (
    get_chunk_with_document,
    get_chunks_by_ids,
    get_surrounding_chunks,
)
from app.database.models import DocumentChunk, SourceDocument
from app.database.session import get_session
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.types import RetrievedPassage, SearchFilters, format_passages_for_agent

log = structlog.get_logger()

_retriever: DocumentRetriever | None = None


def _get_retriever() -> DocumentRetriever:
    """Singleton retriever — used when running on LangGraph Cloud (no retriever in config)."""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever


def _passage_from_chunk(
    chunk: DocumentChunk,
    document: SourceDocument,
    *,
    fusion_score: float = 0.0,
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        page=chunk.page,
        section=chunk.section,
        fusion_score=fusion_score,
        ticker=document.ticker,
        company_name=document.company_name,
        form=document.form,
        filing_date=document.filing_date,
        fiscal_year=document.fiscal_year,
        accession_number=document.accession_number,
        neighbors=[],
    )


def _parse_fiscal_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    years = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return years or None


def _search_sync(
    retriever: DocumentRetriever,
    query: str,
    *,
    ticker: str | None,
    form: str | None,
    fiscal_years: str | None,
) -> list[RetrievedPassage]:
    filters = SearchFilters(
        ticker=ticker,
        form=form,
        fiscal_years=_parse_fiscal_years(fiscal_years),
    )
    return retriever.search(query, filters=filters)


def _read_chunk_sync(chunk_id: UUID) -> RetrievedPassage | None:
    with get_session() as session:
        result = get_chunk_with_document(session, chunk_id)
        if result is None:
            return None
        chunk, document = result
        return _passage_from_chunk(chunk, document)


def _read_chunks_sync(chunk_ids: list[UUID]) -> list[RetrievedPassage]:
    with get_session() as session:
        chunks_by_id = get_chunks_by_ids(session, chunk_ids)
        passages: list[RetrievedPassage] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or chunk.document is None:
                continue
            passages.append(_passage_from_chunk(chunk, chunk.document))
        return passages


def _read_surrounding_sync(chunk_id: UUID, radius: int) -> list[RetrievedPassage]:
    with get_session() as session:
        anchor = get_chunk_with_document(session, chunk_id)
        if anchor is None:
            return []
        anchor_chunk, _ = anchor
        neighbor_chunks = get_surrounding_chunks(session, chunk_id, radius)
        passages: list[RetrievedPassage] = []
        for neighbor_chunk in neighbor_chunks:
            if neighbor_chunk.document is None:
                continue
            passages.append(_passage_from_chunk(neighbor_chunk, neighbor_chunk.document))
        if anchor_chunk.document is not None:
            passages.insert(0, _passage_from_chunk(anchor_chunk, anchor_chunk.document))
        return passages


async def _run_in_thread(fn, /, *args, **kwargs):
    return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))


@tool
async def search_filings(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    ticker: str | None = None,
    form: str | None = None,
    fiscal_years: str | None = None,
) -> Command:
    """Search SEC filings with hybrid retrieval. Optional filters: ticker, form, fiscal_years (comma-separated)."""
    retriever: DocumentRetriever = config["configurable"].get("retriever") or _get_retriever()
    filter_parts = [
        s for s in (
            f"ticker={ticker}" if ticker else None,
            f"form={form}" if form else None,
            f"fiscal_years={fiscal_years}" if fiscal_years else None,
        ) if s
    ]
    log.info("tool call", tool="search_filings", detail=", ".join(filter_parts) or "no filters")
    t0 = time.perf_counter()
    passages: list[RetrievedPassage] = await _run_in_thread(
        _search_sync, retriever, query, ticker=ticker, form=form, fiscal_years=fiscal_years
    )
    log.info("tool done", tool="search_filings", results=len(passages), elapsed=round(time.perf_counter() - t0, 2))
    return Command(update={
        "messages": [ToolMessage(content=format_passages_for_agent(passages), tool_call_id=tool_call_id)],
        "registry_passages": passages,
    })


@tool
async def read_chunk(
    chunk_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the full text of a specific document chunk by UUID."""
    try:
        parsed_id = UUID(chunk_id)
    except ValueError:
        return Command(update={
            "messages": [ToolMessage(content=f"Error: invalid chunk_id {chunk_id!r}.", tool_call_id=tool_call_id)],
        })
    log.info("tool call", tool="read_chunk", chunk_id=chunk_id)
    t0 = time.perf_counter()
    passage: RetrievedPassage | None = await _run_in_thread(_read_chunk_sync, parsed_id)
    log.info("tool done", tool="read_chunk", found=passage is not None, elapsed=round(time.perf_counter() - t0, 2))
    if passage is None:
        return Command(update={
            "messages": [ToolMessage(content=f"Error: chunk {chunk_id} not found.", tool_call_id=tool_call_id)],
        })
    return Command(update={
        "messages": [ToolMessage(content=format_passages_for_agent([passage]), tool_call_id=tool_call_id)],
        "registry_passages": [passage],
    })


@tool
async def read_chunks(
    chunk_ids: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the full text of multiple document chunks in one call."""
    parsed_ids: list[UUID] = []
    for cid in chunk_ids:
        try:
            parsed_ids.append(UUID(cid))
        except ValueError:
            return Command(update={
                "messages": [ToolMessage(content=f"Error: invalid chunk_id {cid!r}.", tool_call_id=tool_call_id)],
            })
    if not parsed_ids:
        return Command(update={
            "messages": [ToolMessage(content="Error: chunk_ids must include at least one UUID.", tool_call_id=tool_call_id)],
        })
    log.info("tool call", tool="read_chunks", count=len(parsed_ids))
    t0 = time.perf_counter()
    passages: list[RetrievedPassage] = await _run_in_thread(_read_chunks_sync, parsed_ids)
    log.info("tool done", tool="read_chunks", results=len(passages), elapsed=round(time.perf_counter() - t0, 2))
    if not passages:
        return Command(update={
            "messages": [ToolMessage(content="Error: none of the requested chunks were found.", tool_call_id=tool_call_id)],
        })
    return Command(update={
        "messages": [ToolMessage(content=format_passages_for_agent(passages), tool_call_id=tool_call_id)],
        "registry_passages": passages,
    })


@tool
async def read_surrounding_chunks(
    chunk_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    radius: int | None = None,
) -> Command:
    """Read chunks before and after a given chunk within the same filing."""
    try:
        parsed_id = UUID(chunk_id)
    except ValueError:
        return Command(update={
            "messages": [ToolMessage(content=f"Error: invalid chunk_id {chunk_id!r}.", tool_call_id=tool_call_id)],
        })
    resolved_radius = radius if radius is not None else settings.retrieval_neighbor_radius
    if resolved_radius < 1:
        return Command(update={
            "messages": [ToolMessage(content="Error: radius must be 1 or greater.", tool_call_id=tool_call_id)],
        })
    log.info("tool call", tool="read_surrounding_chunks", chunk_id=chunk_id, radius=resolved_radius)
    t0 = time.perf_counter()
    passages: list[RetrievedPassage] = await _run_in_thread(_read_surrounding_sync, parsed_id, resolved_radius)
    log.info("tool done", tool="read_surrounding_chunks", results=len(passages), elapsed=round(time.perf_counter() - t0, 2))
    if not passages:
        return Command(update={
            "messages": [ToolMessage(content=f"Error: chunk {chunk_id} not found.", tool_call_id=tool_call_id)],
        })
    return Command(update={
        "messages": [ToolMessage(content=format_passages_for_agent(passages), tool_call_id=tool_call_id)],
        "registry_passages": passages,
    })


AGENT_TOOLS = [search_filings, read_chunk, read_chunks, read_surrounding_chunks]
