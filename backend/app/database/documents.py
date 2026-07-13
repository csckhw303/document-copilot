"""Chunk and source-document lookups for retrieval and agent tools."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, joinedload

from app.database.models import DocumentChunk, SourceDocument


def get_chunks_by_ids(
    session: Session,
    chunk_ids: list[UUID],
) -> dict[UUID, DocumentChunk]:
    if not chunk_ids:
        return {}

    rows = session.scalars(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(chunk_ids))
    ).all()
    return {row.id: row for row in rows}


def get_chunk_with_document(
    session: Session,
    chunk_id: UUID,
) -> tuple[DocumentChunk, SourceDocument] | None:
    chunk = session.scalar(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id == chunk_id)
    )
    if chunk is None or chunk.document is None:
        return None
    return chunk, chunk.document


def get_surrounding_chunks(
    session: Session,
    chunk_id: UUID,
    radius: int,
) -> list[DocumentChunk]:
    if radius < 1:
        return []

    anchor = session.scalar(
        select(DocumentChunk).where(DocumentChunk.id == chunk_id)
    )
    if anchor is None:
        return []

    min_index = anchor.chunk_index - radius
    max_index = anchor.chunk_index + radius
    return list(
        session.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == anchor.document_id,
                DocumentChunk.chunk_index >= min_index,
                DocumentChunk.chunk_index <= max_index,
                DocumentChunk.id != chunk_id,
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
    )


def get_surrounding_chunks_batch(
    session: Session,
    anchors: list[DocumentChunk],
    radius: int,
) -> dict[UUID, list[DocumentChunk]]:
    """Fetch neighbor chunks for many anchors in a single query.

    Batched replacement for calling get_surrounding_chunks() per anchor: the
    anchors are already-loaded chunks (so no per-anchor lookup is needed), and
    every anchor's [index-radius, index+radius] window is covered by one OR'd
    query. Returns anchors' neighbors keyed by anchor id, each excluding the
    anchor itself and ordered by chunk_index.
    """
    if radius < 1 or not anchors:
        return {}

    window_clauses = [
        and_(
            DocumentChunk.document_id == anchor.document_id,
            DocumentChunk.chunk_index >= anchor.chunk_index - radius,
            DocumentChunk.chunk_index <= anchor.chunk_index + radius,
        )
        for anchor in anchors
    ]
    candidates = session.scalars(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(or_(*window_clauses))
        .order_by(DocumentChunk.chunk_index)
    ).all()

    by_document: dict[UUID, list[DocumentChunk]] = defaultdict(list)
    for chunk in candidates:
        by_document[chunk.document_id].append(chunk)

    result: dict[UUID, list[DocumentChunk]] = {}
    for anchor in anchors:
        low = anchor.chunk_index - radius
        high = anchor.chunk_index + radius
        result[anchor.id] = [
            chunk
            for chunk in by_document.get(anchor.document_id, [])
            if low <= chunk.chunk_index <= high and chunk.id != anchor.id
        ]
    return result


def get_chunk_context(
    session: Session,
    chunk_id: UUID,
    radius: int,
) -> list[DocumentChunk] | None:
    anchor = session.scalar(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id == chunk_id)
    )
    if anchor is None:
        return None

    min_index = anchor.chunk_index - radius
    max_index = anchor.chunk_index + radius
    return list(
        session.scalars(
            select(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .where(
                DocumentChunk.document_id == anchor.document_id,
                DocumentChunk.chunk_index >= min_index,
                DocumentChunk.chunk_index <= max_index,
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
    )
