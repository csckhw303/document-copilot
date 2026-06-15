"""OpenAI query embedding for live retrieval."""

from __future__ import annotations

import functools

from openai import OpenAI

from app.config import settings

_client_instance: OpenAI | None = None


def _client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenAI(api_key=settings.openai_api_key)
    return _client_instance


@functools.lru_cache(maxsize=256)
def embed_query(text: str) -> list[float]:
    response = _client().embeddings.create(
        input=[text],
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
    )
    embedding = response.data[0].embedding
    expected_dims = settings.openai_embedding_dimensions
    if len(embedding) != expected_dims:
        raise ValueError(
            f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
        )
    return embedding
