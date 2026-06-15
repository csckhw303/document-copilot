"""Initialise Langfuse observability from app settings.

Pydantic-settings loads .env into Python objects but does NOT write to
os.environ. Langfuse reads os.environ directly, so we bridge the two here.
Call init_tracing() once at startup, before any @observe functions are invoked.
"""

import os

import structlog

log = structlog.get_logger()


def init_tracing() -> None:
    from app.config import settings  # local import avoids circular deps

    if not settings.langfuse_public_key:
        log.info("langfuse disabled (LANGFUSE_PUBLIC_KEY not set)")
        return

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
    log.info("langfuse tracing enabled", host=settings.langfuse_base_url)
