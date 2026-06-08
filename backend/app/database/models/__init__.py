"""
Database models package.

Import all models here so Alembic autogenerate can discover them.
"""

from app.database.base import Base
from app.database.models.chat_message import ChatMessage
from app.database.models.chat_thread import ChatThread
from app.database.models.document_chunk import DocumentChunk
from app.database.models.document_table import DocumentTable
from app.database.models.message_citation import MessageCitation
from app.database.models.message_role import MessageRole
from app.database.models.source_document import SourceDocument
from app.database.models.user import User

__all__ = [
    "Base",
    "User",
    "ChatThread",
    "ChatMessage",
    "MessageCitation",
    "MessageRole",
    "SourceDocument",
    "DocumentChunk",
    "DocumentTable",
]
