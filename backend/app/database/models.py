"""
SQLAlchemy models for Document Copilot.

These models are used by Alembic autogenerate to create migration files.
Postgres/Supabase-specific features (pgvector, generated tsvector, HNSW/GIN
indexes, RLS) are added manually in migration files.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class User(Base):
    """
    User profile linked to Supabase auth.users.
    
    Stores analyst preferences and metadata. The id matches auth.users.id
    from Supabase Auth.
    """
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    chat_threads: Mapped[list["ChatThread"]] = relationship(
        "ChatThread", back_populates="user", cascade="all, delete-orphan"
    )


class SourceDocument(Base):
    """
    Original SEC filing with metadata and normalized Markdown content.
    
    One row per filing. Stores company, filing type, date, source URL,
    accession number, and the extracted/normalized Markdown.
    """
    __tablename__ = "source_documents"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    filing_type: Mapped[str] = mapped_column(String(20), nullable=False)
    filing_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    accession_number: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_source_documents_ticker_year", "ticker", "fiscal_year"),
    )


class DocumentChunk(Base):
    """
    Retrieval-ready chunk with text, embedding, and full-text search vector.
    
    Each chunk stores:
    - chunk text and metadata (page, section, position)
    - OpenAI embedding for semantic search
    - generated tsvector for Postgres full-text search
    - token count
    
    The embedding column uses pgvector's vector type (must enable extension).
    The search_vector column is a generated tsvector (created in migration).
    HNSW index on embedding and GIN index on search_vector are created in migration.
    """
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    document_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # pgvector embedding column (1536 dimensions for OpenAI ada-002/003)
    # The VECTOR type is defined by pgvector extension
    embedding = Column(VECTOR(1536), nullable=True)
    
    # Generated tsvector for full-text search
    # This column is generated in the migration with:
    # generated always as (to_tsvector('english', text)) stored
    # Do not set this column directly
    search_vector = Column("search_vector", nullable=True)
    
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["SourceDocument"] = relationship("SourceDocument", back_populates="chunks")
    citations: Mapped[list["MessageCitation"]] = relationship(
        "MessageCitation", back_populates="chunk"
    )

    __table_args__ = (
        Index("ix_document_chunks_document_chunk", "document_id", "chunk_index"),
    )


class ChatThread(Base):
    """
    Chat conversation container.
    
    Each thread belongs to one user and contains an ordered list of messages.
    """
    __tablename__ = "chat_threads"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_threads")
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="thread", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    """
    Individual message in a chat thread.
    
    Stores both user questions and assistant answers. Assistant messages
    have linked citations through the message_citations table.
    
    The message_data JSON column stores AI SDK-compatible message format
    when needed for frontend compatibility.
    """
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    thread_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # AI SDK format
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    thread: Mapped["ChatThread"] = relationship("ChatThread", back_populates="messages")
    citations: Mapped[list["MessageCitation"]] = relationship(
        "MessageCitation", back_populates="message", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_chat_messages_thread_created", "thread_id", "created_at"),
    )


class MessageCitation(Base):
    """
    Links assistant messages to cited document chunks.
    
    This is the trust mechanism: when an assistant answer makes a claim,
    this table records which filing passage supports it. The frontend uses
    this to show "click citation → see exact source" for verification.
    
    Many-to-many between chat_messages and document_chunks, with metadata
    about the citation (relevance score, specific page/section reference).
    """
    __tablename__ = "message_citations"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    message_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    citation_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # excerpt shown
    relevance_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    page_reference: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    message: Mapped["ChatMessage"] = relationship("ChatMessage", back_populates="citations")
    chunk: Mapped["DocumentChunk"] = relationship("DocumentChunk", back_populates="citations")

    __table_args__ = (
        Index("ix_message_citations_message_chunk", "message_id", "chunk_id"),
    )
