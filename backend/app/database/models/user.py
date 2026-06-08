from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.database.models.chat_thread import ChatThread


class User(Base, TimestampMixin):
    """User profile linked to Supabase auth.users.
    
    The id should match auth.users.id from Supabase Auth.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))

    chat_threads: Mapped[list[ChatThread]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
