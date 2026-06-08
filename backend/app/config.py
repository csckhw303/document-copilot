"""
Single source of truth for environment configuration.
Never call os.getenv directly in app code — import settings from here.
"""

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment.
    Fails fast on startup if required vars are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Supabase (Auth + API) ---
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_anon_key: str = Field(..., description="Supabase anon public key")
    supabase_service_role_key: str = Field(
        ..., description="Supabase service role secret key"
    )

    # --- Postgres (Alembic + direct DB access) ---
    database_url: PostgresDsn = Field(
        ...,
        description="Direct Postgres connection (not pooler). Required for migrations.",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    openai_embedding_dimensions: int = Field(
        default=1536, description="Embedding vector dimensions"
    )

    # --- Server ---
    allowed_origins: str = Field(
        default="http://localhost:5174",
        description="Comma-separated CORS allowed origins",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse allowed_origins into a list for CORS middleware."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


# Global settings instance — import this everywhere
settings = Settings()
