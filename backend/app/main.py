from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.config import settings
from app.logging_config import configure_logging

configure_logging()
log = structlog.get_logger()

app = FastAPI(title="Document Copilot")


@app.on_event("startup")
async def _startup() -> None:
    log.info("server starting", chat_model=settings.openai_chat_model, port=8000)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)