from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


from app.auth.dependencies import CurrentUser, get_current_user
from app.config import settings

app = FastAPI(title="Document Copilot")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/me")
async def me(current_user: CurrentUser = Depends(get_current_user)) -> dict[str, str]:
    return {"id": str(current_user.id), "email": current_user.email}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)