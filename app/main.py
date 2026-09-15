from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Enterprise Research Agent",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "enterprise-research-agent",
    }


app.include_router(router)

# The product demo is intentionally served by the same FastAPI process as the
# API so it has no CORS or separate deployment configuration during Phase 1.
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "web", html=True),
    name="web",
)
