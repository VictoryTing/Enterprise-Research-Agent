from fastapi import FastAPI

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