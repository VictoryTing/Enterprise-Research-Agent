from fastapi import APIRouter, HTTPException

from app.agents.agent import research_agent
from app.api.schemas import ChatRequest, ChatResponse, ResearchRequest, ResearchResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "enterprise-research-agent"}


@router.post("/research", response_model=ResearchResponse, status_code=202)
async def research(request: ResearchRequest) -> dict:
    context = await research_agent.start(request.query)
    return context.to_dict()


@router.get("/runs/{run_id}", response_model=ResearchResponse)
async def get_research_run(run_id: str) -> dict:
    context = research_agent.get_run(run_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    return context.to_dict()


@router.post("/chat", response_model=ChatResponse, deprecated=True)
async def chat(request: ChatRequest) -> ChatResponse:
    """Compatibility endpoint; clients should migrate to POST /research."""
    context = await research_agent.run(request.message)
    return ChatResponse(answer=context.report or context.error or "Research did not produce a report.")
