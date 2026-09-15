from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2_000)


class EvidenceResponse(BaseModel):
    evidence_id: str
    task_id: int
    title: str
    url: str
    content: str
    source: str
    collected_at: str


class TaskResponse(BaseModel):
    task_id: int
    description: str
    status: str
    result: str | None = None
    error: str | None = None


class TraceResponse(BaseModel):
    name: str
    component: str
    duration_ms: float
    status: str
    task_id: int | None = None
    metadata: dict[str, str | int | float | bool]
    error: str | None = None
    timestamp: str


class ResearchResponse(BaseModel):
    run_id: str
    query: str
    status: str
    goal: str | None = None
    tasks: list[TaskResponse]
    evidence: list[EvidenceResponse]
    trace: list[TraceResponse]
    report: str | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
