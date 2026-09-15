from fastapi.testclient import TestClient
import asyncio
from unittest.mock import AsyncMock, patch

from app.agents.agent import ResearchAgent, research_agent
from app.core.planner import ResearchPlan
from app.main import app
from app.observability.context import ResearchContext


def test_web_page_and_api_coexist() -> None:
    client = TestClient(app)

    page = client.get("/")
    health = client.get("/health")

    assert page.status_code == 200
    assert "Enterprise Research Agent" in page.text
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_research_endpoint_returns_accepted_run_immediately() -> None:
    pending_run = ResearchContext(query="Analyze a company")
    with patch.object(research_agent, "start", new=AsyncMock(return_value=pending_run)):
        response = TestClient(app).post("/research", json={"query": "Analyze a company"})

    assert response.status_code == 202
    assert response.json()["run_id"] == pending_run.run_id
    assert response.json()["status"] == "pending"


def test_agent_uses_query_from_research_context() -> None:
    agent = ResearchAgent()
    plan = ResearchPlan(goal="Test goal", tasks=["Test task"])

    with (
        patch("app.agents.agent.create_research_plan", new=AsyncMock(return_value=plan)) as planner,
        patch("app.agents.agent.execute_task", new=AsyncMock(return_value="Finding")),
        patch.object(agent, "_build_report", new=AsyncMock(return_value="Report")),
    ):
        context = asyncio.run(agent.run("Question stored on the context"))

    planner.assert_awaited_once_with("Question stored on the context")
    assert context.status.value == "completed"
