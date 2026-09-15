import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.executor import execute_task
from app.core.planner import create_research_plan
from app.observability.context import ResearchContext, ResearchTask


def test_planner_limits_task_count_to_configured_budget() -> None:
    response = json.dumps(
        {"goal": "Test goal", "tasks": ["one", "two", "three", "four"]}
    )
    with patch("app.core.planner.llm_client.chat", new=AsyncMock(return_value=response)):
        plan = asyncio.run(create_research_plan("Test query"))

    assert plan.tasks == ["one", "two", "three"]


def test_executor_enforces_search_budget_even_when_model_requests_more() -> None:
    context = ResearchContext(query="Test query", goal="Test goal")
    context.tasks = [ResearchTask(task_id=1, description="Test task")]
    tool_calls = [
        SimpleNamespace(
            id=f"call_{index}",
            function=SimpleNamespace(name="search_web", arguments='{"query":"test"}'),
        )
        for index in range(3)
    ]
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Finding", tool_calls=[]))]),
    ]
    tool_result = {"success": True, "tool": "search_web", "data": {"results": []}, "error": None, "attempts": 1}

    with (
        patch("app.core.executor.llm_client.chat_with_tools", new=AsyncMock(side_effect=responses)),
        patch("app.core.executor.execute_tool", new=AsyncMock(return_value=tool_result)) as execute,
    ):
        result = asyncio.run(execute_task(context, task_id=1, max_iterations=2))

    assert result == "Finding"
    assert execute.await_count == 2
