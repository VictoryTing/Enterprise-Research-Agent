"""Tests for search-provider failure handling."""

import asyncio

from app.core import executor


def test_search_provider_error_triggers_retries(
    monkeypatch,
) -> None:
    attempts = 0

    def failing_search(query: str) -> dict:
        nonlocal attempts
        attempts += 1

        return {
            "query": query,
            "results": [],
            "error": "Simulated provider failure",
        }

    monkeypatch.setitem(
        executor.TOOL_FUNCTIONS,
        "search_web",
        failing_search,
    )

    result = asyncio.run(
        executor.execute_tool(
            tool_name="search_web",
            arguments='{"query": "NVIDIA Triton"}',
            max_retries=2,
            timeout=1,
        )
    )

    assert attempts == 3
    assert result["success"] is False
    assert result["data"] is None
    assert result["attempts"] == 3
    assert result["error"] == (
        "Simulated provider failure"
    )
def test_empty_search_results_are_valid(
    monkeypatch,
) -> None:
    attempts = 0

    def empty_search(query: str) -> dict:
        nonlocal attempts
        attempts += 1

        return {
            "query": query,
            "results": [],
        }

    monkeypatch.setitem(
        executor.TOOL_FUNCTIONS,
        "search_web",
        empty_search,
    )

    result = asyncio.run(
        executor.execute_tool(
            tool_name="search_web",
            arguments='{"query": "NVIDIA Triton"}',
            max_retries=2,
            timeout=1,
        )
    )

    assert attempts == 1
    assert result["success"] is True
    assert result["error"] is None
    assert result["data"]["results"] == []