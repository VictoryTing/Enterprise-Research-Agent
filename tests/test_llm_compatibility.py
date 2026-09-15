import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import app.core.llm as llm_module
from httpx import Request, Response
from openai import AuthenticationError
from app.core.llm import LLMClient
from app.core.executor import get_available_tools


def test_compatible_provider_omits_openai_only_reasoning_parameter() -> None:
    client = LLMClient()
    create = AsyncMock(return_value=SimpleNamespace())
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    original_effort = llm_module.LLM_REASONING_EFFORT
    llm_module.LLM_REASONING_EFFORT = ""
    try:
        asyncio.run(client.chat_with_tools([], []))
    finally:
        llm_module.LLM_REASONING_EFFORT = original_effort

    assert "reasoning_effort" not in create.await_args.kwargs


def test_offline_mode_does_not_expose_web_search_tool() -> None:
    tool_names = [tool["function"]["name"] for tool in get_available_tools(search_enabled=False)]
    assert tool_names == ["calculate"]


def test_authentication_error_is_safe_for_frontend_display() -> None:
    provider_error = AuthenticationError(
        "Incorrect API key provided: test-api-key",
        response=Response(401, request=Request("POST", "https://provider.example")),
        body=None,
    )

    safe_error = LLMClient._safe_provider_error(provider_error)

    assert "test-api-key" not in str(safe_error)
    assert "authentication failed" in str(safe_error)
