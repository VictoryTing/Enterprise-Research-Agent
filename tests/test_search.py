import json
from unittest.mock import MagicMock, patch

import app.tools.search as search_module


def test_brave_response_is_mapped_to_evidence_format() -> None:
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "web": {
                "results": [
                    {
                        "title": "Example title",
                        "url": "https://example.com/article",
                        "description": "Example result description",
                    }
                ]
            }
        }
    ).encode("utf-8")
    connection = MagicMock()
    connection.__enter__.return_value = response

    with (
        patch.object(search_module, "BRAVE_SEARCH_API_KEY", "test-brave-key"),
        patch("app.tools.search.urlopen", return_value=connection) as open_request,
    ):
        result = search_module.search_brave("enterprise research")

    request = open_request.call_args.args[0]
    assert request.get_header("X-subscription-token") == "test-brave-key"
    assert "q=enterprise+research" in request.full_url
    assert result == {
        "query": "enterprise research",
        "results": [
            {
                "title": "Example title",
                "url": "https://example.com/article",
                "snippet": "Example result description",
            }
        ],
    }


def test_unknown_provider_returns_safe_error() -> None:
    with patch.object(search_module, "SEARCH_PROVIDER", "unknown"):
        result = search_module.search_web("test")

    assert result["results"] == []
    assert "Unsupported SEARCH_PROVIDER" in result["error"]


def test_search_web_uses_configured_result_budget() -> None:
    with (
        patch.object(search_module, "SEARCH_PROVIDER", "brave"),
        patch.object(search_module, "SEARCH_MAX_RESULTS", 3),
        patch.object(search_module, "search_brave", return_value={"query": "test", "results": []}) as brave,
    ):
        search_module.search_web("test")

    brave.assert_called_once_with("test", max_results=3)
