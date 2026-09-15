"""Provider adapters that normalize web-search results for the Agent."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.config import (
    BRAVE_SEARCH_API_KEY,
    SEARCH_PROVIDER,
    SEARCH_MAX_RESULTS,
    SEARCH_TIMEOUT_SECONDS,
    TAVILY_API_KEY,
)


def _success(query: str, results: list[dict[str, str]]) -> dict[str, Any]:
    return {"query": query, "results": results}


def _error(query: str, message: str) -> dict[str, Any]:
    return {"query": query, "results": [], "error": message}


def search_brave(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search Brave and map its response to the project's evidence format."""
    if not BRAVE_SEARCH_API_KEY:
        return _error(query, "BRAVE_SEARCH_API_KEY is not configured")

    url = "https://api.search.brave.com/res/v1/web/search?" + urlencode(
        {"q": query, "count": max_results}
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
        },
    )
    try:
        with urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return _error(query, f"Brave Search request failed with HTTP {exc.code}")
    except URLError:
        return _error(query, "Brave Search network request failed")
    except (TimeoutError, json.JSONDecodeError) as exc:
        return _error(query, f"Brave Search response could not be processed: {type(exc).__name__}")

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        }
        for item in payload.get("web", {}).get("results", [])
    ]
    return _success(query, results)


def search_tavily(query: str, max_results: int = 5) -> dict[str, Any]:
    """Keep the prior Tavily adapter available for backwards compatibility."""
    if not TAVILY_API_KEY:
        return _error(query, "TAVILY_API_KEY is not configured")

    try:
        from tavily import TavilyClient

        response = TavilyClient(api_key=TAVILY_API_KEY).search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
    except Exception as exc:
        return _error(query, f"Tavily Search request failed: {type(exc).__name__}")

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        }
        for item in response.get("results", [])
    ]
    return _success(query, results)


def search_web(query: str, max_results: int = SEARCH_MAX_RESULTS) -> dict[str, Any]:
    """Use the configured provider while exposing one stable Agent tool."""
    if SEARCH_PROVIDER == "brave":
        return search_brave(query, max_results=max_results)
    if SEARCH_PROVIDER == "tavily":
        return search_tavily(query, max_results=max_results)
    return _error(query, f"Unsupported SEARCH_PROVIDER: {SEARCH_PROVIDER}")
