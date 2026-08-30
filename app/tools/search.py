from typing import Any

from tavily import TavilyClient

from app.core.config import TAVILY_API_KEY


def search_web(query: str) -> dict[str, Any]:
    """
    Search the web using Tavily.
    """

    api_key = TAVILY_API_KEY

    if not api_key:
        return {
            "query": query,
            "results": [],
            "error": "TAVILY_API_KEY is not configured",
        }

    try:
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = []

        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )

        return {
            "query": query,
            "results": results,
        }

    except Exception as e:
        return {
            "query": query,
            "results": [],
            "error": str(e),
        }