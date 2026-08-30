"""Build LLM-ready context from retrieved evidence."""

from __future__ import annotations

from app.memory.evidence import Evidence


class ContextBuilder:
    """Convert retrieved evidence into a structured LLM context."""

    def __init__(self, max_items: int = 5, max_chars: int = 6000) -> None:
        self.max_items = max_items
        self.max_chars = max_chars

    def build(self, evidence: list[Evidence]) -> str:
        if not evidence:
            return "No relevant evidence was retrieved."

        selected = evidence[: self.max_items]

        sections: list[str] = []

        for index, item in enumerate(selected, start=1):
            section = (
                f"[Evidence {index}]\n"
                f"Title: {item.title}\n"
                f"Source: {item.source}\n"
                f"URL: {item.url}\n"
                f"Content: {item.content}\n"
            )

            sections.append(section)

        context = "\n".join(sections)

        if len(context) > self.max_chars:
            context = context[: self.max_chars]

        return context