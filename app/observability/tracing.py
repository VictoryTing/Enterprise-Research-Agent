"""Dependency-free tracing primitives for a research run.

The data model deliberately mirrors a trace/span system so it can later be
exported to OpenTelemetry or Langfuse without rewriting agent code.
"""

from __future__ import annotations

from time import perf_counter
from typing import Type

from app.observability.context import ResearchContext, TraceEvent, TraceStatus
from app.observability.logger import get_logger

logger = get_logger("tracing")


class TraceSpan:
    """Record duration and outcome of a synchronous or asynchronous code block."""

    def __init__(
        self,
        context: ResearchContext | None,
        name: str,
        component: str,
        task_id: int | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        self.context = context
        self.name = name
        self.component = component
        self.task_id = task_id
        self.metadata = metadata or {}
        self._started_at = 0.0
        self._error: str | None = None

    def __enter__(self) -> "TraceSpan":
        self._started_at = perf_counter()
        return self

    def mark_error(self, error: str) -> None:
        self._error = error

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        if self.context is None:
            return False

        error = self._error or (str(exc) if exc else None)
        event = TraceEvent(
            name=self.name,
            component=self.component,
            task_id=self.task_id,
            duration_ms=round((perf_counter() - self._started_at) * 1000, 2),
            status=TraceStatus.ERROR if error else TraceStatus.SUCCESS,
            metadata=self.metadata,
            error=error,
        )
        self.context.add_trace(event)
        logger.info(
            "run=%s task=%s component=%s span=%s status=%s duration_ms=%s",
            self.context.run_id,
            self.task_id,
            self.component,
            self.name,
            event.status.value,
            event.duration_ms,
        )
        return False
