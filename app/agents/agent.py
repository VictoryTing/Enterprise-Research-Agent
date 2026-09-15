"""Application service that orchestrates a complete research run."""

from __future__ import annotations
import asyncio
from app.rag.context_builder import ContextBuilder

from app.core.executor import execute_task
from app.core.config import RAG_MAX_CONTEXT_CHARS, RAG_MAX_EVIDENCE, WEB_SEARCH_ENABLED
from app.core.llm import llm_client
from app.core.planner import create_research_plan
from app.observability.context import ResearchContext, ResearchTask, RunStatus
from app.observability.logger import get_logger
from app.observability.tracing import TraceSpan

logger = get_logger("research_agent")


class ResearchAgent:
    def __init__(self) -> None:
        # Phase 1 deliberately uses process memory;
        # PostgreSQL replaces this in Phase 3.
        self._runs: dict[str, ResearchContext] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

        self.context_builder = ContextBuilder(
            max_items=RAG_MAX_EVIDENCE,
            max_chars=RAG_MAX_CONTEXT_CHARS,
        )

    async def run(self, query: str) -> ResearchContext:
        """Run synchronously for scripts and backwards-compatible callers."""
        context = ResearchContext(query=query)
        self._runs[context.run_id] = context
        await self._execute(context)
        return context

    async def start(self, query: str) -> ResearchContext:
        """Create a run and continue it in the current server's event loop."""
        context = ResearchContext(query=query)
        self._runs[context.run_id] = context
        task = asyncio.create_task(
            self._execute(context),
            name=f"research-run-{context.run_id}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return context

    async def _execute(self, context: ResearchContext) -> None:
        context.status = RunStatus.RUNNING
        try:
            with TraceSpan(context, "research.run", "agent"):
                with TraceSpan(context, "llm.plan", "llm"):
                    plan = await create_research_plan(context.query)
                context.goal = plan.goal
                context.tasks = [
                    ResearchTask(task_id=index, description=description)
                    for index, description in enumerate(plan.tasks, start=1)
                ]
                for task in context.tasks:
                    await execute_task(context, task.task_id)

                report = await self._build_report(context)
                context.complete(report)
                logger.info("run=%s completed evidence=%s", context.run_id, context.evidence_store.count())
        except Exception as exc:
            logger.exception("run=%s failed", context.run_id)
            context.fail(str(exc))

    def get_run(self, run_id: str) -> ResearchContext | None:
        return self._runs.get(run_id)

    async def _build_report(self, context: ResearchContext) -> str:
        findings = "\n\n".join(
            f"### Task {task.task_id}: {task.description}\n"
            f"{task.result or task.error or 'No result'}"
            for task in context.tasks
        )

        evidence = context.evidence_store.get_ranked(
            top_k=RAG_MAX_EVIDENCE
        )

        rag_context = self.context_builder.build(evidence)

        evidence_policy = (
            "Cite factual claims with the evidence IDs provided in the context. Do not invent citations or URLs."
            if WEB_SEARCH_ENABLED and evidence
            else "Start the Risks and Limitations section by stating that web retrieval was unavailable, "
            "so this is a non-real-time analysis based on model knowledge. Do not invent citations or URLs."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a concise Chinese enterprise research report with sections: "
                    "Executive Summary, Key Findings, Risks and Limitations, Sources. "
                    "Use only the provided research findings and evidence. "
                    f"{evidence_policy}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal:\n{context.goal}\n\n"
                    f"Findings:\n{findings}\n\n"
                    f"Retrieved Evidence:\n{rag_context}"
                ),
            },
        ]

        try:
            with TraceSpan(context, "llm.report_synthesis", "llm"):
                return await llm_client.chat(messages)

        except Exception as exc:
            logger.warning(
                "run=%s report synthesis failed: %s",
                context.run_id,
                exc,
            )

            return (
                "# Research Report\n\n"
                f"## Goal\n{context.goal}\n\n"
                f"## Findings\n{findings}\n\n"
                f"## Retrieved Evidence\n{rag_context}"
            )

research_agent = ResearchAgent()
