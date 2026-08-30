"""Application service that orchestrates a complete research run."""

from __future__ import annotations
from app.rag.context_builder import ContextBuilder

from app.core.executor import execute_task
from app.core.llm import llm_client
from app.core.planner import create_research_plan
from app.observability.context import ResearchContext, ResearchTask, RunStatus
from app.observability.logger import get_logger

logger = get_logger("research_agent")


class ResearchAgent:
    def __init__(self) -> None:
        # Phase 1 deliberately uses process memory;
        # PostgreSQL replaces this in Phase 3.
        self._runs: dict[str, ResearchContext] = {}

        self.context_builder = ContextBuilder(
            max_items=5,
            max_chars=6000,
        )

    async def run(self, query: str) -> ResearchContext:
        context = ResearchContext(query=query, status=RunStatus.RUNNING)
        self._runs[context.run_id] = context
        try:
            plan = await create_research_plan(query)
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
        return context

    def get_run(self, run_id: str) -> ResearchContext | None:
        return self._runs.get(run_id)

    async def _build_report(self, context: ResearchContext) -> str:
        findings = "\n\n".join(
            f"### Task {task.task_id}: {task.description}\n"
            f"{task.result or task.error or 'No result'}"
            for task in context.tasks
        )

        evidence = context.evidence_store.get_ranked(
            top_k=5
        )

        rag_context = self.context_builder.build(evidence)

        messages = [
            {
                "role": "system",
                "content": (
                    "Write a concise Chinese enterprise research report with sections: "
                    "Executive Summary, Key Findings, Risks and Limitations, Sources. "
                    "Use only the provided research findings and evidence. "
                    "Cite factual claims with the evidence IDs provided in the context. "
                    "Do not invent citations or URLs."
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
