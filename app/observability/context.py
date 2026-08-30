"""State carried through one enterprise-research run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid

from app.memory.evidence import EvidenceStore


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResearchTask:
    task_id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class ResearchContext:
    query: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: RunStatus = RunStatus.PENDING
    goal: str | None = None
    tasks: list[ResearchTask] = field(default_factory=list)
    evidence_store: EvidenceStore = field(default_factory=EvidenceStore)
    report: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def complete(self, report: str) -> None:
        self.report = report
        self.status = RunStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str) -> None:
        self.error = error
        self.status = RunStatus.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "status": self.status.value,
            "goal": self.goal,
            "tasks": [task.to_dict() for task in self.tasks],
            "evidence": [item.to_dict() for item in self.evidence_store.get_all()],
            "report": self.report,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# Keeps imports in older modules working while the project migrates.
RunContext = ResearchContext
