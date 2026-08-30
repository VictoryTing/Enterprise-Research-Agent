from app.memory.evidence import Evidence, EvidenceStore
from app.observability.context import ResearchContext, ResearchTask, RunStatus, TaskStatus


def test_evidence_store_deduplicates_urls() -> None:
    store = EvidenceStore()
    assert store.add(Evidence(task_id=1, title="First", url="https://example.com", content="a"))
    assert not store.add(Evidence(task_id=2, title="Second", url="https://example.com", content="b"))
    assert store.count() == 1


def test_research_context_serializes_run_state() -> None:
    context = ResearchContext(query="Research OpenAI")
    context.tasks = [ResearchTask(task_id=1, description="Find sources", status=TaskStatus.COMPLETED)]
    context.complete("A cited report")

    result = context.to_dict()
    assert result["status"] == RunStatus.COMPLETED.value
    assert result["tasks"][0]["status"] == TaskStatus.COMPLETED.value
    assert result["report"] == "A cited report"
