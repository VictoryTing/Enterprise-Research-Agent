from app.memory.evidence import Evidence, EvidenceStore
from app.observability.context import ResearchContext, ResearchTask, RunStatus, TaskStatus
from app.observability.tracing import TraceSpan


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


def test_trace_span_records_duration_and_failure() -> None:
    context = ResearchContext(query="Trace a run")
    with TraceSpan(context, "tool.search_web", "tool", task_id=1) as span:
        span.metadata["attempts"] = 2
        span.mark_error("Tool timed out")

    trace = context.to_dict()["trace"][0]
    assert trace["name"] == "tool.search_web"
    assert trace["status"] == "error"
    assert trace["metadata"]["attempts"] == 2
    assert trace["duration_ms"] >= 0
