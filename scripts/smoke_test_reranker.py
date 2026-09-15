"""Manual smoke test for the real BGE reranker."""

from app.memory.evidence import Evidence
from app.rag.reranker import BGEReranker


query = "NVIDIA 在 AI Agent 基础设施方面有哪些产品？"

evidence_items = [
    Evidence(
        task_id=1,
        title="NVIDIA NIM",
        url="https://example.com/nim",
        content=(
            "NVIDIA NIM provides optimized inference "
            "microservices for deploying generative AI models."
        ),
    ),
    Evidence(
        task_id=1,
        title="Apple iPhone",
        url="https://example.com/iphone",
        content="Apple released a new smartphone.",
    ),
    Evidence(
        task_id=1,
        title="NVIDIA AI Enterprise",
        url="https://example.com/ai-enterprise",
        content=(
            "NVIDIA AI Enterprise provides software and "
            "infrastructure for enterprise AI applications."
        ),
    ),
]

reranker = BGEReranker(
    model_name="BAAI/bge-reranker-v2-m3",
    use_fp16=False,
)

scores = reranker.score(
    query=query,
    evidence_items=evidence_items,
)

ranked = sorted(
    evidence_items,
    key=lambda item: scores[item.evidence_id],
    reverse=True,
)

for rank, item in enumerate(ranked, start=1):
    print(
        f"{rank}. {item.title}: "
        f"{scores[item.evidence_id]:.4f}"
    )