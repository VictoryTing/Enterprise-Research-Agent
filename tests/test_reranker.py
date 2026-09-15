"""Tests for the optional BGE evidence reranker."""

from app.memory.evidence import Evidence
from app.rag.reranker import BGEReranker


class FakeBGEModel:
    """Return deterministic scores without loading a real model."""

    def __init__(self) -> None:
        self.received_pairs: list[list[str]] = []

    def compute_score(
        self,
        pairs: list[list[str]],
        normalize: bool = False,
    ) -> list[float]:
        assert normalize is True

        self.received_pairs = pairs

        return [0.91, 0.18]


class StubBGEReranker(BGEReranker):
    """BGE wrapper whose model loading is replaced for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fake_model = FakeBGEModel()

    def _load_model(self) -> None:
        self._model = self.fake_model


def make_evidence(
    title: str,
    content: str,
    url: str,
) -> Evidence:
    return Evidence(
        task_id=1,
        title=title,
        content=content,
        url=url,
    )


def test_bge_reranker_maps_scores_to_evidence_ids() -> None:
    relevant = make_evidence(
        title="NVIDIA AI Infrastructure",
        content="NVIDIA provides GPU infrastructure for AI agents.",
        url="https://example.com/nvidia",
    )

    irrelevant = make_evidence(
        title="Apple Smartphone",
        content="Apple released a new smartphone.",
        url="https://example.com/apple",
    )

    reranker = StubBGEReranker()

    scores = reranker.score(
        query="NVIDIA AI Agent infrastructure",
        evidence_items=[relevant, irrelevant],
    )

    assert scores == {
        relevant.evidence_id: 0.91,
        irrelevant.evidence_id: 0.18,
    }

    assert reranker.fake_model.received_pairs == [
        [
            "NVIDIA AI Agent infrastructure",
            (
                "NVIDIA AI Infrastructure\n"
                "NVIDIA provides GPU infrastructure for AI agents."
            ),
        ],
        [
            "NVIDIA AI Agent infrastructure",
            (
                "Apple Smartphone\n"
                "Apple released a new smartphone."
            ),
        ],
    ]


def test_bge_reranker_skips_model_for_empty_input() -> None:
    reranker = BGEReranker()

    assert reranker.score(
        query="NVIDIA",
        evidence_items=[],
    ) == {}

    assert reranker.score(
        query="   ",
        evidence_items=[
            make_evidence(
                title="NVIDIA",
                content="AI infrastructure",
                url="https://example.com/nvidia",
            )
        ],
    ) == {}

    assert reranker._model is None