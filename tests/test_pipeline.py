from __future__ import annotations

from pathlib import Path

from ragbot.config import Settings
from ragbot.pipeline import RagPipeline, ingest
from ragbot.prompt import NO_EVIDENCE_ANSWER
from ragbot.retriever import HybridSearcher
from ragbot.store import VectorStore

from .conftest import FakeEmbedder, FakeLLM


def _pipeline(
    embedder: FakeEmbedder, llm: FakeLLM, settings: Settings, min_score: float = 0.1
) -> RagPipeline:
    store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
    searcher = HybridSearcher(store, embedder, k=settings.top_k)
    return RagPipeline(searcher, llm, top_k=settings.top_k, min_score=min_score)


def test_색인하면_인덱스가_생긴다(embedder: FakeEmbedder, settings: Settings, sample_pdf: Path):
    report = ingest(sample_pdf, embedder, settings)
    assert report.files == 1
    assert report.chunks > 0
    assert (settings.index_dir / "index.faiss").exists()
    assert (settings.index_dir / "meta.json").exists()


def test_PDF가_없으면_실패한다(embedder: FakeEmbedder, settings: Settings, tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        ingest(empty, embedder, settings)
    except FileNotFoundError as exc:
        assert "PDF" in str(exc)
    else:
        raise AssertionError("빈 디렉터리인데 성공했습니다")


def test_근거가_있으면_LLM을_부르고_인용을_돌려준다(
    embedder: FakeEmbedder, llm: FakeLLM, settings: Settings, sample_pdf: Path
):
    ingest(sample_pdf, embedder, settings)
    answer = _pipeline(embedder, llm, settings).ask("청년창업사관학교 지원 대상 나이 조건")
    assert answer.grounded
    assert answer.text == llm.reply
    assert len(llm.calls) == 1
    assert answer.hits
    # 프롬프트에 출처 표기가 실제로 실려 나가는지
    assert "출처:" in llm.calls[0][1]
    assert "페이지)" in llm.calls[0][1]


def test_근거가_약하면_LLM을_아예_부르지_않는다(
    embedder: FakeEmbedder, llm: FakeLLM, settings: Settings, sample_pdf: Path
):
    ingest(sample_pdf, embedder, settings)
    answer = _pipeline(embedder, llm, settings, min_score=0.99).ask("오늘 서울 날씨 알려줘")
    assert not answer.grounded
    assert answer.text == NO_EVIDENCE_ANSWER
    assert llm.calls == []  # 환각 경로에 진입조차 하지 않는다


def test_top_k를_넘겨_근거_개수를_조절한다(
    embedder: FakeEmbedder, llm: FakeLLM, settings: Settings, sample_pdf: Path
):
    ingest(sample_pdf, embedder, settings)
    assert len(_pipeline(embedder, llm, settings).retrieve("사업화 자금", top_k=2)) == 2


def test_컨텍스트_예산을_넘으면_근거를_잘라낸다(
    embedder: FakeEmbedder, llm: FakeLLM, settings: Settings, sample_pdf: Path
):
    ingest(sample_pdf, embedder, settings)
    store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
    searcher = HybridSearcher(store, embedder, k=4)
    넉넉 = RagPipeline(searcher, llm, top_k=4, min_score=0.0, max_context_chars=100_000)
    빠듯 = RagPipeline(searcher, llm, top_k=4, min_score=0.0, max_context_chars=300)

    _, 많음 = 넉넉.ground("사업화 자금")
    _, 적음 = 빠듯.ground("사업화 자금")
    assert len(적음) < len(많음)
    assert len(적음) >= 1  # 1등은 예산을 넘겨도 반드시 남는다


def test_스트리밍_조각을_이으면_완성_답변이_된다(
    embedder: FakeEmbedder, llm: FakeLLM, settings: Settings, sample_pdf: Path
):
    ingest(sample_pdf, embedder, settings)
    pipeline = _pipeline(embedder, llm, settings)
    _, usable = pipeline.ground("청년창업사관학교 지원 대상")
    assert "".join(pipeline.stream("청년창업사관학교 지원 대상", usable)) == llm.reply
