from __future__ import annotations

from pathlib import Path

from ragbot.models import Chunk
from ragbot.retriever import HybridSearcher, korean_tokens, to_chunk, to_document
from ragbot.store import IndexMeta, VectorStore

from .conftest import FakeEmbedder


def _store(embedder: FakeEmbedder, texts: list[str]) -> VectorStore:
    chunks = [Chunk(id=f"d#{i}", text=t, source="d.pdf", page=i + 1) for i, t in enumerate(texts)]
    vectors = embedder.encode(texts)
    return VectorStore.build(
        chunks,
        vectors,
        IndexMeta.now(
            embed_model="fake-model",
            dimension=int(vectors.shape[1]),
            chunk_size=200,
            chunk_overlap=40,
            chunk_count=len(chunks),
        ),
    )


def test_숫자와_영문은_토큰으로_보존된다():
    tokens = korean_tokens("지원대상은 39세 이하이며 CES 수상기업 포함")
    assert "39세" in tokens
    assert "ces" in tokens


def test_한글_어절은_바이그램으로_쪼개져_조사를_흡수한다():
    # "지원대상은"과 "지원대상"이 겹치는 바이그램을 공유해야 BM25가 둘을 잇는다.
    조사포함 = set(korean_tokens("지원대상은"))
    조사없음 = set(korean_tokens("지원대상"))
    assert 조사포함 & 조사없음


def test_공백이_소실된_공문도_토큰이_생긴다():
    tokens = korean_tokens("◦(대상)39세이하및창업3년이내창업자(예비)")
    assert "39세이하및창업3년이내창업자" in tokens
    assert len(tokens) > 5  # 바이그램 덕에 부분 일치가 가능해진다


def test_Document_왕복에서_인용_정보가_보존된다():
    original = Chunk(id="a#p2#1", text="본문", source="a.pdf", page=2)
    assert to_chunk(to_document(original)) == original


def test_BM25만으로도_정확한_토큰을_찾아낸다(embedder: FakeEmbedder):
    # dense_weight=0 이면 순수 BM25. 법조문처럼 드문 정확 토큰이 대상이다.
    store = _store(
        embedder,
        [
            "사업화 자금 최대 1억원 지원",
            "신산업 분야는 제3조제1항에 따른다",
            "담당부서 연락처 안내",
        ],
    )
    searcher = HybridSearcher(store, embedder, k=3, dense_weight=0.0)
    assert "제3조제1항" in searcher.search("제3조제1항")[0].chunk.text


def test_앙상블은_두_검색기를_섞고_중복을_없앤다(embedder: FakeEmbedder):
    store = _store(embedder, ["사업화 자금 최대 1억원", "글로벌 보육기관", "신청 마감 2월 12일"])
    hits = HybridSearcher(store, embedder, k=3, dense_weight=0.5).search("사업화 자금 1억원")
    assert len({hit.chunk.id for hit in hits}) == len(hits)  # 같은 청크가 두 번 나오지 않는다
    assert all(0.0 <= hit.score <= 1.0001 for hit in hits)  # 코사인 유사도로 환산돼 있다


def test_BM25가_찾아온_문서에도_점수가_붙는다(embedder: FakeEmbedder):
    store = _store(embedder, ["제3조제1항 신산업 분야", "전혀 무관한 다른 문단입니다"])
    hits = HybridSearcher(store, embedder, k=2, dense_weight=0.0).search("제3조제1항")
    assert hits[0].score > 0.0  # RRF 순위가 아니라 해석 가능한 유사도여야 한다


def test_실제_공문에서_하이브리드가_동작한다(embedder: FakeEmbedder, settings, sample_pdf: Path):
    from ragbot.pipeline import ingest

    ingest(sample_pdf, embedder, settings)
    store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
    hits = HybridSearcher(store, embedder, k=3).search("39세 이하 창업 3년 이내")
    assert hits
    assert any("39세" in hit.chunk.text for hit in hits)
