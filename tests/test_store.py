from __future__ import annotations

from pathlib import Path

import pytest

from ragbot.models import Chunk
from ragbot.store import IndexMeta, IndexMismatchError, VectorStore

from .conftest import FakeEmbedder


def _store(embedder: FakeEmbedder, texts: list[str], model: str = "fake-model") -> VectorStore:
    chunks = [Chunk(id=f"d#{i}", text=t, source="d.pdf", page=i + 1) for i, t in enumerate(texts)]
    vectors = embedder.encode(texts)
    meta = IndexMeta.now(
        embed_model=model,
        dimension=int(vectors.shape[1]),
        chunk_size=200,
        chunk_overlap=40,
        chunk_count=len(chunks),
    )
    return VectorStore.build(chunks, vectors, meta)


def test_저장하고_다시_읽으면_같다(embedder: FakeEmbedder, tmp_path: Path):
    store = _store(embedder, ["사업화 자금 지원", "글로벌 창업사관학교", "신청 마감일"])
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, expect_model="fake-model")
    assert len(loaded) == 3
    assert loaded.meta.chunk_count == 3


def test_피클을_쓰지_않는다(embedder: FakeEmbedder, tmp_path: Path):
    # 인덱스 파일 하나로 임의 코드가 실행되던 경로를 없앤 것이 이 설계의 요점이다.
    _store(embedder, ["가나다"]).save(tmp_path)
    assert (tmp_path / "chunks.json").exists()
    assert not list(tmp_path.glob("*.pkl"))


def test_다른_모델로_만든_인덱스는_거부한다(embedder: FakeEmbedder, tmp_path: Path):
    _store(embedder, ["가나다"], model="old-model").save(tmp_path)
    with pytest.raises(IndexMismatchError):
        VectorStore.load(tmp_path, expect_model="new-model")


def test_유사한_문서가_먼저_나오고_점수는_클수록_유사하다(embedder: FakeEmbedder):
    store = _store(
        embedder, ["사업화 자금 최대 1억원", "글로벌 보육기관 액셀러레이터", "담당부서 연락처"]
    )
    hits = store.search(embedder.encode(["사업화 자금 지원 규모"])[0], k=3)
    assert hits[0].chunk.text.startswith("사업화 자금")
    assert hits[0].score > hits[-1].score
    assert 0.0 <= hits[0].score <= 1.0001


def test_k가_문서수보다_커도_안전하다(embedder: FakeEmbedder):
    store = _store(embedder, ["하나", "둘"])
    assert len(store.search(embedder.encode(["하나"])[0], k=50)) == 2


def test_인덱스가_없으면_안내와_함께_실패한다(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        VectorStore.load(tmp_path / "없음")
