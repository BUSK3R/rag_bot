from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragbot import pipeline as pipeline_module
from ragbot.api import create_app
from ragbot.config import Settings
from ragbot.pipeline import RagPipeline, ingest
from ragbot.retriever import HybridSearcher
from ragbot.store import VectorStore

from .conftest import FakeEmbedder, FakeLLM


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
    llm: FakeLLM,
    settings: Settings,
    sample_pdf: Path,
) -> TestClient:
    ingest(sample_pdf, embedder, settings)
    store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
    searcher = HybridSearcher(store, embedder, k=settings.top_k)
    ready = RagPipeline(searcher, llm, top_k=settings.top_k, min_score=settings.min_score)
    monkeypatch.setattr(RagPipeline, "from_settings", classmethod(lambda cls, _s: ready))
    return TestClient(create_app(settings))


def test_헬스가_청크수와_LLM상태를_알려준다(client: TestClient):
    with client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["llm_ready"] is True


def test_질문하면_답과_근거가_온다(client: TestClient):
    with client:
        response = client.post("/api/ask", json={"query": "지원 대상 나이 조건은?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"]
    assert body["sources"][0]["page"] >= 1
    assert 0.0 <= body["sources"][0]["score"] <= 1.0001


def test_빈_질문은_422로_거절한다(client: TestClient):
    with client:
        assert client.post("/api/ask", json={"query": ""}).status_code == 422


def test_파이프라인_적재에_실패해도_서버는_뜨고_원인을_말한다(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
):
    def boom(cls, _s):
        raise FileNotFoundError("인덱스가 없습니다")

    monkeypatch.setattr(RagPipeline, "from_settings", classmethod(boom))
    with TestClient(create_app(settings)) as broken:
        health = broken.get("/api/health").json()
        assert health["status"] == "degraded"
        assert "인덱스" in health["detail"]
        assert broken.post("/api/ask", json={"query": "안녕"}).status_code == 503


def test_조립_지점은_pipeline_한_군데뿐이다():
    # api 는 구현체(OllamaClient/Embedder)를 직접 만들지 않는다. 계층이 새면 여기서 걸린다.
    source = (Path(pipeline_module.__file__).parent / "api.py").read_text(encoding="utf-8")
    assert "OllamaClient" not in source
    assert "Embedder(" not in source


def _events(raw: str) -> list[dict]:
    import json

    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]


def test_스트리밍은_근거를_먼저_보내고_토큰을_흘린다(client: TestClient):
    with client:
        response = client.post("/api/ask/stream", json={"query": "지원 대상 나이 조건은?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    assert events[0]["type"] == "sources"  # 검색 결과가 생성보다 먼저 뜬다
    assert events[0]["sources"]
    assert events[-1] == {"type": "done", "grounded": True}

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1  # 한 번에 몰아 보내지 않는다
    assert "".join(tokens) == "테스트 답변입니다 [1]"


def test_근거가_없으면_스트림도_LLM을_부르지_않는다(
    monkeypatch: pytest.MonkeyPatch,
    embedder: FakeEmbedder,
    llm: FakeLLM,
    settings: Settings,
    sample_pdf: Path,
):
    ingest(sample_pdf, embedder, settings)
    store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
    # min_score 를 1.0 으로 두면 어떤 근거도 통과하지 못한다.
    searcher = HybridSearcher(store, embedder, k=settings.top_k)
    picky = RagPipeline(searcher, llm, top_k=settings.top_k, min_score=1.0)
    monkeypatch.setattr(RagPipeline, "from_settings", classmethod(lambda cls, _s: picky))

    with TestClient(create_app(settings)) as strict:
        events = _events(strict.post("/api/ask/stream", json={"query": "오늘 서울 날씨"}).text)

    assert events[-1] == {"type": "done", "grounded": False}
    assert llm.calls == []  # 생성 경로에 진입조차 하지 않는다
