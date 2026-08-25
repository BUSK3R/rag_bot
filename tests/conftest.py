"""테스트용 가짜 부품.

임베딩 모델(2GB)도 Ollama 서버도 없이 파이프라인 전체를 돌리기 위한 것들이다.
embedder 가 torch 를 늦게 import 하기 때문에 가능하다 — CI 가 몇 초 만에 끝난다.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from ragbot.config import Settings

DIM = 64


def _bigrams(text: str) -> list[str]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    return [cleaned[i : i + 2] for i in range(max(len(cleaned) - 1, 0))] or [cleaned or "∅"]


class FakeEmbedder:
    """문자 바이그램을 해시해 담는 결정적 임베더. 겹치는 글자가 많으면 유사도가 높다.

    파이썬 hash() 는 프로세스마다 시드가 달라 재현이 안 되므로 crc32 를 쓴다.
    """

    @property
    def dimension(self) -> int:
        return DIM

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in _bigrams(text):
                out[row, zlib.crc32(gram.encode("utf-8")) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.ascontiguousarray(out / np.maximum(norms, 1e-9), dtype=np.float32)


class FakeLLM:
    def __init__(self, reply: str = "테스트 답변입니다 [1]", ready: bool = True) -> None:
        self.reply = reply
        self.ready = ready
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        # 실제 Ollama 처럼 여러 조각으로 나눠 보낸다(재조립이 되는지 확인하려고).
        for i in range(0, len(self.reply), 5):
            yield self.reply[i : i + 5]

    def health(self) -> bool:
        return self.ready


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        embed_model="fake-model",
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        min_score=0.1,
    )


@pytest.fixture
def sample_pdf() -> Path:
    """저장소에 들어 있는 실제 공문. 없으면 해당 테스트만 건너뛴다."""
    path = Path(__file__).resolve().parents[1] / "data" / "sample_announcement.pdf"
    if not path.exists():
        pytest.skip("샘플 PDF가 없습니다")
    return path
