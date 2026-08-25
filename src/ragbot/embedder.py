"""문장 → 벡터.

sentence-transformers(따라서 torch) 는 load() 안에서 늦게 import 한다. 임포트만으로
2GB 짜리 스택이 딸려오면 테스트와 CI 가 모델 없이는 한 발짝도 못 나가기 때문이다.
덕분에 tests/ 는 가짜 임베더로 파이프라인 전체를 검증할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class SupportsEncode(Protocol):
    """pipeline 이 의존하는 최소 계약. 테스트는 이걸 구현한 가짜를 끼운다."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Embedder:
    """KURE-v1 래퍼. 정규화된 float32 행렬만 내보낸다."""

    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        """모델을 메모리에 올린다. 여러 번 불러도 안전하다."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        self.load()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32. 각 행은 L2 정규화되어 있어 내적이 곧 코사인 유사도다.

        KURE-v1 은 bge-m3 계열이라 질의/문서에 서로 다른 지시 프리픽스를 붙이지 않는다.
        같은 함수로 양쪽을 인코딩해도 되는 이유.
        """
        self.load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
