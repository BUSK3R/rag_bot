"""파이프라인 전체가 주고받는 값 타입.

가장 먼저 읽어야 할 파일이다. 이 네 개만 알면 나머지 모듈의 시그니처가 전부 읽힌다.
전부 frozen — 검색 결과가 중간 계층에서 조용히 바뀌는 일이 없도록.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """인덱싱 단위. 원문의 어느 페이지에서 왔는지를 항상 들고 다닌다(인용 표기용)."""

    id: str  # 예: "sample_announcement.pdf#p2#3"
    text: str
    source: str  # 파일명
    page: int  # 1-based


@dataclass(frozen=True, slots=True)
class Hit:
    """검색 결과 한 건."""

    chunk: Chunk
    # 코사인 유사도(0~1). 높을수록 유사하다. 이전 구조는 L2 '거리'를 '유사도'라는
    # 이름으로 출력해 크고 작음이 뒤집혀 보였다 — 정규화 벡터 + 내적으로 통일했다.
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    query: str
    text: str
    hits: tuple[Hit, ...]
    # 근거 점수가 임계값에 못 미쳐 LLM 을 부르지 않고 반려한 경우 False.
    grounded: bool
