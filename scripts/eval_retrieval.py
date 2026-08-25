"""검색 방식 비교: 밀집 단독 vs BM25 단독 vs 하이브리드.

각 질문마다 정답이 담긴 청크가 몇 등으로 올라오는지(낮을수록 좋음)를 세 방식에 대해 잰다.
질문을 두 갈래로 나눈 이유가 이 스크립트의 요점이다.

  A. 자연어 질문 — 사용자가 실제로 물어보는 형태. 의미 검색이 강한 영역.
  B. 조회형 질의 — 문맥 없이 정확한 토큰만 던지는 형태("제3조제1항", "S-CoP").
     밀집 임베딩이 무너지는 지점이다.

2026-08-25 실측(1개 공고문, 25청크, KURE-v1):

              A 자연어    B 조회형
    밀집        8/8       3/8   ← 조회형에서 붕괴
    BM25        8/8       8/8
    하이브리드     7/8       8/8

즉 하이브리드는 자연어 질문에서 밀집을 이기지 못한다. 값어치는 보험에 있다 —
조회형에서 밀집이 무너지는 것을 막고, 대신 자연어에서 1문항이 1등에서 3등으로 밀린다.
운영 설정이 top_k=4 라 3등도 근거에 포함되므로 실질 손실은 없다.

주의: 공고문 한 건짜리 코퍼스다. 통계적 의미는 없고 경향만 본다.

    python scripts/eval_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragbot.config import get_settings  # noqa: E402
from ragbot.embedder import Embedder  # noqa: E402
from ragbot.retriever import HybridSearcher  # noqa: E402
from ragbot.store import VectorStore  # noqa: E402

# (질문, 정답 청크에 반드시 들어 있는 문자열)
NATURAL: list[tuple[str, str]] = [
    ("청년창업사관학교 지원 대상 나이 조건이 어떻게 되나요?", "39세"),
    ("사업화 자금은 최대 얼마까지 지원되나요?", "1억원"),
    ("신청 접수 마감은 언제까지인가요?", "2월 12일"),
    ("글로벌창업사관학교는 몇 개사를 선정하나요?", "60개"),
    ("신산업 분야는 어느 규정에 근거하나요?", "제3조제1항"),
    ("투자형 청년창업사관학교는 몇 개소인가요?", "6개소"),
    ("담당자 연락처를 알려주세요", "044-204"),
    ("서류심사를 면제받는 패스트트랙 대상은 누구인가요?", "CES"),
]

# 문맥 없이 정확 토큰만 던지는 질의. 밀집 임베딩이 가장 약한 지점이다.
LOOKUP: list[tuple[str, str]] = [
    ("1985년 1월 24일", "1985년"),
    ("044-204-7955", "7955"),
    ("제3조제1항", "제3조제1항"),
    ("폴스키센터", "폴스키"),
    ("S-CoP", "S-CoP"),
    ("경기북부 파주", "파주"),
    ("CCUS", "CCUS"),
    ("KSC GBC 인포세션", "인포세션"),
]

MODES: list[tuple[str, float]] = [
    ("밀집", 1.0),  # KURE 임베딩만
    ("BM25", 0.0),  # 정확 토큰만
    ("하이브리드", 0.5),
]


def rank_of(hits, needle: str) -> int | None:
    """정답 문자열을 담은 첫 청크의 순위(1-based). 못 찾으면 None."""
    for position, hit in enumerate(hits, start=1):
        if needle in hit.chunk.text:
            return position
    return None


def run(title: str, cases: list[tuple[str, str]], store, searchers, total: int) -> None:
    유효 = [(q, a) for q, a in cases if any(a in c.text for c in store.chunks)]
    for question, answer in cases:
        if (question, answer) not in 유효:
            print(f"  [건너뜀] '{answer}' 가 어느 청크에도 없습니다 — 질문 부적합")

    print(f"\n── {title} ──")
    header = f"{'질문':<34}{'기대 토큰':<12}" + "".join(f"{name:>12}" for name in searchers)
    print(header)
    print("-" * len(header))

    ranks: dict[str, list[int]] = {name: [] for name in searchers}
    for question, answer in 유효:
        row = f"{question[:32]:<34}{answer:<12}"
        for name, searcher in searchers.items():
            rank = rank_of(searcher.search(question, k=total), answer)
            row += f"{(rank if rank else '-'):>12}"
            ranks[name].append(rank if rank else total + 1)
        print(row)

    print("-" * len(header))
    summary = f"{'평균 순위 (낮을수록 좋음)':<46}"
    top1 = f"{'1등으로 맞힌 개수':<46}"
    for name in searchers:
        values = ranks[name]
        summary += f"{sum(values) / len(values):>12.2f}"
        top1 += f"{sum(1 for r in values if r == 1):>12}"
    print(summary)
    print(top1 + f"   (총 {len(유효)}문항)")


def main() -> int:
    settings = get_settings()
    store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
    embedder = Embedder(settings.embed_model, device=settings.embed_device)
    total = len(store)
    print(f"코퍼스: 청크 {total}개 · 임베딩 {settings.embed_model} ({settings.embed_device})")

    searchers = {
        name: HybridSearcher(store, embedder, k=total, dense_weight=weight)
        for name, weight in MODES
    }
    run("A. 자연어 질문", NATURAL, store, searchers, total)
    run("B. 조회형 질의 (정확 토큰)", LOOKUP, store, searchers, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
