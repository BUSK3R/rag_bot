"""실모델 종단 검증: 검색 → 프롬프트 → LLM 생성 → 답변 품질 점검.

가짜 부품 테스트가 절대 답하지 못하는 질문들을 여기서 확인한다.

  1. 답이 공문의 사실과 일치하는가 (기대 토큰이 답변에 있는가)
  2. 근거 번호 [n] 인용을 실제로 다는가
  3. 코퍼스에 없는 질문을 지어내지 않고 반려하는가
  4. 첫 토큰까지 시간(TTFT)과 총 생성 시간

    python scripts/eval_generation.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragbot.config import get_settings  # noqa: E402
from ragbot.pipeline import RagPipeline  # noqa: E402

# (질문, 답변에 반드시 들어 있어야 할 사실 토큰들)
FACT_CASES: list[tuple[str, list[str]]] = [
    ("청년창업사관학교 지원 대상의 나이 조건과 창업 기간 조건은?", ["39세", "3년"]),
    ("청년창업사관학교 사업화 자금은 최대 얼마까지 지원되나요?", ["1억"]),
    ("신청 접수 마감은 언제인가요?", ["2월 12일"]),
    ("글로벌창업사관학교는 몇 개사를 선정하고 사업화 자금은 얼마인가요?", ["60", "1.5억"]),
    ("청년창업사관학교는 전국에 몇 개소가 운영되나요?", ["18개"]),
]

# 코퍼스에 답이 없는 질문 — 지어내면 실패다.
REFUSE_CASES: list[str] = [
    "소상공인 대환대출 금리는 몇 퍼센트인가요?",
    "2026년 모집 일정은 어떻게 되나요?",
]

CITATION = re.compile(r"\[\d+\]")


def main() -> int:
    settings = get_settings()
    print(f"LLM: {settings.llm_model}")
    pipeline = RagPipeline.from_settings(settings)
    if not pipeline.llm_ready():
        print("[중단] Ollama 가 응답하지 않거나 모델이 없습니다.")
        return 1
    print(f"인덱스: 청크 {pipeline.chunk_count}개\n")

    passed = 0
    for question, expected in FACT_CASES:
        started = time.perf_counter()
        hits, usable = pipeline.ground(question)
        retrieved = time.perf_counter() - started

        first_token = None
        parts: list[str] = []
        for piece in pipeline.stream(question, usable):
            if first_token is None:
                first_token = time.perf_counter() - started
            parts.append(piece)
        answer = "".join(parts)
        total = time.perf_counter() - started

        missing = [token for token in expected if token not in answer]
        cited = bool(CITATION.search(answer))
        ok = not missing and cited
        passed += ok

        print(f"{'✅' if ok else '❌'} {question}")
        print(
            f"   검색 {retrieved:.2f}s · 첫토큰 {first_token:.2f}s · 총 {total:.2f}s"
            f" · 근거 {len(usable)}개 · 인용 {'O' if cited else 'X'}"
        )
        if missing:
            print(f"   누락된 사실: {missing}")
        print(f"   답변: {answer[:180].replace(chr(10), ' ')}\n")

    refused = 0
    for question in REFUSE_CASES:
        answer = pipeline.ask(question)
        # grounded=False(검색 단계 반려) 또는 모델이 스스로 없다고 밝히면 합격
        ok = (
            (not answer.grounded)
            or ("명시되어 있지 않" in answer.text)
            or ("찾지 못" in answer.text)
        )
        refused += ok
        label = "반려" if not answer.grounded else "생성"
        print(f"{'✅' if ok else '❌'} [{label}] {question}")
        print(f"   답변: {answer.text[:150].replace(chr(10), ' ')}\n")

    print(f"사실 정확: {passed}/{len(FACT_CASES)} · 환각 억제: {refused}/{len(REFUSE_CASES)}")
    return 0 if passed == len(FACT_CASES) and refused == len(REFUSE_CASES) else 2


if __name__ == "__main__":
    raise SystemExit(main())
