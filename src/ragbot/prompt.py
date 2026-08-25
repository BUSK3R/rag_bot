"""프롬프트 조립. 순수 함수만 두어 모델 없이도 테스트된다.

설계 의도는 하나다 — 답변이 공문에서 어디서 나왔는지 추적 가능해야 한다.
그래서 근거를 [1] [2] 로 번호 매겨 출처·페이지와 함께 넣고, 본문에 그 번호를
달도록 지시한다. 공문에 없는 내용은 지어내지 말고 없다고 말하게 한다.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Hit

SYSTEM_PROMPT = (
    "당신은 정부 공문서를 근거로 답하는 정책 안내 AI입니다.\n"
    "규칙:\n"
    "1. 제공된 [참고 공문] 내용만 근거로 삼으세요. 사전 지식으로 보충하지 마세요.\n"
    "2. 공문에 없는 내용은 추측하지 말고 '공문에 명시되어 있지 않습니다'라고 밝히세요.\n"
    "3. 문장 끝에 근거 번호를 [1] 처럼 표기하세요.\n"
    "4. 금액·연령·날짜 같은 조건은 공문의 표현을 그대로 옮기고 임의로 반올림하지 마세요.\n"
    "5. 한국어로 간결하게 답하세요."
)

# 검색 점수가 임계값에 못 미치면 LLM 을 부르지 않고 이 문장을 그대로 돌려준다.
# 근거 없는 질문에 그럴듯한 답을 만들어내는 경로 자체를 막는 편이 싸고 안전하다.
NO_EVIDENCE_ANSWER = "질문과 관련된 내용을 공문에서 찾지 못했습니다."


def build_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{n}] (출처: {hit.chunk.source}, {hit.chunk.page}페이지)\n{hit.chunk.text}"
        for n, hit in enumerate(hits, start=1)
    )


def build_user_prompt(query: str, hits: Sequence[Hit]) -> str:
    # 형식 지시는 프롬프트 말미에 예시와 함께 둔다. 소형 모델(8B 4bit)은 시스템
    # 프롬프트에만 적은 "[n] 표기" 지시를 실측에서 5/5 무시했다 — 질문 바로 뒤에
    # 구체적 예시로 반복해야 따른다.
    return (
        f"[참고 공문]\n{build_context(hits)}\n\n"
        f"[질문]\n{query}\n\n"
        "위 참고 공문만 근거로, 핵심 조건을 빠뜨리지 말고 답하세요. "
        "각 문장 끝에 근거가 된 공문 번호를 대괄호로 표기하세요. "
        "예: 지원 대상은 39세 이하입니다 [1]."
    )
