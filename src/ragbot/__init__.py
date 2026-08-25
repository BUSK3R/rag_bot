"""공문서 RAG 챗봇 패키지.

의존 방향은 아래에서 위로만 흐른다:

    config ← chunking / embedder / store / llm / prompt ← pipeline ← api

아래 계층은 위 계층을 절대 import 하지 않는다. 이전 구조가 스파게티였던 이유가
rag_engine 하나가 임베딩·검색·프롬프트·LLM 을 전부 들고 있는 신(god) 클래스였기
때문이라, 그 경계를 파일로 못박아 둔다.
"""

__version__ = "0.1.0"
