# rag_bot — 공문서 RAG 챗봇

정부 공문(PDF)을 근거로 답하고, **답변마다 출처와 페이지를 다는** 로컬 RAG 챗봇입니다.
외부 API 키를 쓰지 않습니다. 임베딩과 LLM 모두 내 PC에서 돕니다.

```
질문 → 하이브리드 검색(BM25 + KURE 임베딩) → 근거 부족하면 여기서 반려
     → 프롬프트 → LLM(Ollama, 스트리밍) → 인용 달린 답변
```

## 코드 읽는 순서

파일당 40~170줄, 전부 합쳐 1,100줄 남짓입니다. 아래 순서대로 읽으면 전체가 잡힙니다.

| 파일 | 역할 |
| --- | --- |
| [`models.py`](src/ragbot/models.py) | 오가는 값 네 개(`Chunk` `Hit` `Answer`). **여기부터 읽으세요.** |
| [`config.py`](src/ragbot/config.py) | 설정 단일 출처. 다른 모듈은 `os.environ`을 직접 읽지 않습니다. |
| [`chunking.py`](src/ragbot/chunking.py) | PDF → 정제 → 청크. 한글 공문의 공백 소실 대응이 여기 있습니다. |
| [`embedder.py`](src/ragbot/embedder.py) | 문장 → 정규화 벡터. |
| [`store.py`](src/ragbot/store.py) | FAISS 저장/적재/검색. |
| [`retriever.py`](src/ragbot/retriever.py) | **하이브리드 검색.** LangChain을 쓰는 유일한 곳입니다. |
| [`prompt.py`](src/ragbot/prompt.py) | 시스템 프롬프트와 인용 형식. 순수 함수뿐입니다. |
| [`llm.py`](src/ragbot/llm.py) | LLM 경계. `LLMClient`만 구현하면 백엔드를 갈아끼울 수 있습니다. |
| [`pipeline.py`](src/ragbot/pipeline.py) | 위를 엮는 조립 계층. `ask()`가 이 프로젝트의 전부입니다. |
| [`api.py`](src/ragbot/api.py) | HTTP 경계. RAG 로직은 한 줄도 없습니다. |

의존은 한 방향으로만 흐릅니다 —
`config ← chunking/embedder/store/llm/prompt ← retriever ← pipeline ← api`.
이 규칙은 주석이 아니라 [`tests/test_architecture.py`](tests/test_architecture.py)가 강제합니다.

## LangChain을 쓰는 범위

**검색 계층에만** 씁니다. `retriever.py`와 `chunking.py` 밖에서 LangChain을 import하면
테스트가 깨집니다.

| 쓰는 것 | 이유 |
| --- | --- |
| `RecursiveCharacterTextSplitter` | 경계 처리 예외가 잘 다져져 있습니다. |
| `BM25Retriever` | 정확 토큰 검색. |
| `EnsembleRetriever` | 순위 융합(RRF)은 경계 조건에서 틀리기 쉬워 검증된 구현을 씁니다. |

**안 쓰는 것**: LangGraph, 에이전트, LangSmith 트레이싱, LCEL 체인.

### 왜 하이브리드인가

행정공문은 조건이 **정확한 토큰**으로 박혀 있습니다 — `39세 이하`, `1억원`,
`2월 12일(수) 16시`, `제3조제1항`. 밀집 임베딩(KURE)은 이걸 "나이 관련", "금액 관련"
정도로 뭉개서 숫자가 다른 문단을 1등으로 올리기도 합니다. BM25가 정확히 잡습니다.

주의: `BM25Retriever`의 기본 토크나이저는 `text.split()`이라 한국어에서 거의 동작하지
않습니다. 조사가 붙어 `지원대상은`과 `지원대상`이 다른 토큰이 되고, 공백이 소실된 공문에선
`39세이하및창업3년이내`가 통째로 토큰 하나가 됩니다. `retriever.korean_tokens`가 한글
어절에 한해 문자 바이그램을 덧붙여 이 문제를 흡수합니다(숫자·영문은 정확 일치를 위해 보존).

## 빠른 시작 (로컬)

```bash
uv venv && uv pip install -e ".[dev]"
```

LLM은 Ollama로 서빙합니다. [ollama.com](https://ollama.com) 설치 후:

```bash
ollama pull hf.co/DevQuasar/kakaocorp.kanana-1.5-8b-instruct-2505-GGUF:Q4_K_M
```

색인하고 띄웁니다. 첫 실행 때 임베딩 모델(약 2.2GB)을 내려받습니다.

```bash
ragbot ingest && ragbot serve
```

http://127.0.0.1:8000 접속. 터미널에서 바로 물어볼 수도 있습니다:

```bash
ragbot ask "청년창업사관학교 지원 대상과 나이 조건은?"
```

## API

| 엔드포인트 | 설명 |
| --- | --- |
| `POST /api/ask` | 완성된 답변을 한 번에 반환합니다. |
| `POST /api/ask/stream` | SSE. **근거를 먼저 보내고** 답변을 토큰 단위로 흘립니다. |
| `GET /api/health` | 인덱스 청크 수와 LLM 준비 상태. |

웹 UI는 스트리밍을 쓰고 **첫 글자까지 걸린 시간**을 표시합니다. 체감 지연을 지배하는 건
전체 생성 시간이 아니라 첫 토큰까지의 시간이기 때문입니다.

## Docker

```bash
docker compose -f docker/compose.yaml up --build
```

기본 프로필은 **API 컨테이너만** 띄우고 LLM은 호스트 Ollama(`host.docker.internal:11434`)에
맡깁니다. Windows/macOS의 Docker Desktop은 GPU 패스스루가 까다로워서 이쪽이 훨씬 덜 아픕니다.

Ollama까지 컨테이너로 돌리려면 (NVIDIA Container Toolkit 필요):

```bash
RAGBOT_OLLAMA_BASE_URL=http://ollama:11434 docker compose -f docker/compose.yaml --profile ollama up
```

이미지에는 **CPU 전용 torch**만 들어갑니다. 임베딩은 CPU로 돌리고 GPU는 Ollama가 씁니다.

## 설정

모든 값은 `RAGBOT_` 접두사 환경변수 또는 `.env`로 덮어씁니다. `.env.example`를 복사해 쓰세요.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RAGBOT_LLM_MODEL` | kanana-1.5-8b Q4_K_M | Ollama 모델 이름 |
| `RAGBOT_TOP_K` | 4 | 검색할 청크 수 |
| `RAGBOT_MIN_SCORE` | 0.30 | 최고 점수가 이보다 낮으면 LLM을 부르지 않습니다 |
| `RAGBOT_MAX_CONTEXT_CHARS` | 2400 | 프롬프트에 실을 근거 총 글자수. **응답 속도에 직결됩니다** |
| `RAGBOT_DENSE_WEIGHT` | 0.5 | 하이브리드에서 밀집 검색 비중. 나머지가 BM25 |
| `RAGBOT_CHUNK_SIZE` | 1200 | 바꾸면 다시 `ingest` 해야 합니다 |
| `RAGBOT_EMBED_DEVICE` | cpu | `cuda`로 바꾸면 임베딩도 GPU |

## 테스트

```bash
pytest -q
```

49개가 임베딩 모델도 Ollama 서버도 없이 10초 만에 끝납니다 — `tests/conftest.py`의 가짜
부품이 파이프라인 전체를 대신 돌립니다. `embedder.py`가 torch를 늦게 import하기 때문에
가능하고, 그 사실 자체도 테스트가 지킵니다.

## 설계 메모

- **근거가 약하면 LLM을 아예 부르지 않습니다.** 환각을 사후에 걸러내는 것보다 생성 경로에
  진입하지 않는 편이 확실하고 빠릅니다. 임계값은 개별 청크가 아니라 "질문이 이 문서와
  상관은 있는가"에 겁니다 — 개별로 자르면 BM25가 찾아온 정확 일치 문서가 탈락해
  하이브리드를 쓰는 의미가 없어집니다.
- **앙상블 순위 대신 코사인 유사도를 표시합니다.** RRF 점수는 화면에 띄우기엔 의미가
  흐려서, `store.scores_for`로 실제 유사도를 되찾아 붙입니다.
- **인덱스에 pickle을 쓰지 않습니다.** LangChain FAISS 포맷은 적재 시
  `allow_dangerous_deserialization=True`가 필요했습니다 — 인덱스 파일 하나로 임의 코드가
  실행된다는 뜻이라 배포할 물건에는 둘 수 없습니다. JSON + FAISS 바이너리로 나눴습니다.
- **`faiss.write_index`를 쓰지 않습니다.** C 레벨에서 좁은 경로로 `fopen`하기 때문에
  Windows에서 경로에 한글이 하나만 있어도 죽습니다. 직렬화 후 파이썬이 파일을 씁니다.
- **청크 1200자.** KURE-v1의 `max_seq_length`는 8192입니다. 500자로 자르면 공문의
  조건절이 문장 중간에서 끊깁니다.

## 알려진 이슈

- `langchain-community`는 sunset 상태입니다(import 시 경고). `BM25Retriever`가 거기 있어
  쓰고 있으나, `retriever.py` 뒤에 가려져 있어 교체는 그 파일만 고치면 됩니다.

## 참고

`legacy/`는 재작성 전 원본입니다. 동작하지 않으며 참고용으로만 남겨두었습니다.
`data/faiss_index/`도 이전 LangChain 피클 포맷이라 새 코드는 읽지 않습니다
(새 인덱스는 `data/index/`).
