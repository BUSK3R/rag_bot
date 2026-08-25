# rag_bot — 공문서 RAG 챗봇

정부 공문(PDF)을 근거로 답하고, **답변마다 출처와 페이지를 다는** 로컬 RAG 챗봇입니다.
외부 API 키를 쓰지 않습니다. 임베딩과 LLM 모두 내 PC에서 돕니다.

```
질문 → 질의 재작성(구어체→공문 용어) → 하이브리드 검색(BM25 + KURE 임베딩)
     → 근거 부족하면 여기서 반려 → 프롬프트 → LLM(Ollama, 스트리밍) → 인용 달린 답변
```

---

## 처음 세팅 (팀원 온보딩)

Windows 기준입니다. 전부 로컬에서 돌고 API 키는 필요 없습니다.
디스크 약 10GB(임베딩 모델 2.2GB + LLM 1.6GB + venv 5GB), 첫 세팅 약 20~30분.

### 1. 도구 설치

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — 파이썬 버전과 venv 관리.
  파이썬을 따로 깔 필요 없습니다(uv가 3.11을 받아줍니다).
- **[Ollama](https://ollama.com/download)** — LLM 서빙. 설치하면 백그라운드 서비스로 뜹니다.
- git

### 2. 클론 + 의존성

```bash
git clone https://github.com/BUSK3R/rag_bot.git
cd rag_bot
uv venv --python 3.11
uv pip install -e ".[dev]"
```

torch는 sentence-transformers 의존성으로 CPU 빌드가 자동으로 깔립니다. NVIDIA GPU로
임베딩까지 돌리고 싶을 때만(선택) CUDA 빌드로 교체하세요:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. LLM 받기

```bash
ollama pull exaone3.5:2.4b
```

기본 모델은 EXAONE-3.5-2.4B(1.6GB)입니다. **NC 라이선스(연구 전용)라 시제품 단계에서만**
씁니다 — 상용화 시점에는 Apache 2.0인 Kanana로 교체합니다(아래 설정 표 참고).

### 4. 색인 + 실행

```bash
.venv\Scripts\ragbot ingest
.venv\Scripts\ragbot serve
```

- `ingest`: `data/` 안의 PDF를 청크로 잘라 FAISS 인덱스를 만듭니다.
  **첫 실행 때 임베딩 모델 KURE-v1(2.2GB)을 내려받습니다.**
- `serve`: http://127.0.0.1:8000 에 웹 챗 UI가 뜹니다.

첫 질문은 임베딩 모델 적재 때문에 15~20초 걸리고, 그다음부터는 첫 글자까지 2~3초입니다.

터미널에서 바로 물어볼 수도 있습니다:

```bash
.venv\Scripts\ragbot ask "청년창업사관학교 지원 대상과 나이 조건은?"
```

`ragbot health`로 인덱스·LLM 상태를 점검할 수 있습니다.

### 5. 테스트 (모델 없이 돕니다)

```bash
.venv\Scripts\python -m pytest -q
```

55개가 약 10초에 끝납니다. 임베딩 모델도 Ollama도 필요 없습니다 —
`tests/conftest.py`의 가짜 부품이 파이프라인 전체를 대신 돌립니다.

### 새 공문 추가하기

PDF를 `data/`에 넣고 `ragbot ingest`를 다시 돌리면 끝입니다. 청킹 설정을 바꿨을 때도
재색인이 필요합니다.

### 자주 걸리는 문제

| 증상 | 원인/해결 |
| --- | --- |
| `ragbot: command not found` | venv 활성화(`.venv\Scripts\activate`)를 안 했거나, `.venv\Scripts\ragbot`로 전체 경로 호출 |
| health에서 `llm_ready: false` | Ollama가 안 떠 있거나 모델을 pull 안 함. `ollama list`로 확인 |
| 한글이 `?`로 깨짐 | 터미널 인코딩. CLI가 자체 처리하지만, 직접 python을 부를 땐 `python -X utf8` |
| `FAISS index not found` | `ragbot ingest`를 먼저 |
| 인덱스 로드에서 모델 불일치 오류 | 인덱스를 만든 임베딩 모델과 설정이 다름. 다시 `ingest` |

---

## 코드 읽는 순서

파일당 40~200줄, 전부 합쳐 1,250줄 남짓입니다. 아래 순서대로 읽으면 전체가 잡힙니다.

| 파일 | 역할 |
| --- | --- |
| [`models.py`](src/ragbot/models.py) | 오가는 값 네 개(`Chunk` `Hit` `Answer`). **여기부터 읽으세요.** |
| [`config.py`](src/ragbot/config.py) | 설정 단일 출처. 다른 모듈은 `os.environ`을 직접 읽지 않습니다. |
| [`chunking.py`](src/ragbot/chunking.py) | PDF → 정제 → 청크. 한글 공문의 공백 소실 대응이 여기 있습니다. |
| [`embedder.py`](src/ragbot/embedder.py) | 문장 → 정규화 벡터. torch는 여기서만, 늦게 import합니다. |
| [`store.py`](src/ragbot/store.py) | FAISS 저장/적재/검색. |
| [`retriever.py`](src/ragbot/retriever.py) | **하이브리드 검색.** LangChain을 쓰는 유일한 곳입니다. |
| [`prompt.py`](src/ragbot/prompt.py) | 시스템/재작성 프롬프트와 인용 형식. 순수 함수뿐입니다. |
| [`llm.py`](src/ragbot/llm.py) | LLM 경계(Ollama HTTP). `LLMClient`만 구현하면 백엔드 교체 가능. |
| [`pipeline.py`](src/ragbot/pipeline.py) | 조립 계층. `ask()`가 이 프로젝트의 전부입니다. |
| [`api.py`](src/ragbot/api.py) | HTTP 경계. RAG 로직은 한 줄도 없습니다. |

의존은 한 방향으로만 흐릅니다 —
`config ← chunking/embedder/store/llm/prompt ← retriever ← pipeline ← api`.
이 규칙은 주석이 아니라 [`tests/test_architecture.py`](tests/test_architecture.py)가 강제합니다.

## 설계 결정과 그 실측 근거

전부 이 코퍼스에서 실제로 잰 결과로 정한 값들입니다. 코퍼스가 커지면 재보정하세요.

- **하이브리드 검색 (BM25 + 밀집).** 조회형 질의("제3조제1항", "S-CoP")에서 밀집 단독은
  8문항 중 3개만 1등, BM25는 8/8. 자연어 질문에서는 둘 다 만점. 하이브리드는 밀집의
  조회형 붕괴를 막는 **보험**입니다. `scripts/eval_retrieval.py`로 재현.
- **질의 재작성.** "64살인데 청년끼고 지원되나"에 2.4B는 오판(지원 불가로 단정), 8B는
  회피했지만, "공동창업"으로 바꿔 물으면 둘 다 정답 → 병목은 용어 매핑. 생성 전에 LLM이
  구어체를 격식체로 한 번 옮깁니다(+1초). 소형 모델은 지시문을 무시해서 **퓨샷 패턴**으로
  프롬프트를 짰습니다.
- **`min_score` 0.45.** 유관 질문은 0.55~0.75, 무관 질문("비트코인 시세", "넌뭐야")은
  0.25~0.32로 갈립니다. 게이트에 걸리면 LLM을 부르지 않고 0.1초에 반려합니다.
- **temperature 0.** 0.1에서는 자격 판정 질문이 4회 중 1회 회피로 흔들렸습니다. 사실
  질의응답에 무작위성을 둘 이유가 없습니다.
- **stop 토큰 명시.** Kanana GGUF에 stop 토큰이 없어 `<|eot_id|>`가 답변 텍스트로
  샜습니다. `llm.py`가 직접 박습니다.
- **인용 지시는 사용자 프롬프트 말미에.** 시스템 프롬프트에만 두면 2.4B가 5/5 무시했고,
  질문 뒤에 예시와 함께 반복하면 5/5 따랐습니다.
- **인덱스에 pickle 금지.** 이전 LangChain 포맷은 `allow_dangerous_deserialization`이
  필요했습니다(파일 하나로 임의 코드 실행). JSON + FAISS 바이너리로 분리.
- **`faiss.write_index` 금지.** C 레벨 좁은 경로 `fopen`이라 한글 경로에서 죽습니다
  (한글 사용자 이름이면 그걸로 끝). 직렬화 후 파이썬이 파일을 씁니다.
- **BM25 한국어 토크나이저.** 기본값이 `text.split()`이라 한국어에서 무력합니다.
  `korean_tokens`가 한글 어절에 문자 바이그램을 덧대 조사·붙임 표기를 흡수합니다.

## LangChain을 쓰는 범위

**검색 계층에만** 씁니다(`RecursiveCharacterTextSplitter`, `BM25Retriever`,
`EnsembleRetriever`). `retriever.py`/`chunking.py` 밖에서 import하면 테스트가 깨집니다.
LangGraph·에이전트·트레이싱은 쓰지 않습니다.

## API

| 엔드포인트 | 설명 |
| --- | --- |
| `POST /api/ask` | `{"query": "..."}` → 완성된 답변 + 근거 목록 |
| `POST /api/ask/stream` | SSE. **근거를 먼저 보내고** 답변을 토큰 단위로 흘립니다 |
| `GET /api/health` | 인덱스 청크 수와 LLM 준비 상태 |

## 설정

모든 값은 `RAGBOT_` 접두사 환경변수 또는 `.env`로 덮어씁니다(`.env.example` 참고).

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RAGBOT_LLM_MODEL` | `exaone3.5:2.4b` | 시제품용. 상용화 시 `hf.co/DevQuasar/kakaocorp.kanana-1.5-8b-instruct-2505-GGUF:Q4_K_M` (Apache 2.0) |
| `RAGBOT_REWRITE_QUERY` | `true` | 구어체→격식체 재작성 단계 |
| `RAGBOT_MIN_SCORE` | `0.45` | 최고 유사도가 이보다 낮으면 즉시 반려 |
| `RAGBOT_TOP_K` | `4` | 검색할 청크 수 |
| `RAGBOT_MAX_CONTEXT_CHARS` | `2400` | 근거 총 글자수 상한. 응답 속도에 직결 |
| `RAGBOT_DENSE_WEIGHT` | `0.5` | 하이브리드에서 밀집 비중(나머지 BM25) |
| `RAGBOT_CHUNK_SIZE` | `1200` | 바꾸면 재`ingest` 필요 |
| `RAGBOT_EMBED_DEVICE` | `cpu` | GPU 임베딩은 `cuda` (VRAM은 LLM과 나눠 씁니다) |
| `RAGBOT_LLM_TEMPERATURE` | `0.0` | 생성 온도 |

## 평가 스크립트

```bash
.venv\Scripts\python scripts/eval_retrieval.py    # 검색 3방식 비교 (LLM 불필요)
.venv\Scripts\python scripts/eval_generation.py   # 종단 검증: 사실 정확·인용·환각 억제·속도
```

## Docker (선택)

```bash
docker compose -f docker/compose.yaml up --build
```

API 컨테이너만 띄우고 LLM은 호스트 Ollama를 씁니다. 이미지는 CPU 전용 torch만 담습니다.
CI(GitHub Actions)가 lint→test→이미지 빌드→GHCR 푸시까지 돌립니다.

## 알려진 한계

- **대화 기억이 없습니다.** 매 질문이 독립적이라 "그래서 그건 어떻게 돼?" 같은 후속
  질문이 안 됩니다. 다음 작업 후보.
- 답변 문구가 실행마다 조금 다릅니다(Ollama GPU 비결정성). 결론은 안정적입니다.
- 표 안의 내용(권역별 신청 가능 지역 등)은 HWP→PDF 공백 소실 때문에 품질이 떨어질 수
  있습니다. 검증 안 된 지대입니다.
- 코퍼스가 공고문 1건뿐입니다. `min_score` 등 임계값은 코퍼스 확장 후 재보정하세요.

## 참고

`legacy/`는 재작성 전 원본입니다(동작 안 함, 기록용). `data/faiss_index/`는 옛
LangChain 피클 인덱스로 새 코드는 읽지 않습니다(새 인덱스는 `data/index/`).
