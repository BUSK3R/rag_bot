# rag_bot

정부 공문 PDF를 근거로 답변하는 로컬 RAG 챗봇. 답변마다 출처 파일과 페이지를 표기한다.
임베딩·LLM 모두 로컬에서 실행하며 외부 API 키를 사용하지 않는다.

```
질문 → 질의 재작성(구어체→공문 용어) → 하이브리드 검색(BM25 + KURE 임베딩)
     → 근거 부족 시 반려 → LLM(Ollama) → 인용 포함 답변
```

## 요구사항

- Windows (다른 OS도 동작하나 아래 명령은 Windows 기준)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — 파이썬 3.11을 자동으로 받아주므로 파이썬 별도 설치 불필요
- [Ollama](https://ollama.com/download) — 설치하면 백그라운드 서비스로 상주
- 디스크 여유 약 10GB (모델 4GB + 패키지 5GB)

## 설치

```powershell
git clone https://github.com/BUSK3R/rag_bot.git
cd rag_bot
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
ollama pull exaone3.5:2.4b
```

- git 없이 받으려면 저장소 페이지의 `Code → Download ZIP`. 이후 갱신을 받으려면 clone 권장.
- 터미널을 새로 열면 `.venv\Scripts\activate`를 다시 실행해야 한다. `ragbot` 명령이 인식되지 않는 경우 대부분 이 문제다.
- `requirements.txt`에 `-e .`가 포함되어 있어 `ragbot` CLI까지 함께 설치된다.

## 실행

```powershell
ragbot ingest   # data/ 의 PDF 색인. 최초 실행 시 임베딩 모델(2.2GB) 다운로드
ragbot serve    # http://127.0.0.1:8000 에 웹 UI
```

첫 질문은 모델 적재 때문에 15~20초, 이후 첫 토큰까지 2~3초.

```powershell
ragbot ask "청년창업사관학교 지원 대상과 나이 조건은?"   # CLI 단발 질의
ragbot health                                        # 색인·LLM 상태 점검
```

새 공문은 PDF를 `data/`에 넣고 `ragbot ingest` 재실행.

## 테스트

```powershell
python -m pytest -q
```

55개, 약 10초. 임베딩 모델·Ollama 없이 동작한다(`tests/conftest.py`의 대역 사용).

## 트러블슈팅

| 증상 | 조치 |
| --- | --- |
| `ragbot: 인식할 수 없는 용어` | `.venv\Scripts\activate` 후 재시도 |
| `uv: 인식할 수 없는 용어` | uv 설치 후 터미널 재시작 |
| UI에 `LLM 미준비` | `ollama list`로 모델 존재 확인, 없으면 `ollama pull exaone3.5:2.4b` |
| `FAISS index not found` | `ragbot ingest` 먼저 실행 |
| 임베딩 모델 불일치 오류 | 설정 변경 후 재색인 필요. `ragbot ingest` |
| 콘솔 한글 깨짐 | 출력 인코딩 문제로 데이터는 정상. `python -X utf8 -m ragbot.cli ...` |

## 설정

`RAGBOT_` 접두사 환경변수 또는 `.env` 파일로 재정의한다(`.env.example` 참고).

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RAGBOT_LLM_MODEL` | `exaone3.5:2.4b` | 시제품용(NC 라이선스). 상용 전환 시 Kanana(Apache 2.0): `hf.co/DevQuasar/kakaocorp.kanana-1.5-8b-instruct-2505-GGUF:Q4_K_M` |
| `RAGBOT_REWRITE_QUERY` | `true` | 구어체→격식체 질의 재작성 |
| `RAGBOT_MIN_SCORE` | `0.45` | 최고 유사도가 미달이면 생성 없이 반려 |
| `RAGBOT_TOP_K` | `4` | 검색 청크 수 |
| `RAGBOT_MAX_CONTEXT_CHARS` | `2400` | 근거 총 글자수 상한. 응답 속도에 직결 |
| `RAGBOT_DENSE_WEIGHT` | `0.5` | 하이브리드 중 밀집 검색 비중 |
| `RAGBOT_CHUNK_SIZE` | `1200` | 변경 시 재색인 필요 |
| `RAGBOT_EMBED_DEVICE` | `cpu` | NVIDIA GPU 사용 시 `cuda` |

## API

| 엔드포인트 | 설명 |
| --- | --- |
| `POST /api/ask` | `{"query": "..."}` → 답변 + 근거 목록 |
| `POST /api/ask/stream` | SSE. 근거 프레임 후 토큰 단위 스트리밍 |
| `GET /api/health` | 청크 수, LLM 준비 상태 |

## 프로젝트 구조

의존 방향: `config ← chunking/embedder/store/llm/prompt ← retriever ← pipeline ← api`.
`tests/test_architecture.py`가 이 경계를 검사한다. LangChain은 `retriever.py`,
`chunking.py`에서만 사용한다.

| 파일 | 역할 |
| --- | --- |
| `models.py` | 공용 타입 `Chunk` `Hit` `Answer` |
| `config.py` | 설정 단일 출처 |
| `chunking.py` | PDF 추출·정제·분할. 한글 공문 공백 소실 보정 포함 |
| `embedder.py` | KURE-v1 임베딩. torch 지연 import |
| `store.py` | FAISS 저장/적재/검색 |
| `retriever.py` | BM25 + 밀집 앙상블 검색 |
| `prompt.py` | 시스템/재작성 프롬프트 |
| `llm.py` | Ollama HTTP 클라이언트 |
| `pipeline.py` | 조립 계층. `ask()` |
| `api.py` | FastAPI 라우팅 |

## 설계 노트

측정에 근거해 정한 값들이다. 코퍼스 확장 시 재보정이 필요하다.

- 하이브리드 검색: 조회형 질의("제3조제1항" 등)에서 밀집 단독 3/8, BM25 8/8.
  하이브리드는 밀집의 조회형 실패를 보완한다. 재현: `scripts/eval_retrieval.py`
- 질의 재작성: "청년끼고" 류 구어체에서 2.4B는 오판, 8B는 회피했으나 "공동창업"으로
  바꾸면 양쪽 모두 정답. 소형 모델은 지시문 프롬프트를 무시해 퓨샷 형식을 사용
- `min_score` 0.45: 유관 질의 0.55~0.75, 무관 질의 0.25~0.32로 분포가 갈림
- temperature 0: 0.1에서 자격 판정 답변이 4회 중 1회 흔들림
- stop 토큰 명시: Kanana GGUF에 stop 토큰이 없어 `<|eot_id|>`가 본문으로 유출
- 인용 지시는 사용자 프롬프트 말미에 예시와 함께 배치(시스템 프롬프트만으로는 미준수)
- 인덱스는 JSON + FAISS 바이너리. pickle 및 `faiss.write_index`(비ASCII 경로에서
  실패)를 사용하지 않음
- BM25 토크나이저: 기본 `text.split()`은 한국어에 부적합. 한글 어절에 문자 바이그램 추가

## 평가

```powershell
python scripts/eval_retrieval.py    # 검색 방식 비교 (LLM 불필요)
python scripts/eval_generation.py   # 종단: 사실 정확도·인용·환각 억제·지연
```

## Docker

```powershell
docker compose -f docker/compose.yaml up --build
```

API 컨테이너만 실행하고 LLM은 호스트 Ollama를 사용한다. CI는 lint → test →
이미지 빌드 → GHCR 푸시 순서로 동작한다.

## 알려진 한계

- 대화 이력 없음. 매 질문이 독립적이라 후속 질문("그래서 그건?")이 이어지지 않는다
- 답변 문구가 실행마다 조금씩 다름(Ollama GPU 비결정성). 결론은 안정적
- 표 내부 내용(권역별 신청 지역 등)은 HWP→PDF 공백 소실로 미검증
- 코퍼스가 공고문 1건. 각종 임계값은 이 기준으로 보정된 값

`legacy/`는 재작성 전 코드(비동작, 보존용). `data/faiss_index/`는 구 포맷 인덱스로
현재 코드는 `data/index/`를 사용한다.
