# rag_bot — 공문서 RAG 챗봇

정부 공문 PDF를 읽고, **"몇 페이지에 나온 내용인지"까지 짚어주며** 답하는 챗봇입니다.
전부 내 컴퓨터에서 돌아갑니다 — API 키도, 유료 서비스도 필요 없습니다.

> "청년창업사관학교 지원 대상과 나이 조건은?"
> → "지원 대상은 대표자 연령 39세 이하, 창업 3년 이내(예비 창업자 포함) 기업입니다 **[1]**"
> → **[1] sample_announcement.pdf 1페이지 · 유사도 0.695**

---

# 🚀 세팅 따라하기 (처음부터 끝까지)

Windows 기준이고, 명령은 전부 **복사해서 붙여넣으면** 됩니다.
필요한 것: 디스크 여유 약 10GB, 시간 20~30분 (대부분 다운로드 대기).

## 0단계. 프로그램 2개 설치

**① uv** — 파이썬 설치와 가상환경을 알아서 해주는 도구입니다.
파이썬이 없어도 됩니다. PowerShell을 열고:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 **PowerShell 창을 닫았다 다시 여세요** (PATH 반영).

**② Ollama** — AI 모델을 돌려주는 프로그램입니다.
https://ollama.com/download 에서 Windows용을 받아 설치하세요.
설치하면 알아서 백그라운드에 떠 있습니다 (따로 실행할 필요 없음).

## 1단계. 프로젝트 받기

**git을 써봤다면:**

```powershell
git clone https://github.com/BUSK3R/rag_bot.git
cd rag_bot
```

**git이 처음이라면** 그냥 ZIP으로 받아도 됩니다:
https://github.com/BUSK3R/rag_bot 에서 초록색 **`<> Code` 버튼 → `Download ZIP`** →
압축을 풀고, 그 폴더에서 **Shift+우클릭 → "여기에 PowerShell 창 열기"**.

> 단, ZIP으로 받으면 나중에 팀 변경사항을 받으려면 다시 ZIP을 받아야 합니다.
> 계속 작업할 거라면 git clone을 권합니다. (지금 작업 브랜치는
> `refactor/layered-rag-pipeline` 입니다 — clone 후
> `git checkout refactor/layered-rag-pipeline` 한 번 해주세요.)

## 2단계. 파이썬 환경 만들고 라이브러리 설치

프로젝트 폴더 안에서 (아래 3줄을 차례로):

```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

- 1번째 줄: 이 프로젝트 전용 파이썬 3.11 환경을 만듭니다 (파이썬도 자동으로 받아줍니다)
- 2번째 줄: 그 환경을 켭니다. **프롬프트 앞에 `(rag_bot)` 이 붙으면 성공**
- 3번째 줄: 필요한 라이브러리를 전부 설치합니다 (약 5분, torch 포함 2GB쯤 받습니다)

> ⚠️ 앞으로 PowerShell을 새로 열 때마다 `.venv\Scripts\activate` 를 먼저 해주세요.
> `ragbot` 명령이 안 먹으면 십중팔구 이걸 빼먹은 겁니다.

## 3단계. AI 모델 받기

```powershell
ollama pull exaone3.5:2.4b
```

LG의 한국어 모델 EXAONE(1.6GB)을 받습니다. 몇 분 걸립니다.
`ollama list` 를 쳤을 때 목록에 보이면 성공입니다.

## 4단계. 공문 색인하기

```powershell
ragbot ingest
```

`data/` 폴더의 PDF를 잘게 잘라 검색용 색인을 만듭니다.
**처음 한 번은 임베딩 모델(2.2GB)을 추가로 내려받아서 5~10분 걸립니다.**
마지막에 이렇게 나오면 성공:

```
완료: PDF 1개 → 청크 5개, ...초, 저장 위치 ...\data\index
```

## 5단계. 실행!

```powershell
ragbot serve
```

브라우저에서 **http://127.0.0.1:8000** 을 열면 챗 화면이 뜹니다.
위쪽에 `청크 5개 · LLM 준비됨` 이 보이면 모든 게 정상입니다.

- 첫 질문은 모델을 메모리에 올리느라 15~20초 걸립니다. **그다음부터는 2~3초**입니다.
- 끄려면 PowerShell에서 `Ctrl+C`.

브라우저 없이 터미널에서 바로 물어볼 수도 있습니다:

```powershell
ragbot ask "청년창업사관학교 지원 대상과 나이 조건은?"
```

## 6단계. (선택) 잘 깔렸는지 검사

```powershell
ragbot health        # 색인·모델 상태 점검
python -m pytest -q  # 테스트 55개, 약 10초 (모델 없이도 돕니다)
```

---

## 😵 막혔을 때

| 증상 | 해결 |
| --- | --- |
| `ragbot : 인식할 수 없는 용어` | `.venv\Scripts\activate` 먼저. 프롬프트에 `(rag_bot)` 붙어 있는지 확인 |
| `uv : 인식할 수 없는 용어` | uv 설치 후 PowerShell을 새로 열지 않음 |
| 챗 화면에 `LLM 미준비` | Ollama가 안 떠 있거나 3단계를 건너뜀. `ollama list`로 확인 |
| `FAISS index not found` | 4단계 `ragbot ingest`를 먼저 |
| "모델 불일치" 오류 | 색인을 다른 설정으로 만들었음. `ragbot ingest` 다시 |
| 한글이 `?`로 깨져 보임 | 터미널 문제일 뿐 데이터는 정상. `python -X utf8 -m ragbot.cli ...`로 실행 |
| 답이 이상하거나 느림 | GPU 없는 PC면 느린 게 정상입니다(CPU 생성). 답 품질 문제는 화면 캡처해서 공유해주세요 |

## 📄 새 공문 넣어보기

PDF를 `data/` 폴더에 복사하고 `ragbot ingest` 를 다시 실행하면 끝입니다.
그 내용까지 검색·답변에 바로 반영됩니다.

---

# 🔧 개발자용 정보

여기서부터는 코드를 만질 사람을 위한 내용입니다.

## 동작 흐름

```
질문 → 질의 재작성(구어체→공문 용어) → 하이브리드 검색(BM25 + KURE 임베딩)
     → 근거 부족하면 여기서 반려 → 프롬프트 → LLM(Ollama, 스트리밍) → 인용 달린 답변
```

## 코드 읽는 순서

파일당 40~200줄, 전부 합쳐 1,250줄 남짓입니다.

| 파일 | 역할 |
| --- | --- |
| [`models.py`](src/ragbot/models.py) | 오가는 값 네 개(`Chunk` `Hit` `Answer`). **여기부터 읽으세요.** |
| [`config.py`](src/ragbot/config.py) | 설정 단일 출처. 다른 모듈은 `os.environ`을 직접 읽지 않습니다. |
| [`chunking.py`](src/ragbot/chunking.py) | PDF → 정제 → 청크. 한글 공문의 공백 소실 대응. |
| [`embedder.py`](src/ragbot/embedder.py) | 문장 → 정규화 벡터. torch는 여기서만, 늦게 import합니다. |
| [`store.py`](src/ragbot/store.py) | FAISS 저장/적재/검색. |
| [`retriever.py`](src/ragbot/retriever.py) | **하이브리드 검색.** LangChain을 쓰는 유일한 곳. |
| [`prompt.py`](src/ragbot/prompt.py) | 시스템/재작성 프롬프트와 인용 형식. 순수 함수뿐. |
| [`llm.py`](src/ragbot/llm.py) | LLM 경계(Ollama HTTP). `LLMClient`만 구현하면 백엔드 교체 가능. |
| [`pipeline.py`](src/ragbot/pipeline.py) | 조립 계층. `ask()`가 이 프로젝트의 전부입니다. |
| [`api.py`](src/ragbot/api.py) | HTTP 경계. RAG 로직은 한 줄도 없습니다. |

의존은 `config ← chunking/embedder/store/llm/prompt ← retriever ← pipeline ← api`
한 방향으로만 흐르고, [`tests/test_architecture.py`](tests/test_architecture.py)가 강제합니다.
LangChain은 검색 계층(`retriever.py`/`chunking.py`)에만 존재합니다.

## 설계 결정과 그 실측 근거

전부 실제로 잰 결과로 정한 값들입니다. 코퍼스가 커지면 재보정하세요.

- **하이브리드 검색.** 조회형 질의("제3조제1항", "S-CoP")에서 밀집 임베딩 단독은 8문항 중
  3개만 1등, BM25는 8/8. 하이브리드는 밀집의 조회형 붕괴를 막는 보험입니다.
- **질의 재작성.** "64살인데 청년끼고 지원되나"에 2.4B는 오판, 8B는 회피했지만
  "공동창업"으로 바꿔 물으면 둘 다 정답 → 병목은 용어 매핑. 소형 모델은 지시문을
  무시해서 **퓨샷 패턴**으로 프롬프트를 짰습니다.
- **`min_score` 0.45.** 유관 질문 0.55~0.75 vs 무관 질문 0.25~0.32로 갈립니다.
  게이트에 걸리면 LLM 없이 0.1초에 반려.
- **temperature 0.** 0.1에서는 자격 판정이 4회 중 1회 흔들렸습니다.
- **stop 토큰 명시.** Kanana GGUF에 stop 토큰이 없어 `<|eot_id|>`가 텍스트로 샜습니다.
- **인용 지시는 사용자 프롬프트 말미에.** 시스템 프롬프트에만 두면 2.4B가 5/5 무시.
- **인덱스에 pickle 금지.** 파일 하나로 임의 코드가 실행되는 경로 제거(JSON+FAISS 바이너리).
- **`faiss.write_index` 금지.** 한글 경로에서 죽습니다. 직렬화 후 파이썬이 씁니다.
- **BM25 한국어 토크나이저.** 기본 `text.split()`은 한국어에서 무력. 문자 바이그램으로
  조사·붙임 표기를 흡수합니다(`korean_tokens`).

## API

| 엔드포인트 | 설명 |
| --- | --- |
| `POST /api/ask` | `{"query": "..."}` → 완성된 답변 + 근거 목록 |
| `POST /api/ask/stream` | SSE. 근거를 먼저 보내고 답변을 토큰 단위로 스트리밍 |
| `GET /api/health` | 인덱스 청크 수와 LLM 준비 상태 |

## 설정 바꾸기

`RAGBOT_` 접두사 환경변수 또는 `.env` 파일로 덮어씁니다(`.env.example` 복사해서 쓰세요).

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `RAGBOT_LLM_MODEL` | `exaone3.5:2.4b` | 시제품용(NC 라이선스). 상용화 시 Kanana(Apache 2.0)로: `hf.co/DevQuasar/kakaocorp.kanana-1.5-8b-instruct-2505-GGUF:Q4_K_M` |
| `RAGBOT_REWRITE_QUERY` | `true` | 구어체→격식체 재작성 단계 |
| `RAGBOT_MIN_SCORE` | `0.45` | 최고 유사도가 이보다 낮으면 즉시 반려 |
| `RAGBOT_TOP_K` | `4` | 검색할 청크 수 |
| `RAGBOT_MAX_CONTEXT_CHARS` | `2400` | 근거 총 글자수 상한. 응답 속도에 직결 |
| `RAGBOT_DENSE_WEIGHT` | `0.5` | 하이브리드에서 밀집 비중(나머지 BM25) |
| `RAGBOT_CHUNK_SIZE` | `1200` | 바꾸면 `ragbot ingest` 다시 |
| `RAGBOT_EMBED_DEVICE` | `cpu` | NVIDIA GPU 있으면 `cuda` (VRAM은 LLM과 나눠 씁니다) |

## 평가 스크립트

```powershell
python scripts/eval_retrieval.py    # 검색 3방식 비교 (LLM 불필요)
python scripts/eval_generation.py   # 종단 검증: 사실 정확·인용·환각 억제·속도
```

## Docker (선택)

```powershell
docker compose -f docker/compose.yaml up --build
```

API 컨테이너만 띄우고 LLM은 호스트 Ollama를 씁니다. CI(GitHub Actions)가
lint→test→이미지 빌드→GHCR 푸시까지 돌립니다.

## 알려진 한계

- **대화 기억이 없습니다.** 매 질문이 독립적이라 "그래서 그건?" 같은 후속 질문이 안 됩니다.
- 답변 문구가 실행마다 조금 다릅니다(Ollama GPU 비결정성). 결론은 안정적입니다.
- 표 안의 내용(권역별 신청 지역 등)은 HWP→PDF 공백 소실 때문에 검증 안 된 지대입니다.
- 코퍼스가 공고문 1건뿐입니다. 임계값들은 코퍼스 확장 후 재보정하세요.

## 참고

`legacy/`는 재작성 전 원본입니다(동작 안 함, 기록용). `data/faiss_index/`는 옛 LangChain
피클 인덱스로 새 코드는 읽지 않습니다(새 인덱스는 `data/index/`).
