"""설정 단일 출처.

다른 모듈은 os.environ 을 직접 읽지 않고 이 Settings 만 본다. 로컬 / Docker / CI 가
같은 코드를 돌리되 값만 갈아끼울 수 있어야 하기 때문이다.

모든 항목은 RAGBOT_ 접두사 환경변수로 덮어쓸 수 있다. 예: RAGBOT_TOP_K=5
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 패키지가 설치형(site-packages)일 수도, 소스 트리일 수도 있어 저장소 루트를 파일 기준으로 잡는다.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 경로 ---
    data_dir: Path = REPO_ROOT / "data"
    # 새 인덱스는 data/index 에 쓴다. 옛 data/faiss_index 는 LangChain 피클 포맷이라
    # 포맷이 호환되지 않으며, 되돌릴 수 있도록 건드리지 않고 남겨 둔다.
    index_dir: Path = REPO_ROOT / "data" / "index"

    # --- 임베딩 ---
    embed_model: str = "nlpai-lab/KURE-v1"
    # 질의 임베딩은 한 번에 한 문장뿐이라 CPU 로 충분하다. GPU 6GB 는 LLM 에 양보한다.
    embed_device: str = "cpu"
    embed_batch_size: int = 16

    # --- 청킹 ---
    # KURE-v1 의 max_seq_length 는 8192 다(512 가 아니다). 공문 문단이 통째로 들어가도록
    # 넉넉히 잡는다 — 500자로 자르면 조건절이 문장 중간에서 끊긴다.
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- 검색 ---
    top_k: int = Field(default=4, ge=1, le=20)
    # 코사인 유사도 하한. 이보다 낮은 근거만 나오면 LLM 을 부르지 않고 모른다고 답한다.
    # 실측(2026-08-25, KURE-v1): 유관 질문은 0.55~0.75, 무관 질문("비트코인 시세",
    # "넌뭐야")은 0.25~0.32 로 갈린다. 0.30 은 무관 질문을 아슬아슬하게 통과시켜
    # LLM 생성 비용을 낭비했다. 0.45 가 관측 분포의 한가운데다.
    min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    # 프롬프트에 실을 근거의 총 글자수 상한. 응답 지연의 대부분은 생성이 아니라
    # 긴 프롬프트를 읽는 prefill 이라, top_k 보다 이쪽이 속도에 직접적이다.
    # top_k 개를 다 채우기 전에 예산이 차면 거기서 끊는다.
    max_context_chars: int = Field(default=2400, ge=200)
    # 하이브리드 검색에서 밀집(의미) 검색의 가중치. 나머지가 BM25(정확 토큰) 몫이다.
    # 공문처럼 숫자·날짜·법조문이 많은 문서는 BM25 쪽을 키우는 게 유리할 수 있다.
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- LLM (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    # 시제품 단계라 EXAONE(NC 라이선스, 연구 전용)을 기본값으로 쓴다. 1.7GB 라
    # 6GB VRAM 에 100% 적재되어 TTFT 가 Kanana 8B(75% 적재) 대비 4배 빠르다.
    # 정식 제품화 시점에는 Apache 2.0 인 Kanana 로 되돌릴 것:
    #   RAGBOT_LLM_MODEL=hf.co/DevQuasar/kakaocorp.kanana-1.5-8b-instruct-2505-GGUF:Q4_K_M
    llm_model: str = "exaone3.5:2.4b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_timeout_s: float = 120.0

    # --- 서버 ---
    # 컨테이너에서는 패키지가 site-packages 에 설치되므로 REPO_ROOT 추정이 빗나간다.
    # Dockerfile 이 RAGBOT_WEB_DIR=/app/web 로 덮어쓴다.
    web_dir: Path = REPO_ROOT / "web"
    cors_origins: list[str] = Field(default_factory=list)


@lru_cache
def get_settings() -> Settings:
    """프로세스당 한 번만 읽는다. 테스트에서는 get_settings.cache_clear() 로 초기화."""
    return Settings()
