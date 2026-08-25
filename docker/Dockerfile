# syntax=docker/dockerfile:1

# ── 빌드 ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch 를 CPU 휠로 먼저 못박는다. 그냥 두면 sentence-transformers 가 CUDA 휠(2.5GB+)을
# 끌고 와 이미지가 5GB 를 넘는다. 임베딩은 CPU 에서만 돌리므로 CUDA 는 필요 없다.
# LLM 은 별도 Ollama 컨테이너가 GPU 를 쓴다.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ── 실행 ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    RAGBOT_DATA_DIR=/data \
    RAGBOT_INDEX_DIR=/data/index \
    RAGBOT_WEB_DIR=/app/web

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY web ./web

# 비루트로 돌린다. /data 와 /cache 는 볼륨이 붙는 자리라 미리 소유권을 넘겨둔다.
RUN useradd --create-home --uid 10001 ragbot \
 && mkdir -p /data /cache \
 && chown -R ragbot:ragbot /data /cache /app
USER ragbot

EXPOSE 8000

# 인덱스가 없으면 /api/health 가 degraded 를 돌려주므로 컨테이너가 unhealthy 로 표시된다.
# 임베딩 모델 최초 내려받기가 오래 걸려 start-period 를 넉넉히 잡았다.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import sys,httpx; sys.exit(0 if httpx.get('http://127.0.0.1:8000/api/health',timeout=5).json()['status']=='ok' else 1)"

CMD ["uvicorn", "ragbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
