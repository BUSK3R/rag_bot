"""HTTP 경계. 파이프라인을 감싸기만 하고 RAG 로직은 한 줄도 두지 않는다."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .llm import LLMError
from .models import Answer, Hit
from .pipeline import RagPipeline
from .prompt import NO_EVIDENCE_ANSWER

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    index: int
    source: str
    page: int
    score: float
    excerpt: str


def _sources(hits: Sequence[Hit]) -> list[SourceOut]:
    return [
        SourceOut(
            index=n,
            source=hit.chunk.source,
            page=hit.chunk.page,
            score=round(hit.score, 4),
            excerpt=hit.chunk.text[:200],
        )
        for n, hit in enumerate(hits, start=1)
    ]


class AskResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    sources: list[SourceOut]

    @classmethod
    def of(cls, answer: Answer) -> AskResponse:
        return cls(
            query=answer.query,
            answer=answer.text,
            grounded=answer.grounded,
            sources=_sources(answer.hits),
        )


def _sse(payload: dict[str, object]) -> str:
    """Server-Sent Events 한 프레임. 줄바꿈이 프레임 구분자라 JSON 은 한 줄로 직렬화한다."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class HealthResponse(BaseModel):
    status: str
    chunks: int
    llm_ready: bool
    detail: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 인덱스가 없거나 모델이 안 받아졌어도 서버는 뜬다. 컨테이너가 부팅 루프를 도는
        # 대신 /api/health 가 원인을 말해주는 편이 진단이 빠르다.
        try:
            app.state.pipeline = RagPipeline.from_settings(settings)
            app.state.error = None
            log.info("파이프라인 적재 완료: 청크 %d개", app.state.pipeline.chunk_count)
        except Exception as exc:  # noqa: BLE001 - 기동 실패 원인을 그대로 노출한다
            app.state.pipeline = None
            app.state.error = str(exc)
            log.error("파이프라인 적재 실패: %s", exc)
        yield

    app = FastAPI(title="공문 RAG 챗봇", version="0.1.0", lifespan=lifespan)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _pipeline() -> RagPipeline:
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail=app.state.error or "파이프라인 미적재")
        return app.state.pipeline

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pipeline: RagPipeline | None = app.state.pipeline
        if pipeline is None:
            return HealthResponse(
                status="degraded", chunks=0, llm_ready=False, detail=app.state.error
            )
        return HealthResponse(
            status="ok", chunks=pipeline.chunk_count, llm_ready=pipeline.llm_ready()
        )

    # async 가 아닌 def 로 둔다. 임베딩(CPU)과 Ollama 호출이 동기 블로킹이라
    # async 로 두면 이벤트 루프를 통째로 붙잡는다. def 면 FastAPI 가 스레드풀로 넘긴다.
    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return AskResponse.of(_pipeline().ask(request.query, request.top_k))

    @app.post("/api/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        """근거를 먼저 한 프레임 보내고, 답변을 토큰 단위로 흘린다.

        체감 지연을 지배하는 건 전체 생성 시간이 아니라 첫 글자까지의 시간이다.
        검색(임베딩+FAISS)은 스트림을 열기 전에 끝내므로 근거 목록이 즉시 뜬다.
        """
        pipeline = _pipeline()
        hits, usable = pipeline.ground(request.query, request.top_k)

        def frames() -> Iterator[str]:
            yield _sse(
                {"type": "sources", "sources": [s.model_dump() for s in _sources(usable or hits)]}
            )
            if not usable:
                yield _sse({"type": "token", "text": NO_EVIDENCE_ANSWER})
                yield _sse({"type": "done", "grounded": False})
                return
            try:
                for piece in pipeline.stream(request.query, usable):
                    yield _sse({"type": "token", "text": piece})
            except LLMError as exc:
                yield _sse({"type": "error", "message": str(exc)})
                return
            yield _sse({"type": "done", "grounded": True})

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            # 프록시가 스트림을 모아뒀다 한꺼번에 내보내면 스트리밍의 의미가 없다.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    web_dir = settings.web_dir
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()
