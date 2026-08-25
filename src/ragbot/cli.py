"""명령줄 진입점: ingest / ask / serve / health."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import get_settings
from .embedder import Embedder
from .pipeline import RagPipeline, ingest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragbot", description="공문서 RAG 챗봇")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="PDF를 색인한다")
    p_ingest.add_argument("path", nargs="?", type=Path, help="PDF 파일 또는 디렉터리")

    p_ask = sub.add_parser("ask", help="한 번 질문한다")
    p_ask.add_argument("query")
    p_ask.add_argument("--top-k", type=int, default=None)

    p_serve = sub.add_parser("serve", help="API 서버를 띄운다")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    sub.add_parser("health", help="인덱스와 LLM 상태를 점검한다")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows 콘솔 기본 코드페이지(cp949)로는 한글 출력이 UnicodeEncodeError 로 죽는다.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = _build_parser().parse_args(argv)
    settings = get_settings()

    if args.command == "ingest":
        target = args.path or settings.data_dir
        embedder = Embedder(
            settings.embed_model,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
        report = ingest(target, embedder, settings)
        print(
            f"완료: PDF {report.files}개 → 청크 {report.chunks}개, "
            f"{report.seconds:.1f}초, 저장 위치 {report.index_dir}"
        )
        return 0

    if args.command == "health":
        try:
            pipeline = RagPipeline.from_settings(settings)
        except Exception as exc:  # noqa: BLE001
            print(f"인덱스 이상: {exc}")
            return 1
        print(f"인덱스 청크 {pipeline.chunk_count}개")
        print(f"LLM({settings.llm_model}) 준비됨: {pipeline.llm_ready()}")
        return 0

    if args.command == "ask":
        pipeline = RagPipeline.from_settings(settings)
        answer = pipeline.ask(args.query, args.top_k)
        print(f"\n{answer.text}\n")
        if answer.hits:
            print("근거:")
            for n, hit in enumerate(answer.hits, start=1):
                print(f"  [{n}] {hit.chunk.source} {hit.chunk.page}p (유사도 {hit.score:.3f})")
        return 0 if answer.grounded else 2

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "ragbot.api:app", host=args.host, port=args.port, reload=args.reload, factory=False
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
