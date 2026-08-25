"""조립 계층. 아래 모듈들을 엮기만 하고, 자체 로직은 최소로 둔다.

ask() 의 흐름이 곧 이 프로젝트의 전부다:

    질문 → 하이브리드 검색 → 근거 부족하면 여기서 반려 → 프롬프트 → LLM → 인용 달린 답변
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_pdf
from .config import Settings
from .embedder import Embedder, SupportsEncode
from .llm import LLMClient, LLMError, OllamaClient
from .models import Answer, Chunk, Hit
from .prompt import (
    NO_EVIDENCE_ANSWER,
    REWRITE_SYSTEM,
    SYSTEM_PROMPT,
    build_rewrite_prompt,
    build_user_prompt,
)
from .retriever import HybridSearcher
from .store import IndexMeta, VectorStore

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestReport:
    files: int
    chunks: int
    index_dir: Path
    seconds: float


def find_pdfs(target: Path) -> list[Path]:
    """파일이면 그 하나, 디렉터리면 그 안의 PDF 전부(정렬해 재현 가능하게)."""
    if target.is_file():
        return [target]
    return sorted(p for p in target.glob("*.pdf") if p.is_file())


def ingest(target: Path, embedder: SupportsEncode, settings: Settings) -> IngestReport:
    """PDF(들) → 청크 → 벡터 → 인덱스 저장."""
    started = time.perf_counter()
    pdfs = find_pdfs(target)
    if not pdfs:
        raise FileNotFoundError(f"PDF를 찾지 못했습니다: {target}")

    chunks: list[Chunk] = []
    for pdf in pdfs:
        found = chunk_pdf(pdf, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        log.info("%s → 청크 %d개", pdf.name, len(found))
        chunks.extend(found)
    if not chunks:
        raise ValueError("추출된 텍스트가 없습니다. 스캔 PDF라면 OCR이 필요합니다.")

    vectors = embedder.encode([c.text for c in chunks])
    store = VectorStore.build(
        chunks,
        vectors,
        IndexMeta.now(
            embed_model=settings.embed_model,
            dimension=int(vectors.shape[1]),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_count=len(chunks),
        ),
    )
    store.save(settings.index_dir)
    return IngestReport(
        files=len(pdfs),
        chunks=len(chunks),
        index_dir=settings.index_dir,
        seconds=time.perf_counter() - started,
    )


class RagPipeline:
    def __init__(
        self,
        searcher: HybridSearcher,
        llm: LLMClient,
        *,
        top_k: int,
        min_score: float,
        max_context_chars: int = 2400,
        rewrite_query: bool = False,
    ) -> None:
        self._searcher = searcher
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score
        self._max_context_chars = max_context_chars
        self._rewrite_query = rewrite_query

    @classmethod
    def from_settings(cls, settings: Settings) -> RagPipeline:
        """조립 지점(composition root). 실제 구현체를 고르는 곳은 여기 한 군데뿐이다."""
        embedder = Embedder(
            settings.embed_model,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
        store = VectorStore.load(settings.index_dir, expect_model=settings.embed_model)
        return cls(
            HybridSearcher(store, embedder, k=settings.top_k, dense_weight=settings.dense_weight),
            OllamaClient(
                settings.ollama_base_url,
                settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout_s=settings.llm_timeout_s,
            ),
            top_k=settings.top_k,
            min_score=settings.min_score,
            max_context_chars=settings.max_context_chars,
            rewrite_query=settings.rewrite_query,
        )

    @property
    def chunk_count(self) -> int:
        return self._searcher.chunk_count

    def llm_ready(self) -> bool:
        return self._llm.health()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        return self._searcher.search(query, top_k or self._top_k)

    def _select(self, hits: Sequence[Hit]) -> tuple[Hit, ...]:
        """생성에 쓸 근거를 고른다.

        임계값은 개별 청크가 아니라 '질문이 이 문서와 상관은 있는가' 에 건다. 하이브리드에서
        BM25 가 찾아온 문서는 밀집 점수가 낮아도 정확 토큰이 일치해 올라온 것이라, 개별로
        자르면 하이브리드를 쓰는 의미가 없어진다. 대신 최고 점수가 임계값에 못 미치면
        질문 자체가 코퍼스와 무관하다고 보고 통째로 반려한다.

        통과한 뒤에는 앙상블 순위대로 글자수 예산이 찰 때까지만 담는다. 1등은 예산을
        넘겨도 무조건 넣는다 — 안 그러면 긴 청크 하나뿐일 때 근거가 통째로 비어버린다.
        """
        if not hits or max(hit.score for hit in hits) < self._min_score:
            return ()
        picked: list[Hit] = []
        used = 0
        for hit in hits:
            cost = len(hit.chunk.text)
            if picked and used + cost > self._max_context_chars:
                break
            picked.append(hit)
            used += cost
        return tuple(picked)

    def rewrite(self, query: str) -> str:
        """구어체 질문을 공문 용어로 정규화한다. 실패하면 원문을 그대로 쓴다.

        실측: "청년끼고 지원되나"는 2.4B 가 오판(지원 불가로 단정), 8B 가 회피했지만,
        "공동창업"으로 바꿔 물으면 둘 다 정답이었다. 병목이 모델 체급이 아니라 용어
        매핑이므로 검색·생성 전에 한 번 옮긴다. 재작성 결과가 원문의 3배를 넘거나
        여러 줄이면 모델이 설명을 붙인 것으로 보고 버린다.
        """
        if not self._rewrite_query:
            return query
        try:
            candidate = self._llm.complete(REWRITE_SYSTEM, build_rewrite_prompt(query))
        except LLMError:
            return query
        candidate = candidate.strip().strip('"').strip()
        # 퓨샷 패턴을 그대로 이어 써서 "격식체:" 라벨까지 출력하는 경우가 실측에서 나왔다.
        candidate = candidate.removeprefix("격식체:").strip()
        if not candidate or "\n" in candidate or len(candidate) > max(len(query) * 3, 120):
            return query
        return candidate

    def ground(
        self, query: str, top_k: int | None = None
    ) -> tuple[tuple[Hit, ...], tuple[Hit, ...]]:
        """(검색된 전체, 근거로 실제 쓸 것). 스트리밍 API 가 근거를 먼저 내보낼 수 있게 분리.

        재작성은 하지 않는다 — 호출자가 rewrite() 결과를 넘겨야 검색과 생성이 같은
        질문을 쓴다는 것이 코드에 드러난다(api 의 스트리밍 경로가 이 규약에 의존한다).
        """
        hits = tuple(self.retrieve(query, top_k))
        return hits, self._select(hits)

    def ask(self, query: str, top_k: int | None = None) -> Answer:
        effective = self.rewrite(query)
        hits, usable = self.ground(effective, top_k)
        if not usable:
            # 근거가 약하면 LLM 을 아예 부르지 않는다. 환각을 사후에 걸러내는 것보다
            # 생성 경로에 진입하지 않는 편이 확실하고 응답도 빠르다.
            return Answer(query=query, text=NO_EVIDENCE_ANSWER, hits=hits, grounded=False)
        text = self._llm.complete(SYSTEM_PROMPT, build_user_prompt(effective, usable))
        return Answer(query=query, text=text, hits=usable, grounded=True)

    def stream(self, query: str, usable: Sequence[Hit]) -> Iterator[str]:
        """근거가 정해진 뒤 답변 조각을 흘려보낸다. 근거 선별은 ground() 가 이미 했다."""
        return self._llm.stream(SYSTEM_PROMPT, build_user_prompt(query, usable))
