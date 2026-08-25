"""검색 계층. LangChain 은 이 파일 안에서만 쓴다.

왜 하이브리드인가 — 행정공문은 조건이 정확한 토큰으로 박혀 있다.
"39세 이하", "1억원", "2월 12일(수) 16시", "제3조제1항" 같은 것들이다.
밀집 임베딩(KURE)은 이걸 "나이 관련", "금액 관련" 정도로 뭉개서, 정작 숫자가 다른
문단을 1등으로 올려 보내기도 한다. BM25 는 정확히 일치하는 토큰을 잡는다.
둘을 섞으면 의미 검색의 유연함과 정확 일치의 신뢰성을 같이 가져간다.

앙상블(RRF 융합)은 직접 짜지 않고 LangChain 의 EnsembleRetriever 를 쓴다.
순위 융합은 경계 조건에서 틀리기 쉬운 부분이라 검증된 구현이 낫다.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .embedder import SupportsEncode
from .models import Chunk, Hit
from .store import VectorStore

_NON_WORD = re.compile(r"[^0-9A-Za-z가-힣]+")
_HAS_HANGUL = re.compile(r"[가-힣]")


def korean_tokens(text: str) -> list[str]:
    """BM25 용 한국어 토크나이저.

    BM25Retriever 의 기본 전처리는 `text.split()` 이라 한국어에서 거의 동작하지 않는다.
    조사가 붙어 "지원대상은" 과 "지원대상" 이 다른 토큰이 되고, 공백이 소실된 공문에서는
    "39세이하및창업3년이내" 가 통째로 토큰 하나가 된다.

    형태소 분석기(kiwipiepy 등)를 쓰면 정확하지만 의존성이 커서, 한글 어절에 한해
    문자 바이그램을 함께 넣어 조사와 붙임 표기를 흡수한다. 숫자·영문은 그대로 두어야
    "39", "16", "CES" 같은 정확 일치가 살아난다.
    """
    tokens: list[str] = []
    for word in _NON_WORD.sub(" ", text.lower()).split():
        tokens.append(word)
        if len(word) > 2 and _HAS_HANGUL.search(word):
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={"id": chunk.id, "source": chunk.source, "page": chunk.page},
    )


def to_chunk(document: Document) -> Chunk:
    meta = document.metadata
    return Chunk(
        id=meta["id"], text=document.page_content, source=meta["source"], page=meta["page"]
    )


class DenseRetriever(BaseRetriever):
    """FAISS 밀집 검색을 LangChain 리트리버로 감싼다.

    임베더가 아니라 인코딩 함수를 받는다 — HybridSearcher 가 질의 벡터를 한 번만
    계산해 재사용하려면 호출 지점이 바깥에 있어야 한다.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: VectorStore
    encode_query: Any
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            to_document(hit.chunk) for hit in self.store.search(self.encode_query(query), self.k)
        ]


class HybridSearcher:
    """BM25 + 밀집 앙상블. 파이프라인은 이 클래스만 알고 LangChain 은 모른다."""

    def __init__(
        self,
        store: VectorStore,
        embedder: SupportsEncode,
        *,
        k: int = 4,
        dense_weight: float = 0.5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k
        self._cached_query: str | None = None
        self._cached_vector: np.ndarray | None = None

        self._dense = DenseRetriever(store=store, encode_query=self._encode, k=k)
        documents = [to_document(chunk) for chunk in store.chunks]
        if documents:
            self._bm25 = BM25Retriever.from_documents(documents, preprocess_func=korean_tokens)
            self._bm25.k = k
            # id_key 를 주면 같은 청크를 두 검색기가 찾아왔을 때 본문 해시가 아니라
            # 청크 id 로 중복을 판정한다.
            self._retriever: BaseRetriever = EnsembleRetriever(
                retrievers=[self._bm25, self._dense],
                weights=[1.0 - dense_weight, dense_weight],
                id_key="id",
            )
        else:
            self._bm25 = None
            self._retriever = self._dense

    def _encode(self, query: str) -> np.ndarray:
        """질의 임베딩은 한 번만. 밀집 검색과 점수 재계산이 같은 벡터를 쓴다."""
        if query != self._cached_query:
            self._cached_query = query
            self._cached_vector = self._embedder.encode([query])[0]
        return self._cached_vector  # type: ignore[return-value]

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        limit = k or self._k
        self._dense.k = limit
        if self._bm25 is not None:
            self._bm25.k = limit

        documents = self._retriever.invoke(query)[:limit]
        chunks = [to_chunk(document) for document in documents]
        # 앙상블 순위는 RRF 값이라 화면에 띄우기엔 의미가 흐리다. 실제 코사인 유사도를
        # 되찾아 붙여, BM25 가 찾아온 문서에도 해석 가능한 점수가 달리게 한다.
        scores = self._store.scores_for(self._encode(query), [chunk.id for chunk in chunks])
        return [Hit(chunk=chunk, score=scores.get(chunk.id, 0.0)) for chunk in chunks]
