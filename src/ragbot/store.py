"""FAISS 인덱스 + 청크 본문의 저장/적재/검색.

두 가지를 이전 구조에서 바꿨다.

1. IndexFlatL2 → IndexFlatIP. 벡터가 정규화돼 있으므로 내적이 곧 코사인 유사도이고,
   "클수록 유사"라는 직관과 화면 표기가 일치한다. L2 거리를 유사도라 부르던 혼선이 사라진다.
2. pickle → JSON. LangChain 포맷은 적재할 때 allow_dangerous_deserialization=True 가
   필요했다. 인덱스 파일 하나로 임의 코드가 실행된다는 뜻이라, 배포할 물건에는 둘 수 없다.

인덱스 입출력에 faiss.write_index/read_index 를 쓰지 않는 이유는 아래 _dump/_load 참고.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from .models import Chunk, Hit

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"
_META_FILE = "meta.json"


def _dump_index(index: faiss.Index, path: Path) -> None:
    """faiss.write_index 대신 직렬화 후 파이썬이 파일을 쓴다.

    write_index/read_index 는 C 레벨에서 좁은(char*) 경로로 fopen 하므로, Windows 에서
    경로에 한글이 하나라도 있으면 "Illegal byte sequence" 로 죽는다. 사용자 이름이나
    상위 폴더가 한글이면 앱이 통째로 못 뜬다는 뜻이라 우회한다.
    """
    path.write_bytes(faiss.serialize_index(index).tobytes())


def _load_index(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8).copy())


class IndexMismatchError(RuntimeError):
    """인덱스를 만든 조건과 지금 설정이 어긋날 때. 조용히 헛검색하는 것보다 낫다."""


@dataclass(frozen=True, slots=True)
class IndexMeta:
    embed_model: str
    dimension: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    created_at: str

    @staticmethod
    def now(**kwargs: object) -> IndexMeta:
        return IndexMeta(created_at=datetime.now(UTC).isoformat(timespec="seconds"), **kwargs)  # type: ignore[arg-type]


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], meta: IndexMeta) -> None:
        self._index = index
        self._chunks = chunks
        self._row_of = {chunk.id: row for row, chunk in enumerate(chunks)}
        self.meta = meta

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """원문 전체가 필요한 검색기(BM25 등)를 위해 노출한다."""
        return self._chunks

    def scores_for(self, query_vector: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]:
        """지정한 청크들의 코사인 유사도.

        하이브리드 검색에서는 BM25 가 찾아온 문서도 결과에 섞인다. 그런 문서에도
        화면에 보일 점수와 생성 여부를 가르는 판단 근거가 필요한데, IndexFlat 은
        원본 벡터를 그대로 들고 있어 reconstruct 로 정확한 값을 되찾을 수 있다.
        """
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(-1)
        scores: dict[str, float] = {}
        for chunk_id in chunk_ids:
            row = self._row_of.get(chunk_id)
            if row is not None:
                scores[chunk_id] = float(np.dot(query, self._index.reconstruct(row)))
        return scores

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, meta: IndexMeta) -> VectorStore:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"청크 {len(chunks)}개와 벡터 {vectors.shape[0]}개가 맞지 않습니다")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, meta)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _dump_index(self._index, directory / _INDEX_FILE)
        (directory / _CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        (directory / _META_FILE).write_text(
            json.dumps(asdict(self.meta), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, *, expect_model: str | None = None) -> VectorStore:
        if not (directory / _INDEX_FILE).exists():
            raise FileNotFoundError(
                f"인덱스가 없습니다: {directory}. 먼저 `ragbot ingest` 를 실행하세요."
            )
        meta = IndexMeta(**json.loads((directory / _META_FILE).read_text(encoding="utf-8")))
        if expect_model and meta.embed_model != expect_model:
            raise IndexMismatchError(
                f"인덱스는 {meta.embed_model} 로 만들어졌는데 지금 설정은 {expect_model} 입니다. "
                "다시 ingest 하세요."
            )
        chunks = [
            Chunk(**item)
            for item in json.loads((directory / _CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        return cls(_load_index(directory / _INDEX_FILE), chunks, meta)

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        """상위 k건을 유사도 내림차순으로. query_vector 는 (dim,) 또는 (1, dim)."""
        if not self._chunks:
            return []
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, min(k, len(self._chunks)))
        return [
            Hit(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], indices[0], strict=True)
            if i != -1
        ]
