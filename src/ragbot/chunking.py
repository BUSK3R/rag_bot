"""PDF → 정제 → 청크.

분할 자체는 LangChain 의 RecursiveCharacterTextSplitter 에 맡긴다. 직접 짠 분할기보다
경계 처리 예외가 잘 다져져 있고, 소형 모델일수록 청크 경계가 답변 품질을 좌우한다.

정제(clean_text)는 우리 몫이다. 한글 공문(HWP → PDF)은 텍스트 추출 시 공백이 통째로
사라지는데, LangChain 에 이걸 다루는 도구는 없다. 실제 원본에서 뽑은 예:

    □지원대상및규모◦(대상)39세이하및창업3년이내창업자(예비)

단어 사이 띄어쓰기 복원은 별도 모델(PyKoSpacing 등)이 필요해 의존성 대비 이득이
낮다고 보고 유보했다. 대신 불릿·괄호 경계만 되살려도 문단 구조가 돌아오고,
그 경계가 곧 분할 지점이 되므로 검색 품질에 실제로 기여한다.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .models import Chunk

# 앞에서부터 시도하고 안 되면 다음으로 내려간다. 마지막 ""는 강제 절단이라,
# 공백이 소실돼 경계가 아예 없는 표 덩어리도 한도를 넘지 않는다.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# 공문 불릿. 이 앞에서 줄을 나누면 붙어버린 목록이 다시 항목별로 분리된다.
_BULLETS = "◦□▪‣※●○·"
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮➀➁➂➃➄➅➆➇➈➉"

# 페이지 표기 "- 2 -". 앞뒤 공백을 요구하지 않으면 044-204-7950 의 "-204-" 나
# 2025-01-22 의 "-01-" 까지 삼켜 숫자를 찢어놓는다(실제로 전화번호가 깨졌다).
_PAGE_MARKER = re.compile(r"(?:^|(?<=\s))-\s*\d{1,3}\s*-(?=\s|$)", re.MULTILINE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_BEFORE_BULLET = re.compile(rf"(?<!\n)(?=[{_BULLETS}{_CIRCLED}])")
_AFTER_BULLET = re.compile(rf"(?<=[{_BULLETS}])(?=\S)")
_AFTER_PAREN = re.compile(r"(?<=\))(?=[0-9가-힣])")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """추출 직후의 날텍스트를 인덱싱 가능한 형태로 정돈한다."""
    text = _CONTROL.sub("", raw)
    text = _PAGE_MARKER.sub("\n", text)  # 머리말/꼬리말의 "- 1 -"
    text = _BEFORE_BULLET.sub("\n", text)  # 붙어버린 목록을 항목별로
    text = _AFTER_BULLET.sub(" ", text)  # "□지원목적" → "□ 지원목적"
    text = _AFTER_PAREN.sub(" ", text)  # "(대상)39세" → "(대상) 39세"
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def make_splitter(*, chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_SEPARATORS,
        keep_separator=False,
    )


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    return make_splitter(chunk_size=chunk_size, overlap=overlap).split_text(text)


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """(1-based 페이지 번호, 정제된 텍스트) 목록. 빈 페이지는 건너뛴다."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def chunk_pdf(path: Path, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """PDF 한 개를 Chunk 목록으로. 페이지 경계는 넘지 않는다(인용 페이지가 흐려지므로)."""
    splitter = make_splitter(chunk_size=chunk_size, overlap=overlap)
    source = path.name
    chunks: list[Chunk] = []
    for page_number, text in load_pdf_pages(path):
        for index, piece in enumerate(splitter.split_text(text)):
            chunks.append(
                Chunk(
                    id=f"{source}#p{page_number}#{index}",
                    text=piece,
                    source=source,
                    page=page_number,
                )
            )
    return chunks
