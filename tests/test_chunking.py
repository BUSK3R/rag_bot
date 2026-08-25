from __future__ import annotations

from pathlib import Path

from ragbot.chunking import chunk_pdf, clean_text, split_text


def test_페이지_번호_머리말이_제거된다():
    assert "- 1 -" not in clean_text("- 1 -\n보도자료")


def test_붙어버린_불릿_목록이_줄로_분리된다():
    # HWP→PDF 추출에서 실제로 나오는 형태
    raw = "□지원대상및규모◦(대상)39세이하◦(지원규모)850명"
    lines = clean_text(raw).splitlines()
    assert len(lines) == 3
    assert lines[0] == "□ 지원대상및규모"


def test_괄호_뒤_공백이_복원된다():
    assert "(대상) 39세" in clean_text("◦(대상)39세이하")


def test_모든_청크가_한도를_넘지_않는다():
    text = "\n\n".join(f"{n}번 문단입니다. " * 20 for n in range(12))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert chunks
    assert all(len(c) <= 300 for c in chunks)


def test_경계가_없는_긴_덩어리도_한도를_지킨다():
    chunks = split_text("가" * 1000, chunk_size=120, overlap=0)
    assert all(len(c) <= 120 for c in chunks)
    assert "".join(chunks) == "가" * 1000


def test_인접_청크가_겹친다():
    text = "\n".join(f"{n}번 줄" for n in range(200))
    chunks = split_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    tail = chunks[0].splitlines()[-1]
    assert tail in chunks[1]


def test_겹침이_없으면_내용이_한_번씩만_나온다():
    text = "\n".join(f"줄{n}" for n in range(100))
    joined = "\n".join(split_text(text, chunk_size=150, overlap=0))
    assert joined.replace("\n", "") == text.replace("\n", "")


def test_실제_공문을_청크로_만든다(sample_pdf: Path):
    chunks = chunk_pdf(sample_pdf, chunk_size=1200, overlap=200)
    assert chunks
    assert all(c.page >= 1 for c in chunks)
    assert all(c.source == sample_pdf.name for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)  # id 는 유일해야 한다
    assert any("청년창업사관학교" in c.text for c in chunks)
