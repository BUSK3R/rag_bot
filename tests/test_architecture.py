"""계층 경계를 코드로 지킨다. 주석에 적어둔 규칙은 언젠가 어긋난다."""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ragbot"
# LangChain 을 써도 되는 파일. 검색 계층과 분할기뿐이다.
LANGCHAIN_ALLOWED = {"retriever.py", "chunking.py"}


def _modules() -> list[Path]:
    return sorted(p for p in SRC.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_langchain은_검색_계층_밖으로_새지_않는다(module: Path):
    if module.name in LANGCHAIN_ALLOWED:
        return
    source = module.read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "langchain" in line
    ]
    assert not offenders, f"{module.name} 이 LangChain 을 직접 import 합니다: {offenders}"


def test_하위_계층은_상위_계층을_모른다():
    """config ← chunking/embedder/store/llm/prompt ← retriever ← pipeline ← api"""
    상위 = {
        "config.py": {
            "chunking",
            "embedder",
            "store",
            "llm",
            "prompt",
            "retriever",
            "pipeline",
            "api",
        },
        "models.py": {
            "config",
            "chunking",
            "embedder",
            "store",
            "llm",
            "prompt",
            "retriever",
            "pipeline",
            "api",
        },
        "prompt.py": {"chunking", "embedder", "store", "llm", "retriever", "pipeline", "api"},
        "store.py": {"retriever", "pipeline", "api", "llm"},
        "llm.py": {"store", "retriever", "pipeline", "api"},
        "retriever.py": {"pipeline", "api"},
        "pipeline.py": {"api"},
    }
    for name, 금지 in 상위.items():
        source = (SRC / name).read_text(encoding="utf-8")
        for 모듈 in 금지:
            assert f"from .{모듈} import" not in source, f"{name} 이 상위 계층 {모듈} 을 참조합니다"


def test_torch는_지연_import되어_모듈_임포트만으로_딸려오지_않는다():
    # 이게 깨지면 테스트와 CI 가 2GB 스택 없이는 못 돈다.
    source = (SRC / "embedder.py").read_text(encoding="utf-8")
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("sentence_transformers" in line or "torch" in line for line in top_level)
