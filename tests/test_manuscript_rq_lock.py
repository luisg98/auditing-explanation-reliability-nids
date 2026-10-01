from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_central_research_question_is_unchanged() -> None:
    """Keep the agreed research question verbatim across editorial revisions."""
    locked = (ROOT / "latex" / "research_question.txt").read_text(encoding="utf-8").strip()
    introduction = (
        ROOT / "latex" / "sections" / "introduction.tex"
    ).read_text(encoding="utf-8")

    # LaTeX source wraps prose across lines; whitespace normalisation preserves
    # the wording and punctuation while making line wrapping irrelevant.
    normalised_introduction = " ".join(introduction.split())
    assert locked in normalised_introduction
