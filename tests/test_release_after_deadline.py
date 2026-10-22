import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("release", Path("scripts/release_after_deadline.py"))
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_the_held_back_block_is_removed_and_everything_else_stays():
    text = "a\n\n# >>> HELD BACK UNTIL AFTER THE DEADLINE (x)\nsubmissions/\n# <<< HELD BACK\n\nb\n"
    out = release.strip_held_back(text)
    assert "submissions/" not in out and out.startswith("a") and out.rstrip().endswith("b")


def test_the_commented_card_is_switched_on_with_the_film_section():
    text = "<!-- held-back:card\n<img src=\"card.png\">\nheld-back:card-end -->\ntext\n<!-- held-back:film\n## The film\nheld-back:film-end -->\n"
    out = release.restore_film(text)
    assert "card.png" in out and "## The film" in out and "held-back" not in out and "-->" not in out


def test_the_commented_film_section_is_switched_on():
    text = "x\n<!-- held-back:film\n## The film\nsome text\nheld-back:film-end -->\n\ny\n"
    out = release.restore_film(text)
    assert "## The film" in out and "held-back" not in out and "-->" not in out


def test_the_repository_holds_its_results_back_until_the_deadline():
    ignore = Path(".gitignore").read_text()
    assert "HELD BACK UNTIL AFTER THE DEADLINE" in ignore and "submissions/tdi_track/*.csv" in ignore and "data/tdi_track/outputs/final/" in ignore
    assert "<!-- held-back:film" in Path("README.md").read_text() and "<!-- held-back:card" in Path("README.md").read_text()


def test_nothing_that_names_blinded_results_would_be_pushed():
    from cypmode.audit import public_readiness as pr

    names = pr.blinded_names()
    if not names:
        return  # a checkout without the organisers' inputs has nothing to compare with
    assert pr.prediction_findings(pr.candidate_files(), names) == []
