# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

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


def test_the_public_card_gives_way_to_the_full_card():
    text = ("# T\n\n<!-- public:card begin -->\n<img src=\"public.png\">\n<!-- public:card end -->\n\n"
            "<!-- held-back:card\n<img src=\"full.png\">\nheld-back:card-end -->\n\nbody\n")
    out = release.restore_film(release.remove_public_card(text))
    assert "full.png" in out and "public.png" not in out and "public:card" not in out and "held-back" not in out and out.endswith("body\n")


def test_the_readme_shows_the_public_card_and_holds_the_full_card_back():
    text = Path("README.md").read_text()
    public = text[text.index("<!-- public:card begin -->"):text.index("<!-- public:card end -->")]
    held = text[text.index("<!-- held-back:card"):text.index("held-back:card-end -->")]
    assert "cypmode_card-light.png" in public and "cypmode_card_full" not in public
    assert "cypmode_card_full-light.png" in held and "cypmode_card-light.png" not in held
    ignore = Path(".gitignore").read_text()
    assert "docs/card/cypmode_card_full-light.png" in ignore and "docs/card/interactive/full/" in ignore
    assert "docs/card/cypmode_card-light.png" not in ignore and "docs/card/interactive/\n" not in ignore
    assert "docs/card/cypmode_interactive_preview.gif" in public and Path("docs/card/cypmode_interactive_preview.gif").exists()  # the loop of the public page is part of the public card
    assert "cypmode_interactive_preview" not in ignore and "cypmode_interactive_preview" not in held


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


def test_the_public_edition_of_the_notebook_is_removed_when_the_complete_one_is_released(tmp_path):
    from scripts.release_after_deadline import remove_public_edition

    assert remove_public_edition(tmp_path) is False
    (tmp_path / "notebooks/public").mkdir(parents=True)
    (tmp_path / "notebooks/public/cypmode_notebook_public.html").write_text("x")
    assert remove_public_edition(tmp_path) is True and not (tmp_path / "notebooks/public").exists()
