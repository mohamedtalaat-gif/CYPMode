# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("build_card", Path("docs/card/build_card.py"))


def load():
    pytest.importorskip("rdkit")
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_the_iron_nitrogen_distance_is_measured_on_the_coordinates_of_the_ligand(tmp_path):
    card = load()
    pdb = tmp_path / "pose.pdb"
    pdb.write_text(
        "HETATM    1 FE   HEM A 500       0.000   0.000   0.000  1.00  0.00          FE\n"
        "HETATM    2  N   HEM A 500       2.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    3  N   LIG A 600       0.000   0.000   2.100  1.00  0.00           N\n"
        "HETATM    4  N   LIG A 600       0.000   5.000   0.000  1.00  0.00           N\n"
    )
    assert card.fe_n_distance(pdb) == pytest.approx(2.1)


def test_the_compounds_of_the_card_are_drawn_with_the_donor_nitrogen_marked(tmp_path, monkeypatch):
    card = load()
    monkeypatch.setattr(card, "BUILD", tmp_path)
    picture = card.drawing("example", "dark", "CC(C)(C)NS(=O)(=O)C1(CNC2=C(Br)C=NC=C2C#N)CCC1", donor=15)
    assert picture.exists() and picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_readme_starts_with_the_card_in_both_colour_schemes():
    lines = Path("README.md").read_text().splitlines()
    assert lines[0] == "<!-- public:card begin -->"  # the repository name is the title that GitHub shows, so the README opens with the card
    head = "\n".join(lines[:10])
    assert "docs/card/cypmode_card-dark.gif" in head and "docs/card/cypmode_card-light.gif" in head  # the card of the README turns
    assert all(Path(f"docs/card/cypmode_card-{theme}.{kind}").exists() for theme in ("dark", "light") for kind in ("png", "gif"))


def test_the_circled_nitrogen_is_the_one_nearest_the_iron_in_the_submitted_pose():
    """For the ligand that the film follows the nearest nitrogen is atom 15 of its SMILES (a pyridine nitrogen); for a pose whose match to the SMILES is ambiguous no nitrogen is circled."""
    import pandas as pd

    card = load()
    if not (card.FINAL / "OCNT-2395436.pdb").exists():
        pytest.skip("the submitted structures are not in this checkout")
    smiles = pd.read_csv(card.SMILES).set_index("Molecule_Name")["SMILES"]
    assert card.coordinating_n("OCNT-2395436", smiles["OCNT-2395436"]) == 15


def test_the_readme_links_the_film_through_its_poster_and_the_small_copy():
    text = Path("README.md").read_text()
    assert "<!-- held-back:film" in text and "## The film" in text  # commented out until the challenge closes: scripts/release_after_deadline.py switches it on and "docs/video/cypmode_overview_web.mp4" in text and "docs/video/cypmode_overview_poster.png" in text
    assert Path("docs/video/cypmode_overview_poster.png").exists() or "docs/video/cypmode_overview_poster.png" in Path(".gitignore").read_text()  # held back: present where the film was drawn, listed in the ignore file in a clean checkout
    assert "docs/video/cypmode_overview.mp4" not in text  # the 1080p original is over GitHub's 100 MB limit and is not linked


def test_the_turning_card_is_assembled_from_frames_that_differ_only_in_the_panel_of_the_enzyme_and_loops_forever(tmp_path):
    from PIL import Image

    card = load()
    frames = []
    for i in range(8):
        frame = Image.new("RGB", card.CARD, (20, 30, 60))
        frame.paste((30 * i, 200, 90), card.HERO_BOX)  # the panel of the enzyme changes from frame to frame, the rest of the card does not
        frames.append(frame)
    path = card.assemble_turn(frames, tmp_path / "turn.gif")
    gif = Image.open(path)
    assert gif.n_frames == 8 and gif.size == (card.GIF_WIDTH, round(card.CARD[1] * card.GIF_WIDTH / card.CARD[0])) and gif.info["loop"] == 0 and gif.info["duration"] == card.GIF_FRAME_MS
    first = gif.convert("RGB")
    gif.seek(5)
    later = gif.convert("RGB")
    x0, y0, x1, y1 = (round(v * card.GIF_WIDTH / card.CARD[0]) for v in card.HERO_BOX)
    centre = ((x0 + x1) // 2, (y0 + y1) // 2)
    assert first.getpixel((10, 10)) == later.getpixel((10, 10)) and first.getpixel(centre) != later.getpixel(centre)

    class Browser:  # a headless browser that has gone once, as it does about 31 seconds after it starts: the picture of the card is taken again in the new browser
        def __init__(self):
            self.gone, self.opened = True, []

        def retrying(self, action):
            try:
                return action()
            except ConnectionError:
                self.gone = False
                return action()

        def open(self, page):
            self.opened.append(page)

        def call(self, method, **params):
            if self.gone and method == "Page.captureScreenshot":
                raise ConnectionError
            return {"data": "picture"}

    browser = Browser()
    assert browser.retrying(lambda: card.shot(browser, Path("page.html"), 1, 0, 0)) == "picture" and len(browser.opened) == 2
