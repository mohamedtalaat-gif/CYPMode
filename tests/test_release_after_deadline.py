# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import importlib.util
import json
import re
from pathlib import Path

import yaml

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
    assert "cypmode_card-light.gif" in public and "cypmode_card_full" not in public  # the card that turns is the picture of the README
    assert "cypmode_card_full-light.gif" in held and "cypmode_card-light.gif" not in held
    ignore = Path(".gitignore").read_text()
    assert "docs/card/cypmode_card_full-light.png" in ignore and "docs/card/interactive/full/" in ignore
    assert "docs/card/cypmode_card-light.png" not in ignore and "docs/card/interactive/\n" not in ignore
    assert Path("docs/card/cypmode_card-light.gif").exists() and Path("docs/card/cypmode_card-dark.gif").exists()  # the turning card of the public edition is in the repository
    assert "docs/card/cypmode_card-light.gif" not in ignore.splitlines() and "docs/card/cypmode_card_full-light.gif" in ignore.splitlines() and "docs/card/cypmode_card_full-dark.gif" in ignore.splitlines()


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


def test_each_edition_of_the_readme_shows_its_card_turning():
    text = Path("README.md").read_text()
    shown_now = re.sub(r"<!-- held-back:card.*?held-back:card-end -->", "", re.sub(r"<!-- held-back:film.*?held-back:film-end -->", "", text, flags=re.DOTALL), flags=re.DOTALL)
    complete = release.restore_film(release.remove_public_card(text))
    assert "docs/card/cypmode_card-light.gif" in shown_now and "cypmode_card_full" not in shown_now  # the public edition: the card of the public structures, turning
    assert "docs/card/cypmode_card_full-light.gif" in complete and "docs/card/cypmode_card-light.gif" not in complete  # the complete edition: the card of the submitted structures, drawn at the release
    assert "docs/card/cypmode_card_full-light.gif" in Path(".gitignore").read_text().splitlines() and "docs/card/cypmode_card-light.gif" not in Path(".gitignore").read_text().splitlines()
    script = Path("scripts/release_after_deadline.py").read_text()
    assert 'build_card.py"), "--full"' in script and 'build_interactive_card.py"), "--full"' in script  # the release builds the full card, with its turning version, and the full page
    assert "turning_card" in Path("docs/card/build_card.py").read_text() and "cypmode_card_full" in Path("docs/card/build_card.py").read_text()


def test_the_zenodo_record_is_described_as_the_citation_file_describes_it_and_the_version_is_the_same_everywhere():
    cff = yaml.safe_load(Path("CITATION.cff").read_text())
    zenodo = json.loads(Path(".zenodo.json").read_text())
    assert zenodo["title"] == cff["title"] and zenodo["description"] == cff["abstract"] and zenodo["keywords"] == cff["keywords"] and zenodo["upload_type"] == "software"
    assert zenodo["version"] == str(cff["version"]) and zenodo["license"] == cff["license"].lower() == "agpl-3.0-or-later"
    assert [c["name"] for c in zenodo["creators"]] == [f"{a['family-names']}, {a['given-names']}" for a in cff["authors"]]
    assert [c["orcid"] for c in zenodo["creators"]] == [a["orcid"].rsplit("/", 1)[-1] for a in cff["authors"]]
    version = str(cff["version"])
    assert f'version = "{version}"' in Path("pyproject.toml").read_text()
    for name in ("README.md", "USE_AND_CITATION.md"):
        assert f"version = {{{version}}}" in Path(name).read_text(), name


def _repository_with_a_commit(folder):
    import subprocess

    folder.mkdir(parents=True)
    (folder / "app.py").write_text("print('x')\n")
    env = {"GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@example.org", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@example.org", "PATH": __import__("os").environ["PATH"], "HOME": str(folder)}
    for command in (["init", "-q"], ["add", "app.py"], ["commit", "-q", "-m", "first"]):
        subprocess.run(["git", "-C", str(folder), *command], check=True, capture_output=True, env=env)


def test_the_folders_of_the_held_back_block_are_read_and_wildcards_and_files_are_left_out():
    text = "x\n# >>> HELD BACK UNTIL AFTER THE DEADLINE (d)\n# a comment/\nexplorer/full/\nsubmissions/*.csv\nsubmissions/manifest.json\ndata/outputs/\n# <<< HELD BACK\nexplorer/other/\n"
    assert release.held_back_directories(text) == ["explorer/full/", "data/outputs/"]


def test_a_held_back_folder_with_its_own_repository_is_bundled_and_loses_its_git_but_keeps_its_files(tmp_path):
    import subprocess

    _repository_with_a_commit(tmp_path / "repo/explorer/full")
    _repository_with_a_commit(tmp_path / "repo/third_party/vendored")  # not in the block: never touched
    gitignore = "x\n# >>> HELD BACK UNTIL AFTER THE DEADLINE (d)\nexplorer/full/\n# <<< HELD BACK\n"
    root = tmp_path / "repo"
    found = release.nested_repositories(root, gitignore)
    assert found == [root / "explorer/full"]
    bundle = release.detach_repository(found[0], tmp_path / "backup", root)
    assert bundle.name == "explorer_full.bundle" and bundle.parent == tmp_path / "backup"
    assert subprocess.run(["git", "-C", str(tmp_path / "repo/third_party/vendored"), "bundle", "list-heads", str(bundle)], capture_output=True, check=False).returncode == 0  # a bundle that git can read
    assert not (root / "explorer/full/.git").exists() and (root / "explorer/full/app.py").exists()
    assert (root / "third_party/vendored/.git").exists()
    assert release.nested_repositories(root, gitignore) == []


def test_nothing_is_removed_when_the_bundle_cannot_be_made(tmp_path):
    import subprocess

    import pytest

    folder = tmp_path / "repo/explorer/full"
    folder.mkdir(parents=True)
    subprocess.run(["git", "-C", str(folder), "init", "-q"], check=True, capture_output=True)  # no commit: git refuses an empty bundle
    with pytest.raises(subprocess.CalledProcessError):
        release.detach_repository(folder, tmp_path / "backup", tmp_path / "repo")
    assert (folder / ".git").exists()
