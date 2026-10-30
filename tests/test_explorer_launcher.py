# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

IMPORT_TO_PACKAGE = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow", "chembl_structure_pipeline": "chembl-structure-pipeline"}
NOT_THIRD_PARTY = {"cypmode", "simulators", "json", "sys", "pathlib", "math", "re", "os"}


def requirement_names(path: str) -> set[str]:
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert all("==" in line for line in lines), f"{path}: a requirement is not pinned"
    return {line.split("==")[0].lower().replace("_", "-") for line in lines}


def third_party_imports(*paths: Path) -> set[str]:
    found: set[str] = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    stdlib = set(__import__("sys").stdlib_module_names)
    return {IMPORT_TO_PACKAGE.get(name, name).lower().replace("_", "-") for name in found - stdlib - NOT_THIRD_PARTY}


@pytest.mark.skipif(sys.platform == "win32", reason="the queue and the launchers are bash scripts of Linux and macOS; Windows has run_cypmode.bat instead")
def test_the_launchers_parse_and_prepare_their_own_environment_instead_of_asking_the_user_to():
    for script in ("scripts/explorer.sh", "run_cypmode.command"):
        assert subprocess.run(["bash", "-n", script], check=False).returncode == 0, script
    assert "ensure_env" in Path("scripts/explorer.sh").read_text() and "bash scripts/explorer.sh run" in Path("run_cypmode.command").read_text()
    for launcher in ("run_cypmode.command", "run_cypmode.bat"):
        assert "No .venv-marimo found" not in Path(launcher).read_text(), launcher
    bat = Path("run_cypmode.bat").read_text()
    assert "pip install -r" in bat and "requirements.installed" in bat and "astral.sh/uv" in bat and "EXPLORER_ASSUME_YES" in bat
    shell = Path("scripts/explorer.sh").read_text()
    assert "astral.sh/uv" in shell and "Press Enter to continue" in shell and "EXPLORER_ASSUME_YES" in shell  # the download of uv is announced and asked for


def test_every_requirement_is_pinned_and_covers_what_its_notebook_imports():
    names = requirement_names("environment/requirements-lock-marimo.txt")
    assert third_party_imports(Path("explorer/cypmode_explorer.py")) <= names
    if Path("explorer/full/app.py").exists():
        full = requirement_names("explorer/full/requirements.txt")
        files = [Path("explorer/full/app.py"), *Path("explorer/full/simulators").glob("*.py")]
        assert third_party_imports(*files) <= full, third_party_imports(*files) - full


@pytest.mark.skipif(sys.platform == "win32", reason="the queue and the launchers are bash scripts of Linux and macOS; Windows has run_cypmode.bat instead")
def test_the_plan_mode_names_the_notebook_that_would_run_and_changes_nothing():
    before = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False).stdout
    out = subprocess.run(["bash", "scripts/explorer.sh", "plan"], capture_output=True, text=True, check=False)
    assert out.returncode == 0
    expected = "explorer/full/app.py" if Path("explorer/full/app.py").exists() else "explorer/cypmode_explorer.py"
    assert f"notebook: {expected}" in out.stdout and re.search(r"environment: (ready|to be built)", out.stdout)
    assert subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False).stdout == before
