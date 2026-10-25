# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Rules about the wording of the text of the project that a script can check.

The text of the project (the notebook, the reports, the checklist, the documents, the comments of the code) is written in the voice of the researcher who did the work. A decision, a request or a step is
written as what was decided or done, with its date, and never as the act of a person who is named by a role: no text refers to an owner.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".sh", ".txt", ".toml", ".cff", ".js", ".html", ".tex"}
EXEMPT = (
    "audit/AUDIT_REPORT.md",  # generated: it quotes the name of the check
    "audit/AUDIT_REPORT.run2.md",
    "cypmode/audit/prose_rules.py",  # these two name the word on purpose
    "tests/test_prose_rules.py",
)
EXEMPT_NAMES = ("LICENSE", "LICENCE", "COPYING")  # licence texts that are not ours
OWNER = re.compile(r"\bowner(?:'s|s)?\b", re.IGNORECASE)


def project_files(root: Path = ROOT) -> list[str]:
    """The files git tracks or would add, as paths relative to the root."""
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=False).stdout.splitlines()
    return sorted({name for name in listed if (root / name).is_file()})


def mentions_of_the_owner(root: Path = ROOT, files: list[str] | None = None) -> list[str]:
    """``path:line`` of every line of a text file of the project that refers to an owner."""
    found = []
    for name in files if files is not None else project_files(root):
        path = Path(name)
        if path.suffix not in TEXT_SUFFIXES or name in EXEMPT or path.name.upper().startswith(EXEMPT_NAMES) or name.startswith(("third_party/", "data/")):
            continue
        for number, line in enumerate((root / name).read_text(errors="ignore").splitlines(), start=1):
            if OWNER.search(line):
                found.append(f"{name}:{number}")
    return found
