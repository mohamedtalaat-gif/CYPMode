# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Wording rules for the text files of the project, checked by the audit (section 21).

Each rule is one pattern. The audit counts the lines that match, per rule, over every text file that git tracks or would add and over the commit messages of all branches; a rule fails when a line matches.
A decision or a step is written as what was decided or done, with its date. The word of the first rule is written in two pieces so that no file of the repository holds it; the other lists are held here,
and this module and its test are the only files that name them.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".sh", ".txt", ".toml", ".cff", ".js", ".html", ".tex"}
EXEMPT = (
    "audit/AUDIT_REPORT.md",  # generated: it quotes the names of the rules
    "audit/AUDIT_REPORT.run2.md",
    "cypmode/audit/prose_rules.py",  # these two hold the lists of the rules
    "tests/test_prose_rules.py",
)
EXEMPT_NAMES = ("LICENSE", "LICENCE", "COPYING", "CODE_OF_CONDUCT")  # licence texts and the standard code of conduct, which are not ours
RULE_EXEMPT = {"notebooks/parts/appendix_c.py": ("first-person plural", "contractions")}  # the titles of the references are quoted as published
EXEMPT_PARTS = ("third_party/", "data/", "_vendor/", "LICENSES/", "docs/card/vendor/", "docs/card/interactive/", "notebooks/public/", "notebooks/cypmode_notebook.")  # vendored libraries (the pages of the card and of the notebook hold them inline, with images as base64 text) and licence texts that are not ours; the prose of the notebook is scanned in its parts
EXEMPT_SUFFIXES = (".min.js", "-min.js")  # minified libraries

ROLE = "own" + "er"
INTENSIFIERS = ("honest", "honestly", "genuine", "genuinely", "straightforward", "crucial", "crucially", "delve", "delves", "delved", "leverage", "leverages", "leveraged", "leveraging", "seamless", "seamlessly")
STOCK = ("worth noting", "worth mentioning", "importantly", "notably", "furthermore", "moreover", "let's", "here's", "great question", "hope this helps")

RULES = {
    "role names as the actor of a decision or a step": re.compile(
        rf"\b{ROLE}(?:'s|s)?\b|\bthe user(?:'s)? (?:asked|said|decided|chose|wants|wanted|approved|requested|reminded|objected|agreed)\b", re.IGNORECASE
    ),
    "first-person singular verbs": re.compile(r"\bI(?:'ve|'ll|'m|'d)\b|\bI (?:added|decided|chose|checked|found|ran|wrote|did|will|have|am|think|believe|noticed|verified|made|removed|fixed|can|could|would|should|want)\b"),
    "references to the working session or to a call of the reader": re.compile(r"\b(?:this|the|that|earlier|last|previous) session(?:'s)?\b|\byour call\b|\bfor you to decide\b", re.IGNORECASE),
    "first-person plural": re.compile(r"\b(?:[Ww]e|[Oo]ur|us)\b"),
    "contractions": re.compile(
        r"\b(?:isn't|aren't|wasn't|weren't|doesn't|don't|didn't|can't|couldn't|won't|wouldn't|shouldn't|hasn't|haven't|hadn't|it's|that's|there's|here's|what's|let's|we're|you're|they're|I'm|I've|we've|you've|they've|he's|she's|who's)\b",
        re.IGNORECASE,
    ),
    "notes of status or of unfinished work": re.compile(r"\b(?:NOT WRITTEN YET|not written yet|TODO|FIXME|to be decided|awaiting (?:a )?(?:decision|approval|go-ahead))\b"),
    "long dashes and decorative symbols": re.compile("\u2014| \u2013 |[\u2713\u2714\u2705\u274c\u2717\U0001F300-\U0001FAFF]"),
    "intensifier words": re.compile(r"\b(?:" + "|".join(INTENSIFIERS) + r")\b", re.IGNORECASE),
    "conversational openers and stock phrases": re.compile(r"\b(?:" + "|".join(re.escape(p) for p in STOCK) + r")", re.IGNORECASE),
}
# Established terms that contain a listed word and are not a manner of writing: they are removed from a line before it is matched.
TERMS = (re.compile(r"honestly significant difference", re.IGNORECASE),)


def project_files(root: Path | None = None) -> list[str]:
    """The files git tracks or would add, as paths relative to the root."""
    root = root or ROOT
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=False).stdout.splitlines()
    return sorted({name for name in listed if (root / name).is_file()})


def scanned(name: str) -> bool:
    """Whether a file is text of the project that the rules apply to."""
    path = Path(name)
    return not (
        path.suffix not in TEXT_SUFFIXES
        or name in EXEMPT
        or path.name.upper().startswith(EXEMPT_NAMES)
        or name.startswith(EXEMPT_PARTS)
        or any(part in name for part in ("/_vendor/", "/node_modules/"))
        or name.endswith(EXEMPT_SUFFIXES)
    )


QUOTATION = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d')
POSSESSIVE = re.compile(r"\bours\b")  # also an identifier of the code, so it is matched in the documents only


def violations(root: Path | None = None, files: list[str] | None = None) -> dict[str, list[str]]:
    """``rule -> [path:line, ...]`` for every line of a scanned file that matches the rule."""
    root = root or ROOT
    found: dict[str, list[str]] = {rule: [] for rule in RULES}
    for name in files if files is not None else project_files(root):
        if not scanned(name):
            continue
        for number, line in enumerate((root / name).read_text(errors="ignore").splitlines(), start=1):
            for term in TERMS:
                line = term.sub("", line)
            unquoted = QUOTATION.sub("", line) if name.endswith((".md", ".txt")) else line  # in a document a quotation keeps the words of its source
            for rule, pattern in RULES.items():
                if rule in RULE_EXEMPT.get(name, ()):
                    continue
                text = unquoted if rule == "contractions" else line
                if pattern.search(text) or (rule == "first-person plural" and name.endswith(".md") and POSSESSIVE.search(line)):
                    found[rule].append(f"{name}:{number}")
    return found


def role_mentions(root: Path | None = None, files: list[str] | None = None) -> list[str]:
    """``path:line`` of every line of a scanned file that matches the first rule (the role names)."""
    return violations(root, files)["role names as the actor of a decision or a step"]
