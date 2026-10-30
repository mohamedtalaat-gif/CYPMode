# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Publish what was held back: ``python scripts/release_after_deadline.py [--dry-run] [--rebuild-card]``.

Until the challenge closes (2026-11-03 23:59 UTC) the repository carries the code and the reports but no result for a blinded compound: the HELD BACK block of ``.gitignore`` keeps the
predictions, the submitted structures, the pictures and the film that show them, and the notebook exports on the local machine, and the film section and the full card of the README are commented out, while the public card (crystal structures of the PDB only) is shown. This script,
which refuses to run before the deadline, removes that block, removes the public edition of the notebook (the complete one replaces it), replaces the public card by the full card, restores the film section and (with ``--rebuild-card``) rebuilds the README card with the submitted structures. It changes nothing
else and never commits or pushes; it prints what becomes visible so that it can be reviewed first.
"""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = datetime(2026, 11, 3, 23, 59, tzinfo=timezone.utc)
BLOCK = re.compile(r"\n# >>> HELD BACK UNTIL AFTER THE DEADLINE.*?# <<< HELD BACK\n", re.DOTALL)
HELD_SECTIONS = (("<!-- held-back:film\n", "held-back:film-end -->\n"), ("<!-- held-back:card\n", "held-back:card-end -->\n"))  # the film section and the full card at the top of the README
PUBLIC_CARD = re.compile(r"<!-- public:card begin -->\n.*?<!-- public:card end -->\n\n?", re.DOTALL)  # the public card, which the full card replaces


def strip_held_back(gitignore: str) -> str:
    """The .gitignore without its HELD BACK block."""
    return BLOCK.sub("\n", gitignore)


def remove_public_card(readme: str) -> str:
    """The README without the public card, which the full card replaces."""
    return PUBLIC_CARD.sub("", readme)


def restore_film(readme: str) -> str:
    """The README with the commented-out film section and full card switched on."""
    for opening, closing in HELD_SECTIONS:
        readme = readme.replace(opening, "").replace(closing, "")
    return readme


def remove_public_edition(root: Path = ROOT) -> bool:
    """Remove the public edition of the notebook, which the complete edition replaces; True when there was one."""
    public = root / "notebooks/public"
    if public.exists():
        shutil.rmtree(public)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show what would change and write nothing")
    parser.add_argument("--rebuild-card", action="store_true", help="rebuild the README card in its full mode (PyMOL and Chrome needed)")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if now < DEADLINE and not args.dry_run:
        print(f"the challenge closes at {DEADLINE:%Y-%m-%d %H:%M} UTC and it is {now:%Y-%m-%d %H:%M} UTC: nothing is released before then (use --dry-run to look)")
        return 1
    gitignore, readme = (ROOT / ".gitignore"), (ROOT / "README.md")
    new_ignore, new_readme = strip_held_back(gitignore.read_text()), restore_film(remove_public_card(readme.read_text()))
    print(f"HELD BACK block {'found' if new_ignore != gitignore.read_text() else 'not found'}; held-back sections {'found' if new_readme != readme.read_text() else 'not found'}")
    print(f"the public edition of the notebook {'is there and would be removed: the complete edition replaces it' if (ROOT / 'notebooks/public').exists() else 'is not there'}")
    if args.dry_run:
        return 0
    remove_public_edition()
    gitignore.write_text(new_ignore)
    readme.write_text(new_readme)
    if args.rebuild_card:
        subprocess.run([sys.executable, str(ROOT / "docs/card/build_card.py"), "--full"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "docs/card/build_interactive_card.py"), "--full"], cwd=ROOT, check=True)
    print("what the push would now add (review it before you commit; the readiness scan with --after-deadline checks the rest):")
    subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
