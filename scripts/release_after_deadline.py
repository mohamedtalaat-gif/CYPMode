# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Publish what was held back: ``python scripts/release_after_deadline.py [--dry-run] [--rebuild-card]``.

Until the challenge closes (2026-11-03 23:59 UTC) the repository carries the code and the reports but no result for a blinded compound: the HELD BACK block of ``.gitignore`` keeps the
predictions, the submitted structures, the pictures and the film that show them, and the notebook exports on the local machine, and the film section and the full card of the README are commented out, while the public card (crystal structures of the PDB only) is shown. This script,
which refuses to run before the deadline, removes that block, removes the public edition of the notebook (the complete one replaces it), replaces the public card by the full card, restores the film section and (with ``--rebuild-card``) rebuilds the README card with the submitted structures. A held-back folder that carries a repository of its own (the explorer) is first written to a verified
bundle and loses its ``.git``, because git would otherwise record the folder as a link and leave its files out. It changes nothing else and never commits or pushes; it prints what becomes visible so that it can be reviewed first.
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
HELD_INLINE = re.compile(r"<!-- held-back:(\w+) (.*?) held-back:\1-end -->")  # a sentence of one line (a row of a table) that the complete edition shows
PUBLIC_INLINE = re.compile(r"<!-- public:(\w+) begin -->.*?<!-- public:\1 end -->")  # the sentence of the same line that the public edition shows and the complete one replaces
PUBLIC_CARD = re.compile(r"<!-- public:card begin -->\n.*?<!-- public:card end -->\n\n?", re.DOTALL)  # the public card, which the full card replaces


def strip_held_back(gitignore: str) -> str:
    """The .gitignore without its HELD BACK block."""
    return BLOCK.sub("\n", gitignore)


def remove_public_card(readme: str) -> str:
    """The README without the public card, which the full card replaces."""
    return PUBLIC_INLINE.sub("", PUBLIC_CARD.sub("", readme))


def restore_film(readme: str) -> str:
    """The README with the commented-out film section and full card switched on."""
    for opening, closing in HELD_SECTIONS:
        readme = readme.replace(opening, "").replace(closing, "")
    return HELD_INLINE.sub(lambda m: m.group(2), readme)


def remove_public_edition(root: Path = ROOT) -> bool:
    """Remove the public edition of the notebook, which the complete edition replaces; True when there was one."""
    public = root / "notebooks/public"
    if public.exists():
        shutil.rmtree(public)
        return True
    return False


def held_back_directories(gitignore: str) -> list[str]:
    """The folders that the HELD BACK block lists (entries that end with a slash and hold no wildcard)."""
    block = BLOCK.search(gitignore)
    lines = [line.strip() for line in block.group(0).splitlines()] if block else []
    return [line for line in lines if line.endswith("/") and not line.startswith("#") and not re.search(r"[*?\[]", line)]


def nested_repositories(root: Path, gitignore: str) -> list[Path]:
    """The held-back folders that carry a repository of their own: once the block is gone, git would record such a folder as a link to that repository and leave its files out of the commit."""
    return [root / folder for folder in held_back_directories(gitignore) if (root / folder / ".git").exists()]


def detach_repository(folder: Path, backup_dir: Path, root: Path = ROOT) -> Path:
    """Write the history of the repository in `folder` to a bundle in `backup_dir`, verify the bundle, then remove the `.git` of the folder; its files stay. Returns the bundle. Nothing is removed unless the bundle verifies."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    bundle = backup_dir / (folder.relative_to(root).as_posix().replace("/", "_") + ".bundle")
    staged = bundle.with_name(bundle.name + ".new")
    subprocess.run(["git", "-C", str(folder), "bundle", "create", str(staged), "--all"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(folder), "bundle", "verify", str(staged)], check=True, capture_output=True)
    staged.replace(bundle)
    entry = folder / ".git"
    shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show what would change and write nothing")
    parser.add_argument("--backup-dir", type=Path, default=None, help="where the bundle of a held-back repository goes before its .git is removed (default: the folder next to the repository, named after it with -backup)")
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
    nested = nested_repositories(ROOT, gitignore.read_text())
    backup_dir = args.backup_dir or ROOT.parent / f"{ROOT.name}-backup"
    for folder in nested:
        print(f"{folder.relative_to(ROOT)} has a repository of its own: its history {'would be' if args.dry_run else 'is'} written to a bundle in {backup_dir} and its .git removed, so that git records its files")
    if args.dry_run:
        return 0
    for folder in nested:
        print(f"bundle of {folder.relative_to(ROOT)}: {detach_repository(folder, backup_dir)}")
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
