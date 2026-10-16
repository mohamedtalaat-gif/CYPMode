"""Scan the repository for content that should not be published.

Checks, over the files git tracks or would add (third-party clones and virtual environments excluded):

  * private local paths (``/Users/<name>``, ``/Volumes/...``) in documents and code;
  * credential-shaped strings (Hugging Face, GitHub, AWS and generic API tokens);
  * e-mail addresses other than the project's public contact lines;
  * files above 95 MB (GitHub rejects a push that contains a file of 100 MB or more);
  * long verbatim runs copied from the reference books and lecture notes kept outside the repository (checked by comparing 12-word
    shingles against ``--library`` when a directory is given);
  * results for the blinded compounds, before the challenge closes: a file that names five or more blinded compounds (in its text or its file names, inside a zip too), or a table with
    one row per blinded compound (750 rows for the TDI test set, 20 for the structure set), is a prediction or a result for them and must not be pushed. The files that are held back are listed
    in the HELD BACK block of ``.gitignore``; the few that name blinded compounds without being a result (the organisers' inputs, the training file, documents that mention compounds in prose)
    are listed in ``REVIEWED`` with the reason. ``--history`` scans every blob of the git history the same way, because a push carries the history.

Findings are printed by file with counts and the first matching line; the exit status is non-zero when any is found.

Usage: python -m cypmode.audit.public_readiness [--library DIR]
"""

import argparse
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = (".py", ".md", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".txt", ".cff", ".html", ".json", ".ipynb", ".csv")
SKIP_PARTS = ("third_party", ".venv", ".venv-ml", ".venv-boltz", ".venv-audit", ".venv-openfold3", "__pycache__", ".git")
PATTERNS = {
    "private path": re.compile(r"/Users/[A-Za-z0-9._-]+|/Volumes/[A-Za-z0-9._ -]+/"),
    "credential": re.compile(r"hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{32,}"),
    "e-mail": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
ALLOWED_EMAIL_DOMAINS = ("example.com", "users.noreply.github.com")
CONTACT_FILES = ("CITATION.cff", "CODE_OF_CONDUCT.md", "SECURITY.md")  # the project's public contact lines
MAX_BYTES = 95 * 1024 * 1024
BLINDED_IDS = re.compile(rb"OCNT-\d{7}")
BLINDED_ROWS = (750, 20)  # the size of the TDI test set and of the structure set: a table of that length with one row per compound is a result for them
IDS_THRESHOLD = 5
REVIEWED = {  # files that name blinded compounds without being a result for them, reviewed on 2026-10-16
    "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv": "the organisers' public test inputs (structures only)",
    "data/structure_track/inputs/structure_track_test_blinded.csv": "the organisers' public structure-track inputs (structures only)",
    "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv": "the organisers' training file; 15 of its compounds are structure ligands",
    "data/tdi_track/outputs/comparisons/oof_direct_inhibition.csv": "out-of-fold predictions for training compounds only",
    "data/tdi_track/outputs/oof_direct_inhibition.csv": "the same file at its earlier path (history)",
    "data/structure_track/inputs/structure_track_retry.csv": "an input list of ligands to run",
    "data/structure_track/inputs/restraint_expansion_ligands.csv": "public crystal ligands of the PDB, not blinded compounds",
    "data/tdi_track/outputs/comparisons/softmax_curve_CYP3A4.csv": "cross-validation summary: one row per selector and cut (20 rows), no compound",
    "data/tdi_track/outputs/comparisons/softmax_curve_CYP2D6.csv": "cross-validation summary: one row per selector and cut (20 rows), no compound",
    "data/tdi_track/outputs/comparisons/baseline_reproduction.csv": "the official baseline's scores in the project's cross-validation",
    "METHOD_REPORT.md": "prose and a dump of the recipe (samples per ligand); no per-compound result",
    "SUBMISSION_CHECKLIST.md": "prose naming the compounds of the leak guard",
    "docs/card/build_card.py": "the identifiers of the six poses of the full card, in code",
    "submissions/structure_track/README.md": "lists the file names of the zip and its SHA-256, and how the poses were chosen; no coordinates",
}
SHINGLE = 12


def candidate_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.splitlines()
    return [ROOT / f for f in sorted(set(listed)) if not any(part in SKIP_PARTS for part in Path(f).parts) and (ROOT / f).is_file()]


def scan_text(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, pattern in PATTERNS.items():
        hits = [m.group(0) for m in pattern.finditer(text)]
        if name == "e-mail":
            hits = [h for h in hits if not h.lower().endswith(ALLOWED_EMAIL_DOMAINS)]
        if hits:
            found[name] = hits
    return found


def shingles(text: str, n: int = SHINGLE) -> set[int]:
    """Hashes of the n-word runs of ``text``, keeping only those whose hash is a multiple of 4 (the same rule on both sides of a
    comparison, so a copied run is found whatever its offset, with a quarter of the memory)."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    hashed = (zlib.crc32(" ".join(words[i : i + n]).encode()) for i in range(max(len(words) - n + 1, 0)))
    return {h for h in hashed if h % 4 == 0}


def library_shingles(directory: Path) -> set[int]:
    out: set[int] = set()
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in (".txt", ".md") and f.is_file():
            out |= shingles(f.read_text(errors="ignore"))
    return out


def blinded_names() -> set[str]:
    """The identifiers of the blinded TDI test compounds and of the structure-track ligands."""
    names: set[str] = set()
    for rel in ("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv", "data/structure_track/inputs/structure_track_test_blinded.csv"):
        path = ROOT / rel
        if path.exists():
            names |= {m.decode() for m in BLINDED_IDS.findall(path.read_bytes())}
    return names


def blinded_hits(path: Path, names: set[str]) -> tuple[int, int]:
    """(distinct blinded identifiers named in the file or its path, data rows of a table), for one file."""
    blobs = [str(path.relative_to(ROOT)).encode()]
    rows = 0
    if path.suffix.lower() == ".zip":
        import zipfile

        try:
            blobs += [n.encode() for n in zipfile.ZipFile(path).namelist()]
        except zipfile.BadZipFile:
            pass
    elif path.stat().st_size < 200_000_000:
        data = path.read_bytes()
        blobs.append(data)
        if path.suffix.lower() in (".csv", ".tsv"):
            rows = max(data.count(b"\n") - 1, 0)
    ids = {m.decode() for blob in blobs for m in BLINDED_IDS.findall(blob)} & names
    return len(ids), rows


def prediction_findings(files: list[Path], names: set[str], reviewed: dict = REVIEWED) -> list[str]:
    """One message per file that holds results for the blinded compounds and is not on the reviewed list."""
    out = []
    for path in files:
        rel = str(path.relative_to(ROOT))
        if rel in reviewed:
            continue
        ids, rows = blinded_hits(path, names)
        if ids >= IDS_THRESHOLD or rows in BLINDED_ROWS:
            out.append(f"RESULT FOR BLINDED COMPOUNDS: {rel} ({ids} identifiers, {rows} table rows)")
    return out


def history_findings(names: set[str], reviewed: dict = REVIEWED) -> list[str]:
    """The same test on every blob of the git history (a push carries all of it); one message per path."""
    objects = subprocess.run(["git", "rev-list", "--objects", "--all"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.splitlines()
    batch = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    found: dict[str, int] = {}
    seen: set[str] = set()
    for line in objects:
        sha, _, path = line.partition(" ")
        if sha in seen or not path:
            continue
        seen.add(sha)
        batch.stdin.write((sha + "\n").encode())
        batch.stdin.flush()
        header = batch.stdout.readline().split()
        if len(header) < 3:
            continue
        size = int(header[2])
        data = batch.stdout.read(size)
        batch.stdout.read(1)
        if header[1] != b"blob" or size > 200_000_000 or path in reviewed:
            continue
        ids = len({m.decode() for m in BLINDED_IDS.findall(data)} & names)
        if ids >= IDS_THRESHOLD:
            found[path] = max(found.get(path, 0), ids)
    batch.stdin.close()
    return [f"RESULT FOR BLINDED COMPOUNDS IN THE HISTORY: {path} ({ids} identifiers)" for path, ids in sorted(found.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="also scan every blob of the git history")
    parser.add_argument("--after-deadline", action="store_true", help="skip the results scan: the held-back files are meant to be published now")
    parser.add_argument("--library", type=Path, help="directory of reference texts to compare against")
    args = parser.parse_args()
    library = library_shingles(args.library) if args.library and args.library.exists() else set()
    problems = 0
    for path in candidate_files():
        rel = path.relative_to(ROOT)
        if path.stat().st_size > MAX_BYTES:
            print(f"LARGE FILE: {rel} ({path.stat().st_size // (1024 * 1024)} MB)")
            problems += 1
        if path.suffix.lower() not in TEXT or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(errors="ignore")
        for kind, hits in scan_text(text).items():
            if kind == "e-mail" and rel.name in CONTACT_FILES:
                continue
            print(f"{kind.upper()}: {rel} x{len(hits)}  e.g. {hits[0][:70]}")
            problems += 1
        if library and (shingles(text) & library):
            print(f"VERBATIM LIBRARY TEXT: {rel} shares {len(shingles(text) & library)} {SHINGLE}-word runs with the reference texts")
            problems += 1
    names = blinded_names()
    findings = [] if args.after_deadline else prediction_findings(candidate_files(), names) + (history_findings(names) if args.history else [])
    for message in findings:
        print(message)
    problems += len(findings)
    print(f"public readiness: {problems} finding(s) in {len(candidate_files())} files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
