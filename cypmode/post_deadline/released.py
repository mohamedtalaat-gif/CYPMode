# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The files that the organisers released after the deadline, fetched by name and recorded by SHA-256.

``docs/post_deadline/PLAN.md`` names its sources: the unblinded TDI test file and the 20 released structures of the dataset of the challenge, and the evaluation code and the Boltz-2 baseline
structures of the tutorial repository of the organisers at a fixed commit. ``python -m cypmode.post_deadline.released`` copies them to ``data/post_deadline/inputs`` (not tracked) and writes
``data/post_deadline/released_manifest.json`` (tracked) with the source, the size and the SHA-256 of every file, so that a later run can check that it reads the same bytes. ``--verify`` only
checks. Nothing here reads a label or a structure; the questions of the plan do that, in their own modules.
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/post_deadline/inputs"
MANIFEST = ROOT / "data/post_deadline/released_manifest.json"

DATASET = "https://huggingface.co/datasets/openadmet/cyp-challenge-train-test"
TDI_FILE = "cyp-challenge-TEST-UNBLINDED_TDI.csv"
STRUCTURE_FOLDER = "structures_unblinded"
TUTORIAL = "OpenADMET/CYP-Challenge-Tutorial"
TUTORIAL_COMMIT = "cc02086f735b8ce6398c48ea103cd5f0b2177962"  # 2026-11-01, the last commit of the evaluation code when the plan was written
KEPT_FROM_TUTORIAL = ("evaluation/", "inputs/boltz-2_cyp3a4_predictions/", "LICENSE")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(dest: Path, sources: dict[str, str]) -> list[dict]:
    """One entry per file under `dest`: the path relative to it, the source, the size and the SHA-256, sorted by path."""
    return [{"path": name, "source": sources.get(name, ""), "bytes": (dest / name).stat().st_size, "sha256": sha256_of(dest / name)} for name in sorted(sources)]


def verify(dest: Path, manifest: dict) -> list[str]:
    """The files of the manifest that are missing under `dest` or whose bytes differ from it."""
    wrong = []
    for entry in manifest["files"]:
        path = dest / entry["path"]
        if not path.exists() or path.stat().st_size != entry["bytes"] or sha256_of(path) != entry["sha256"]:
            wrong.append(entry["path"])
    return wrong


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def structure_names() -> list[str]:
    """The files of the released structures, as the dataset lists them."""
    listing = json.loads(_download(f"https://huggingface.co/api/datasets/{DATASET.rsplit('datasets/', 1)[1]}/tree/main/{STRUCTURE_FOLDER}"))
    return sorted(item["path"] for item in listing if item["path"].endswith(".pdb"))


def _extract_tutorial(dest: Path) -> None:
    """Unpack the evaluation code and the baseline structures of the tutorial repository at the fixed commit, unless they are already there."""
    if (dest / "organisers/evaluation/evaluate_predictions.py").exists():
        return
    archive = _download(f"https://codeload.github.com/{TUTORIAL}/tar.gz/{TUTORIAL_COMMIT}")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            relative = member.name.split("/", 1)[1] if "/" in member.name else ""
            if member.isfile() and relative.startswith(KEPT_FROM_TUTORIAL):
                target = dest / "organisers" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())


def fetch(dest: Path = DEST, manifest_path: Path = MANIFEST) -> dict:
    """Copy the released files to `dest` (files that are already there are kept) and write the manifest."""
    sources: dict[str, str] = {}
    for name in [TDI_FILE, *structure_names()]:
        target = dest / "dataset" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{DATASET}/resolve/main/{name}"
        if not target.exists():
            target.write_bytes(_download(url))
        sources[f"dataset/{name}"] = url
    _extract_tutorial(dest)
    for path in sorted((dest / "organisers").rglob("*")):
        if path.is_file():
            sources[path.relative_to(dest).as_posix()] = f"https://github.com/{TUTORIAL}/blob/{TUTORIAL_COMMIT}/{path.relative_to(dest / 'organisers').as_posix()}"
    manifest = {"tutorial_commit": TUTORIAL_COMMIT, "files": describe(dest, sources)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check the files against the manifest and fetch nothing")
    args = parser.parse_args()
    if args.verify:
        wrong = verify(DEST, json.loads(MANIFEST.read_text()))
        print("the released files match the manifest" if not wrong else f"{len(wrong)} file(s) differ or are missing: {wrong[:5]}")
        return 1 if wrong else 0
    manifest = fetch()
    print(f"{len(manifest['files'])} files, {sum(e['bytes'] for e in manifest['files']) / 1e6:.1f} MB; manifest at {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
