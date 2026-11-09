# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The data record: the files that the repository does not hold (the caches, the nested-validation outputs and the raw outputs of the structure engines) packed into a few archives, and the way back.

``data_record/record.json`` is the policy: it names the groups of files that form the record (a folder, minus some sub-folders) and the folders and patterns that are left out, each with the reason.
``build`` packs every group into one ``.tar.xz``, cuts an archive larger than ``part_bytes`` of the policy into parts that can be sent one by one, and writes ``data_record/MANIFEST.json`` (the size and SHA-256 of each archive and of each part) and, per group, the sorted list of its paths. ``fetch`` copies or downloads the archives (the parts of a cut archive are joined), refuses any whose
SHA-256 differs, extracts them where the code looks for them and checks that every listed path is there. ``missing`` is the cheap presence check that the audit runs before it reads a file. A file counts as part of the record when git
does not track it, whether the ``.gitignore`` hides it or not, so the same policy holds before and after the held-back files are committed.
"""

import fnmatch
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "data_record"
POLICY = f"{FOLDER}/record.json"
MANIFEST = f"{FOLDER}/MANIFEST.json"
LISTS = f"{FOLDER}/files"


def load_policy(root: Path = ROOT) -> dict:
    return json.loads((root / POLICY).read_text())


def untracked_files(root: Path, paths: list[str]) -> list[str]:
    """Every file under `paths` that git does not track, ignored or not, sorted."""
    out = subprocess.run(["git", "ls-files", "-o", "-z", "--", *paths], cwd=root, capture_output=True, text=True, check=True).stdout
    return sorted(f for f in out.split("\0") if f)


def _under(path: str, folders: list[str]) -> bool:
    """Whether the path is one of the folders (or inside one) or matches a pattern with wildcards, which may stand for a folder as well as for a file."""
    for entry in folders:
        entry = entry.rstrip("/")
        if "*" in entry or "?" in entry or "[" in entry:
            if fnmatch.fnmatchcase(path, entry) or fnmatch.fnmatchcase(path, entry + "/*"):
                return True
        elif path == entry or path.startswith(entry + "/"):
            return True
    return False


def group_files(root: Path, group: dict) -> list[str]:
    return [f for f in untracked_files(root, group["paths"]) if not _under(f, group.get("exclude", []))]


def left_out(root: Path, policy: dict | None = None) -> list[str]:
    """Untracked files under the folders of the policy that belong to no group and to no left-out folder: nothing is dropped without being named."""
    policy = policy or load_policy(root)
    folders = sorted({p for g in policy["groups"] for p in g["paths"]} | {p for e in policy["not_in_the_record"] for p in e["paths"] if not any(c in p for c in "*?[")})
    named = set().union(*(set(group_files(root, g)) for g in policy["groups"]))
    left = [p for e in policy["not_in_the_record"] for p in e["paths"]]
    return [f for f in untracked_files(root, folders) if f not in named and not _under(f, left)]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _normalised(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _split(archive: Path, part_bytes: int) -> list[dict]:
    """Cut the archive into parts of at most `part_bytes` bytes, remove the whole file and return the manifest entries of the parts."""
    parts = []
    with open(archive, "rb") as source:
        while True:
            name = f"{archive.name}.part-{len(parts):03d}"
            digest, size = hashlib.sha256(), 0
            with open(archive.parent / name, "wb") as part:
                while size < part_bytes:
                    block = source.read(min(1 << 20, part_bytes - size))
                    if not block:
                        break
                    part.write(block)
                    digest.update(block)
                    size += len(block)
            if size == 0:
                (archive.parent / name).unlink()
                break
            parts.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
    archive.unlink()
    return parts


def pack(root: Path, group: dict, out_dir: Path, suffix: str = ".tar.xz", part_bytes: int = 0) -> dict:
    """Write the archive of one group and the sorted list of its paths; return the manifest entry. An archive larger than `part_bytes` (when that is not 0) is kept as parts, and the entry then lists them."""
    files = group_files(root, group)
    if not files:
        raise SystemExit(f"group {group['name']}: no untracked file under {group['paths']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{group['name']}{suffix}"
    with tarfile.open(archive, "w:xz", preset=6) as tar:
        for name in files:
            tar.add(root / name, arcname=name, recursive=False, filter=_normalised)
    (root / LISTS).mkdir(parents=True, exist_ok=True)
    with open(root / LISTS / f"{group['name']}.txt.gz", "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(("\n".join(files) + "\n").encode())
    entry = {"name": group["name"], "archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_of(archive), "files": len(files), "files_bytes": sum((root / f).stat().st_size for f in files)}
    if part_bytes and entry["bytes"] > part_bytes:
        entry["parts"] = _split(archive, part_bytes)
    return entry


def build(root: Path, out_dir: Path) -> dict:
    policy = load_policy(root)
    manifest = {"record": policy.get("record", {}), "archives": [pack(root, g, out_dir, policy.get("archive_suffix", ".tar.xz"), policy.get("part_bytes", 0)) for g in policy["groups"]]}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def listed_paths(root: Path, name: str) -> list[str]:
    with gzip.open(root / LISTS / f"{name}.txt.gz", "rt") as f:
        return [line for line in f.read().splitlines() if line]


def missing(root: Path = ROOT) -> dict[str, list[str]]:
    """Per group of the manifest, the listed paths that are not on disk; empty when the record has not been made yet or nothing is missing."""
    if not (root / MANIFEST).exists():
        return {}
    gone = {e["name"]: [p for p in listed_paths(root, e["name"]) if not (root / p).exists()] for e in json.loads((root / MANIFEST).read_text())["archives"]}
    return {name: paths for name, paths in gone.items() if paths}


def message_for(gone: dict[str, list[str]]) -> str:
    lines = ["the data of the record is not complete on this machine:"] + [f"  {name}: {len(paths)} file(s) missing, e.g. {paths[0]}" for name, paths in gone.items()]
    return "\n".join(lines + ["get it with: python scripts/fetch_data.py fetch   (the archives of the data record, checked by SHA-256), or rebuild the TDI caches with scripts/reproduce_tdi.sh"])


def _obtain(source: str, entry: dict, work: Path, url_suffix: str = "") -> Path:
    """The archive of `entry` from a folder or from a URL prefix (then downloaded into `work`)."""
    if not source.startswith(("http://", "https://")):
        return Path(source) / entry["archive"]
    target = work / entry["archive"]
    urllib.request.urlretrieve(f"{source.rstrip('/')}/{entry['archive']}{url_suffix}", target)
    return target


def _assemble(source: str, entry: dict, work: Path, url_suffix: str = "") -> Path:
    """The whole archive of an entry that was cut into parts: every part is obtained and checked, and the parts are joined in `work`."""
    target = work / entry["archive"]
    downloaded = source.startswith(("http://", "https://"))
    with open(target, "wb") as whole:
        for part in entry["parts"]:
            path = _obtain(source, {"archive": part["name"]}, work, url_suffix)
            if path.stat().st_size != part["bytes"] or sha256_of(path) != part["sha256"]:
                raise SystemExit(f"{path}: the size or the SHA-256 differs from the manifest; nothing was extracted from it")
            with open(path, "rb") as piece:
                shutil.copyfileobj(piece, whole, 1 << 20)
            if downloaded:
                path.unlink()
    return target


def fetch(root: Path, source: str | None = None, only: list[str] | None = None, work: Path | None = None) -> list[str]:
    """Copy or download the archives, refuse any whose SHA-256 or size differs from the manifest, extract them under `root` and check the listed paths. Returns the names of the groups restored."""
    manifest = json.loads((root / MANIFEST).read_text())
    source = source or manifest["record"].get("base_url")
    if not source:
        raise SystemExit("the manifest names no address for the archives: pass --source (a folder or a URL prefix)")
    work = work or root / FOLDER / "downloads"
    work.mkdir(parents=True, exist_ok=True)
    done = []
    for entry in manifest["archives"]:
        if only and entry["name"] not in only:
            continue
        suffix = manifest["record"].get("url_suffix", "")
        archive = _assemble(source, entry, work, suffix) if entry.get("parts") else _obtain(source, entry, work, suffix)
        if archive.stat().st_size != entry["bytes"] or sha256_of(archive) != entry["sha256"]:
            raise SystemExit(f"{archive}: the size or the SHA-256 differs from the manifest; nothing was extracted from it")
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(root, filter="data")
        if entry.get("parts"):
            archive.unlink()  # the joined copy is not needed once it is extracted
        absent = [p for p in listed_paths(root, entry["name"]) if not (root / p).exists()]
        if absent:
            raise SystemExit(f"{entry['name']}: {len(absent)} listed path(s) are not there after extraction, e.g. {absent[0]}")
        done.append(entry["name"])
    return done


def terms_read(root: Path = ROOT) -> list[dict]:
    """What was read of the licence of each source and model behind the files of the record, as the policy states it."""
    return load_policy(root).get("terms_read", [])
