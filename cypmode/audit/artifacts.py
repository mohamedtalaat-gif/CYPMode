# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Provenance of the files that present the results: the README card, the poster, the film and the figures.

A number printed on a card or drawn in a frame of the film is only as good as the build that put it there. Each of these artifacts is written by a builder that records, in
``audit/artifacts.json``, the SHA-256 of every file it wrote and a fingerprint of the numbers it was built from (``fingerprint``: everything ``docs/video/facts.py`` reads from the final
outputs, the ledger, the audit and the repository). ``verify`` then answers, for the pre-submission gate, three questions per artifact:

  * is every recorded output still the file the builder wrote (nobody edited or replaced it)?
  * were the numbers it shows the final ones (the recorded fingerprint equals the current one)?
  * deep check: does the artifact's own source still produce the same content? The card's HTML and the DOM of every frame of the film and of the poster are deterministic, so their
    digests are recomputed and compared; the PNG and the H.264 files themselves are not compared pixel by pixel, because rendering the same frame twice differs in a few dozen pixels by
    less than a level of 255 and the encoder adds its own differences.

The audit report of ``audit/AUDIT_REPORT.md`` is generated twice and compared byte for byte (``scripts/run_audit.sh``); this module extends the same discipline to the presentation files.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "audit/artifacts.json"
REQUIRED = ("readme_card", "interactive_card", "poster", "film", "figures")  # the artifacts whose record the gate demands
PASS, FAIL = "PASS", "FAIL"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint() -> str:
    """SHA-256 of every number that the card, the poster, the film and the figures take from the project's files."""
    sys.path[:0] = [str(ROOT / "docs/video")]
    import facts

    payload = {"facts": asdict(facts.load()), "real": facts.real_data()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(root)) if root in resolved.parents else str(path)


def load(manifest: Path = MANIFEST) -> dict:
    return json.loads(manifest.read_text()) if manifest.exists() else {}


def record(name: str, outputs: list[Path], extra: dict | None = None, fingerprint_fn=fingerprint, manifest: Path = MANIFEST, root: Path = ROOT) -> None:
    """Called by a builder after it has written its files: the hash of each output, the fingerprint of the numbers it was built from and the builder's own digests (``extra``)."""
    data = load(manifest)
    data[name] = {"outputs": {_relative(Path(p), root): sha256_file(Path(p)) for p in outputs},
                  "fingerprint": fingerprint_fn(), "extra": extra or {}, "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def verify(deep: bool = False, fingerprint_fn=fingerprint, manifest: Path = MANIFEST, deep_checks: dict | None = None, root: Path = ROOT) -> list[tuple[str, str, str]]:
    """One row (name, PASS or FAIL, detail) per question and artifact; a missing record is a failure, never a skip."""
    data, rows = load(manifest), []
    current = fingerprint_fn() if data else None
    for name in REQUIRED:
        entry = data.get(name)
        if entry is None:
            rows.append((f"{name}: built by its builder and recorded", FAIL, "no record in audit/artifacts.json; run its build"))
            continue
        changed = [rel for rel, digest in entry["outputs"].items() if not (root / rel).exists() or sha256_file(root / rel) != digest]
        rows.append((f"{name}: the files are the ones the builder wrote", PASS if not changed else FAIL, f"{len(entry['outputs'])} files" if not changed else "changed or missing: " + ", ".join(changed[:3])))
        rows.append((f"{name}: built from the final numbers", PASS if entry["fingerprint"] == current else FAIL,
                     f"fingerprint {current[:12]}" if entry["fingerprint"] == current else f"built from {entry['fingerprint'][:12]}, the numbers are now {current[:12]}; rebuild it"))
        checker = (deep_checks if deep_checks is not None else DEEP_CHECKS).get(name)
        if deep and checker is not None:
            ok, detail = checker(entry)
            rows.append((f"{name}: its source still produces the recorded content", PASS if ok else FAIL, detail))
    return rows


def asset_digest() -> str:
    """SHA-256 over the content of every picture the film draws from files (the PyMOL stills, the painted sprites, the Blender clips): the DOM names them by path, so the audit must hash what is inside."""
    base = ROOT / "docs/video/anime"
    digest = hashlib.sha256()
    for folder in ("assets", "characters/cut", "clips"):
        for path in sorted((base / folder).rglob("*")):
            if path.is_file() and path.suffix in (".png", ".json", ".jpg"):
                digest.update(f"{path.relative_to(base)}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def deep_card(entry: dict) -> tuple[bool, str]:
    sys.path[:0] = [str(ROOT / "docs/card")]
    import build_card

    expected = entry["extra"].get("html_sha256", {})
    now = {theme: hashlib.sha256(build_card.card_html(theme, entry["extra"].get("mode", "public")).encode()).hexdigest() for theme in build_card.THEMES}
    differing = [theme for theme in now if now[theme] != expected.get(theme)]
    return not differing, "card HTML identical for both themes" if not differing else f"the HTML of the {', '.join(differing)} card differs from the recorded one"


def _film_digests(poster_only: bool) -> dict:
    sys.path[:0] = [str(ROOT / "docs/video"), str(ROOT / "docs/video/anime")]
    import facts as facts_module
    import render_video

    (ROOT / "docs/video/anime/facts.js").write_text(facts_module.to_javascript(facts_module.load(), facts_module.real_data()))
    return render_video.dom_digests(poster_only)


def deep_poster(entry: dict) -> tuple[bool, str]:
    now = _film_digests(True)["poster"]
    return now == entry["extra"].get("dom_digest"), "poster DOM identical" if now == entry["extra"].get("dom_digest") else "the DOM of the poster differs from the recorded one"


def deep_film(entry: dict) -> tuple[bool, str]:
    now = _film_digests(False)["film"]
    same, assets = now == entry["extra"].get("dom_digest"), asset_digest() == entry["extra"].get("assets")
    detail = "the DOM of every frame and the content of every picture it draws are identical" if same and assets else (
        "the DOM of at least one frame differs from the recorded one" if not same else "a picture the film draws (still, sprite or clip) differs from the recorded one")
    return same and assets, detail


def deep_interactive_card(entry: dict) -> tuple[bool, str]:
    sys.path[:0] = [str(ROOT / "docs/card")]
    import build_interactive_card

    now = hashlib.sha256(build_interactive_card.page(entry["extra"].get("mode", "public")).encode()).hexdigest()
    same = now == entry["extra"].get("html_sha256")
    return same, "interactive card HTML identical" if same else "the HTML of the interactive card differs from the recorded one"


DEEP_CHECKS = {"readme_card": deep_card, "interactive_card": deep_interactive_card, "poster": deep_poster, "film": deep_film}
