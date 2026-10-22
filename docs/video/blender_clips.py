# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Render the cinematic molecule clips of the film with Blender: ``python docs/video/blender_clips.py [--only enzyme] [--frames 8]``.

Each clip is drawn from real coordinates by ``blender/scene.py`` (Blender with the Molecular Nodes add-on): a translucent enzyme, the heme and the ligand glowing, a camera orbit with depth of
field. The frames go to ``anime/clips/<name>/f_NNNN.png`` and ``anime/assets_clips.js`` tells the film which clips exist, their size, frame rate and, per frame, the screen position of the atoms
the callouts follow (``fe`` and ``coordinating_n``). The film falls back to the PyMOL pictures of ``pymol_clips.py`` for any clip that is missing.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLIPS = HERE / "anime" / "clips"
INDEX = HERE / "anime" / "assets_clips.js"
BLENDER = shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"
SCENE = HERE / "blender" / "scene.py"
STRUCTURE = ROOT / "data/structure_track/inputs/pdb_cache/6MA7.cif"  # CYP3A4 with heme and the Type II ligand TPF, the structure the film already shows
FPS, SIZE = 24, (1280, 720)

HEME = {"selection": "resname HEM", "style": "ball_and_stick", "colour": [1.0, 0.38, 0.05], "strength": 0.9}
LIGAND = {"selection": "resname TPF", "style": "ball_and_stick", "colour": [0.05, 0.8, 0.75], "strength": 1.0}
LINING = {"selection": "protein and around 4.5 resname TPF", "style": "sticks", "colour": [0.5, 0.8, 1.0], "strength": 0.35}
ANCHORS = {"fe": {"resname": "HEM", "name": "FE"}, "coordinating_n": {"resname": "TPF", "element": "N", "nearest_to": "fe"}}

# name -> (seconds, configuration)
SCENES = {
    # the whole enzyme turning slowly, the heme and the ligand glowing inside
    "enzyme": (8.0, {"highlight": [HEME, LIGAND], "protein_alpha": 0.5, "particles": 90, "fstop": 1.6, "lens": 50,
                     "camera": {"start": {"azimuth": 20, "elevation": 18, "radius": 135}, "end": {"azimuth": 110, "elevation": 22, "radius": 112}}}),
    # the pocket: the nitrogen of the ligand over the iron of the heme, the lining residues in teal, a slow approach
    "pocket": (9.0, {"highlight": [LINING, HEME, LIGAND], "protein_alpha": 0.22, "particles": 50, "fstop": 2.0, "lens": 60,
                     "camera": {"start": {"azimuth": 40, "elevation": 28, "radius": 70}, "end": {"azimuth": 105, "elevation": 22, "radius": 46}}}),
}


def render(name: str, seconds: float, extra: dict, frames: int | None) -> int:
    total = frames or round(seconds * FPS)
    out = CLIPS / name
    if out.exists():
        shutil.rmtree(out)
    config = {"backend": "molecularnodes", "structure": str(STRUCTURE), "protein_selection": "protein", "anchors": ANCHORS, "focus_anchor": "fe", "frames": total, "fps": FPS,
              "resolution": list(SIZE), "samples": 40, "output": str(out), **extra}
    path = CLIPS / f"{name}.json"
    CLIPS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))
    subprocess.run([BLENDER, "-b", "-P", str(SCENE), "--", "--config", str(path)], check=True, stdout=subprocess.DEVNULL)
    path.unlink()
    return total


def write_index(done: dict[str, int]) -> None:
    entries = {}
    if INDEX.exists():  # a run with --only keeps the entries of the clips it did not render
        text = INDEX.read_text()
        entries = {k: v for k, v in json.loads(text[text.index("{") : text.rindex("}") + 1]).items() if k not in done}
    for name, total in done.items():
        raw = json.loads((CLIPS / name / "anchors.json").read_text())
        anchors = {label: [[round(raw[str(f)][label][0], 4), round(1 - raw[str(f)][label][1], 4)] for f in range(1, total + 1)] for label in ANCHORS}
        entries[name] = {"pattern": f"clips/{name}/f_%04d.png", "frames": total, "fps": FPS, "size": list(SIZE), "anchors": anchors}
    INDEX.write_text("window.CLIPS = " + json.dumps(entries) + ";\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(SCENES))
    parser.add_argument("--frames", type=int, help="render only this many frames of each clip (a preview)")
    args = parser.parse_args()
    done = {}
    for name, (seconds, extra) in SCENES.items():
        if args.only and name not in args.only:
            continue
        done[name] = render(name, seconds, extra, args.frames)
        print(f"{name}: {done[name]} frames", flush=True)
    write_index(done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
