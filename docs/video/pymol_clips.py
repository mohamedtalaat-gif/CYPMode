# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Render the real-structure pictures of the film with PyMOL: ``python docs/video/pymol_clips.py``.

The pictures are drawn from coordinates: the enzyme with its heme and a bound ligand, the heme pocket of a Type II and of a Type I crystal structure, and the pose of a donor ligand
predicted without and with the iron-nitrogen restraint, in the same pocket. They are copied to ``docs/video/anime/assets/`` and described in ``assets.js`` (their size, the pixel position of
the iron, the ligand and the coordinating nitrogen, and the measured iron-nitrogen distances) for the film page to use.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "docs/figures/gen"))

import pymol_scene as ps

ASSETS = HERE / "anime" / "assets"
OUT = ROOT / "data/structure_track/outputs/out/cyp3a4"
EXAMPLE = "OCNT-2395436"  # the donor ligand followed through the film (its predicted pose is shown as it is)
RESTRAINT_EXAMPLE = "OCNT-2395712"  # a donor ligand whose unguided prediction misses the iron by more than 5 angstrom
SIZE = (1800, 1100)


def predicted(name: str) -> Path:
    return next((OUT / f"boltz_results_cyp3a4_{name}" / "predictions" / f"cyp3a4_{name}").glob("*_model_0.cif"))


def iron_nitrogen(path: Path) -> float:
    """The distance in angstrom from the heme iron to the nearest nitrogen atom of the ligand of a predicted complex."""
    model = MMCIFParser(QUIET=True).get_structure("x", str(path))[0]
    atoms = [a for a in model.get_atoms()]
    iron = next(a.coord for a in atoms if a.element.upper() == "FE")
    ligand_n = [a.coord for a in atoms if a.get_parent().get_resname().startswith("LIG") and a.element.upper() == "N"]
    return float(min(np.linalg.norm(c - iron) for c in ligand_n))


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    restrained, unguided = predicted(f"{RESTRAINT_EXAMPLE}-rs"), predicted(RESTRAINT_EXAMPLE)
    own = predicted(f"{EXAMPLE}-rs")
    scenes = {
        "enzyme": ps.overview("6MA7", "TPF", size=SIZE),
        "pocket_type_ii": ps.pocket("6MA7", "TPF", size=SIZE),
        "pocket_type_i": ps.pocket("6OOB", "MWS", size=SIZE),
        "cap_pose": ps.predicted_pocket(own, size=SIZE, key=f"official_restrained_{EXAMPLE}"),
        "example_restrained": ps.predicted_pocket(restrained, size=SIZE, key=f"official_restrained_{RESTRAINT_EXAMPLE}"),
        "example_unguided": ps.predicted_pocket(unguided, size=SIZE, key=f"official_unguided_{RESTRAINT_EXAMPLE}"),
        "example_compare": ps.pose_comparison(restrained, [(unguided, "0xff6b6b")], size=SIZE, key=f"restraint_vs_unguided_{RESTRAINT_EXAMPLE}"),
    }
    described = {}
    for name, scene in scenes.items():
        shutil.copyfile(scene.png, ASSETS / f"{name}.png")
        described[name] = {"file": f"assets/{name}.png", "size": list(SIZE), "anchors": scene.anchors}
    described["distances"] = {"ligand": RESTRAINT_EXAMPLE, "restrained": round(iron_nitrogen(restrained), 2), "unguided": round(iron_nitrogen(unguided), 2), "own": round(iron_nitrogen(own), 2), "own_ligand": EXAMPLE}
    (HERE / "anime" / "assets.js").write_text("window.ASSETS = " + json.dumps(described, indent=1) + ";\n")
    print(f"wrote {len(scenes)} pictures to {ASSETS}; iron-nitrogen distance of {RESTRAINT_EXAMPLE}: unguided {described['distances']['unguided']} A, restrained {described['distances']['restrained']} A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
