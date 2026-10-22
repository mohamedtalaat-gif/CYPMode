# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The card at the top of the README: ``python docs/card/build_card.py`` writes ``cypmode_card-dark.png`` and ``cypmode_card-light.png`` next to this file.

Everything on the card is read from the repository's own files when it is built: the pictures of the enzyme are PyMOL renders of the submitted structures (``final/_pdb_files``), the compounds are
drawn by RDKit from their SMILES with the nitrogen that coordinates the heme iron marked, the iron-nitrogen distances are measured on the submitted coordinates, and the numbers come from
``docs/video/facts.py`` (the same files that feed the animation). PyMOL and Google Chrome are needed; renders are cached in ``docs/figures/_pymol_cache``.

Two modes. ``public`` (the default) shows only public structures: the hero is the crystal structure of CYP3A4 with fluconazole (PDB 6MA7) and the six tiles are crystal complexes of the PDB with
imidazole and pyridine ligands, so that nothing on the card is a predicted structure of a blinded ligand; it is the card that goes into the repository before the challenge closes. ``--full`` draws
the submitted structures (the hero and six of the twenty poses with the iron-nitrogen distances measured on them); it is built only after the deadline, by ``scripts/release_after_deadline.py``.
"""

import base64
import hashlib
import html
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/figures/gen"), str(ROOT / "docs/video"), str(ROOT / "docs/video/anime")]

import facts as facts_module
import pymol_scene
from chrome import Chrome
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from cypmode.validation.build_input import heme_donor_atom

FINAL = ROOT / "data/structure_track/outputs/cyp3a4/final/_pdb_files"
SMILES = ROOT / "data/structure_track/inputs/structure_track_test_blinded.csv"
BUILD = HERE / "_build"
CARD = (1920, 960)
SCALE = 1.25
HERO = "OCNT-2395436"  # the ligand that the film follows: a Type II donor with a measured TDI value
# Six of the twelve submitted poses whose ligand has a Type II donor nitrogen, picked by hand and not at random: the ligand that the film follows (OCNT-2395436), the ligand whose restraint changes the
# pose most on the benchmark of the film (OCNT-2395712), and four that show imidazole, triazole and pyridine donors in different scaffolds. The card says so; ``docs/figures`` has the gallery of all 20.
STRIP = ("OCNT-2395436", "OCNT-2395712", "OCNT-2313300", "OCNT-2312554", "OCNT-2395503", "OCNT-2535519")

PUBLIC_HERO = ("6MA7", "TPF")  # CYP3A4 with fluconazole, crystal structure from the PDB
PUBLIC_STRIP = (("CL6", "8SPD"), ("A1ASP", "9BVB"), ("J9K", "4D78"), ("D7M", "6BD5"), ("A1ASS", "9BV8"), ("80K", "6BD6"))  # (ligand code, PDB entry): one Type II donor each, imidazole and pyridine
EXPANSION = ROOT / "data/structure_track/inputs/restraint_expansion_ligands.csv"  # SMILES and the deposited iron-ligand distance of the public crystal ligands

THEMES = {
    "dark": {"page": "linear-gradient(135deg, #0a0f1e 0%, #111b36 60%, #0c2433 100%)", "ink": "#eef3ff", "muted": "#9fb0cf", "panel": "#0d1426", "line": "#26324f", "accent": "#2ee6c8", "chip": "rgba(46,230,200,0.12)",
             "bond": (0.93, 0.95, 1.0), "n": (0.36, 0.62, 1.0), "o": (1.0, 0.42, 0.42), "s": (0.95, 0.82, 0.29), "hl": (0.18, 0.9, 0.78)},
    "light": {"page": "linear-gradient(135deg, #ffffff 0%, #f1f5fd 60%, #e6f4f2 100%)", "ink": "#12203a", "muted": "#566684", "panel": "#ffffff", "line": "#d5deee", "accent": "#0e9f8a", "chip": "rgba(14,159,138,0.10)",
              "bond": (0.1, 0.15, 0.3), "n": (0.14, 0.34, 0.77), "o": (0.77, 0.23, 0.23), "s": (0.72, 0.52, 0.04), "hl": (0.05, 0.62, 0.54)},
}


def fe_n_distance(pdb: Path) -> float:
    """Distance from the iron to the nearest nitrogen of the ligand (residue LIG) in a submitted structure, in angstrom."""
    iron, nitrogen = [], []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if element == "FE":
                iron.append(xyz)
            elif element == "N" and line[17:20].strip() == "LIG":
                nitrogen.append(xyz)
    return float(min(np.linalg.norm(n - iron[0]) for n in nitrogen))


HERO_BODY = '''
cmd.load({path!r}, "s")
cmd.remove("solvent")
cmd.hide("everything")
cmd.select("lig", "resn {ligand}"); cmd.select("hem", "resn HEM")
cmd.set_color("g1", [0.17, 0.29, 0.62]); cmd.set_color("g2", [0.31, 0.50, 0.85]); cmd.set_color("g3", [0.42, 0.78, 0.92]); cmd.set_color("g4", [0.86, 0.95, 1.0])
cmd.show("cartoon", "polymer"); cmd.spectrum("resi", "g1 g2 g3 g4", "polymer"); cmd.set("cartoon_transparency", 0.1); cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_oval_length", 1.6)
cmd.show("sticks", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.set("stick_radius", 0.3, "hem")
cmd.show("spheres", "elem FE"); cmd.set("sphere_scale", 1.0, "elem FE"); cmd.color(PAL["iron"], "elem FE")
cmd.show("sticks", "lig"); cmd.color(PAL["ligand"], "lig and elem C"); cmd.set("stick_radius", 0.4, "lig")
cmd.show("spheres", "lig"); cmd.set("sphere_scale", 0.5, "lig"); cmd.set("sphere_transparency", 0.7, "lig"); cmd.color(PAL["ligand"], "lig and elem C")
cmd.set("ray_shadow", "on"); cmd.set("ambient", 0.45); cmd.set("depth_cue", 1); cmd.set("fog_start", 0.25); cmd.set("light_count", 3)
cmd.orient("polymer"); cmd.turn("y", 25); cmd.turn("x", -12); cmd.center("lig or hem"); cmd.create("halo", "lig or elem FE"); cmd.hide("everything", "halo"); cmd.show("spheres", "halo"); cmd.set("sphere_scale", 2.6, "halo"); cmd.set("sphere_transparency", 0.88, "halo"); cmd.color(PAL["ligand"], "halo")\ncmd.zoom("polymer", {zoom})
'''


def hero(theme: str, mode: str = "public") -> pymol_scene.Scene:
    path, ligand, key, zoom = (FINAL / f"{HERO}.pdb", "LIG", "card_hero3", -24) if mode == "full" else (pymol_scene.cif(PUBLIC_HERO[0]), PUBLIC_HERO[1], f"card_hero_crystal_{PUBLIC_HERO[0]}_mid", -10)
    return pymol_scene.render(key, HERO_BODY.format(path=str(path), ligand=ligand, zoom=zoom), (2008, 1088), theme, {"fe": "elem FE", "ligand": "lig"}, files=(path,))


def tile(name: str, theme: str) -> pymol_scene.Scene:
    return pymol_scene.predicted_pocket(FINAL / f"{name}.pdb", theme, (576, 400), key=f"card_tile_{name}")


def coordinating_n(name: str, smiles: str) -> int | None:
    """The index (in the SMILES) of the ligand nitrogen nearest to the iron in the submitted pose, or None when the pose cannot be matched to the SMILES unambiguously."""
    from rdkit.Chem import AllChem

    lines = (FINAL / f"{name}.pdb").read_text().splitlines()
    iron = [np.array([float(x[30:38]), float(x[38:46]), float(x[46:54])]) for x in lines if x.startswith(("ATOM", "HETATM")) and (x[76:78].strip() or "").upper() == "FE"]
    block = "\n".join(x for x in lines if x.startswith("HETATM") and x[17:20].strip() == "LIG") + "\nEND\n"
    template, pose = Chem.MolFromSmiles(smiles), Chem.MolFromPDBBlock(block, removeHs=True, proximityBonding=True)
    if not iron or pose is None:
        return None
    pose = AllChem.AssignBondOrdersFromTemplate(template, pose)
    conformer, nearest = pose.GetConformer(), set()
    for match in pose.GetSubstructMatches(template, uniquify=False, maxMatches=64):
        distances = {i: float(np.linalg.norm(np.array(conformer.GetAtomPosition(match[i])) - iron[0])) for i, a in enumerate(template.GetAtoms()) if a.GetSymbol() == "N"}
        nearest.add(min(distances, key=distances.get))
    return nearest.pop() if len(nearest) == 1 else None


def drawing(name: str, theme: str, smiles: str, donor: int | None = None) -> Path:
    """The compound in 2D on a transparent background, the nitrogen that coordinates the iron in the submitted pose circled (``donor``, an atom index of the SMILES)."""
    colours = THEMES[theme]
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    canvas = rdMolDraw2D.MolDraw2DCairo(300, 240)
    options = canvas.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2.4
    options.setAtomPalette({-1: colours["bond"], 7: colours["n"], 8: colours["o"], 16: colours["s"], 9: colours["bond"], 17: colours["bond"], 35: colours["o"]})
    atoms = [donor] if donor is not None else []
    canvas.DrawMolecule(mol, highlightAtoms=atoms, highlightAtomColors={a: colours["hl"] for a in atoms}, highlightAtomRadii={a: 0.55 for a in atoms})
    canvas.FinishDrawing()
    path = BUILD / f"draw_{name}_{theme}.png"
    path.write_bytes(canvas.GetDrawingText())
    return path


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def card_html(theme: str, mode: str = "public") -> str:
    c = THEMES[theme]
    facts = facts_module.load()
    full = mode == "full"
    smiles = pd.read_csv(SMILES).set_index("Molecule_Name")["SMILES"]
    crystals = pd.read_csv(EXPANSION).set_index("comp_id")
    scene = hero(theme, mode)
    panel_w, panel_h = 1004, 544
    labels = ""
    for key, text, dx, dy in (("fe", "heme iron", -150, 70), ("ligand", f"predicted ligand {HERO}" if full else "fluconazole (PDB 6MA7)", 40, -80)):
        x, y = scene.anchors[key][0] / 2008 * panel_w, scene.anchors[key][1] / 1088 * panel_h
        labels += (f'<div class="callout" style="left:{x + dx:.0f}px;top:{y + dy:.0f}px">{html.escape(text)}</div>'
                   f'<svg class="leader" width="{panel_w}" height="{panel_h}"><line x1="{x:.0f}" y1="{y:.0f}" x2="{x + dx + (0 if dx > 0 else 100):.0f}" y2="{y + dy + 14:.0f}" stroke="{c["accent"]}" stroke-width="2"/>'
                   f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6" fill="none" stroke="{c["accent"]}" stroke-width="2"/></svg>')
    tiles = ""
    for entry in (STRIP if full else PUBLIC_STRIP):
        if full:
            name = entry
            pocket, sketch, label, distance = tile(name, theme), drawing(name, theme, smiles[name], coordinating_n(name, smiles[name])), name, fe_n_distance(FINAL / f"{name}.pdb")
        else:
            comp, pdb = entry
            row = crystals.loc[comp]
            pocket, sketch = pymol_scene.pocket(pdb, comp, theme, (576, 400)), drawing(f"{comp}_{pdb}", theme, row["smiles"], (heme_donor_atom(row["smiles"]) or (None,))[0])
            label, distance = f"{comp} &middot; PDB {pdb}", float(row["min_fe_distance"])
        tiles += (f'<div class="tile"><img class="pocket" src="{data_uri(pocket.png)}"><img class="sketch" src="{data_uri(sketch)}">'
                  f'<div class="cap"><b>{label}</b><span>Fe-N {distance:.2f} &Aring;</span></div></div>')
    stats = [(f"{facts.auc['CYP3A4']:.3f} / {facts.auc['CYP2D6']:.3f}", "nested-CV AUC, CYP3A4 / CYP2D6"), (f"{facts.poses - facts.pose_failures} / {facts.poses}", "poses passing all 22 PoseBusters checks"),
             (f"{facts.gates}", f"ideas judged by a pre-registered gate, {facts.gates - facts.gates_not_met} met")]
    note = (f"Six of the {facts.poses} submitted poses, picked by hand: the circled nitrogen is the one nearest the heme iron in the submitted structure, and all {facts.poses - facts.pose_failures} poses pass the 22 PoseBusters checks."
            if full else "Public crystal structures of CYP3A4 from the PDB: the enzyme with fluconazole and six complexes with imidazole and pyridine ligands; the circled nitrogen is the Type II donor and the "
            f"distance is the deposited one. The project's own {facts.poses} predicted structures are published after the challenge closes.")
    stat_html = "".join(f'<div class="stat"><div class="num">{n}</div><div class="lab">{html.escape(t)}</div></div>' for n, t in stats)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}} html,body{{margin:0;background:#000}}
.card{{width:{CARD[0]}px;height:{CARD[1]}px;padding:56px;position:relative;overflow:hidden;background:{c['page']};color:{c['ink']};font-family:'Avenir Next','Helvetica Neue',Arial,sans-serif}}
.eyebrow{{font-size:22px;letter-spacing:5px;font-weight:700;color:{c['accent']}}}
h1{{margin:6px 0 0;font-size:142px;line-height:1;font-weight:800;font-style:italic;letter-spacing:-3px}}
.lede{{margin-top:14px;font-size:27px;line-height:1.36;color:{c['muted']};width:760px}}
.chips{{margin-top:20px;display:flex;flex-direction:column;gap:10px;width:760px}}
.chip{{display:inline-block;padding:9px 20px;border-radius:999px;background:{c['chip']};border:1.5px solid {c['accent']};font-size:22px;font-weight:700;width:max-content}}
.stats{{position:absolute;left:56px;top:508px;width:760px;display:flex;gap:20px}} .stat:nth-child(1){{width:300px}} .stat:nth-child(2){{width:190px}} .stat:nth-child(3){{width:230px}}
.stat .num{{font-size:42px;font-weight:800;color:{c['accent']};white-space:nowrap}} .stat .lab{{font-size:17px;line-height:1.3;color:{c['muted']};margin-top:2px}}
.hero{{position:absolute;left:860px;top:56px;width:{panel_w}px;height:{panel_h}px;border-radius:26px;overflow:hidden;border:1.5px solid {c['line']};background:{c['panel']}}}
.hero img{{width:100%;height:100%;display:block}}
.callout{{position:absolute;padding:8px 16px;border-radius:12px;background:{c['panel']};border:1.5px solid {c['accent']};font-size:21px;font-weight:700;color:{c['ink']}}}
.leader{{position:absolute;left:0;top:0}}
.strip{{position:absolute;left:56px;top:640px;width:1808px;display:flex;gap:16px}}
.tile{{position:relative;width:288px;height:264px;border-radius:20px;overflow:hidden;border:1.5px solid {c['line']};background:{c['panel']}}}
.tile .pocket{{width:288px;height:200px;display:block;object-fit:cover}}
.tile .sketch{{position:absolute;right:6px;top:6px;width:130px;height:104px;border-radius:12px;background:{c['panel']};opacity:0.96}}
.cap{{position:absolute;left:0;right:0;bottom:0;height:64px;padding:10px 16px;display:flex;flex-direction:column;justify-content:center;border-top:1.5px solid {c['line']}}}
.note{{position:absolute;left:56px;top:912px;width:1808px;font-size:16px;line-height:1.3;color:{c['muted']}}}
.cap b{{font-size:22px}} .cap span{{font-size:19px;color:{c['accent']};font-weight:700}}
</style></head><body><div class="card">
<div class="eyebrow">OPENADMET CYP INHIBITION BLIND CHALLENGE</div>
<h1>CYPMode</h1>
<div class="lede">Time-dependent CYP inhibition and CYP3A4&ndash;ligand structures, predicted by models whose every addition passed a pre-registered gate.</div>
<div class="chips"><span class="chip">TDI classification &middot; CYP3A4 and CYP2D6</span><span class="chip">Structure prediction &middot; CYP3A4 &middot; 20 ligands</span></div>
<div class="stats">{stat_html}</div>
<div class="hero"><img src="{data_uri(scene.png)}">{labels}</div>
<div class="strip">{tiles}</div>
<div class="note">{note}</div>
</div></body></html>"""


def main() -> int:
    mode = "full" if "--full" in sys.argv else "public"
    BUILD.mkdir(exist_ok=True)
    written, digests = [], {}
    with Chrome() as chrome:
        for theme in THEMES:
            page = BUILD / f"card-{theme}.html"
            source = card_html(theme, mode)
            digests[theme] = hashlib.sha256(source.encode()).hexdigest()
            page.write_text(source)
            chrome.open(page)
            time.sleep(1.0)
            chrome.call("Emulation.setDeviceMetricsOverride", width=CARD[0], height=CARD[1], deviceScaleFactor=SCALE, mobile=False)
            time.sleep(0.5)
            data = chrome.call("Page.captureScreenshot", format="png", clip={"x": 0, "y": 0, "width": CARD[0], "height": CARD[1], "scale": 1})["data"]
            target = HERE / f"cypmode_card-{theme}.png"
            target.write_bytes(base64.b64decode(data))
            written.append(target)
            print(f"wrote {target.name} ({target.stat().st_size / 1e6:.2f} MB)")
    from cypmode.audit.artifacts import record

    record("readme_card", written, {"html_sha256": digests, "mode": mode})
    return 0


if __name__ == "__main__":
    sys.exit(main())
