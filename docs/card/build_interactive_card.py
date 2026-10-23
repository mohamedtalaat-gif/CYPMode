# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The interactive card: ``python docs/card/build_interactive_card.py`` writes ``docs/card/interactive/index.html``, one self-contained page.

It is the card of the README (``build_card.py``) with the figures made interactive. GitHub removes scripts and frames from a README and does not let an image answer the pointer, so the
interactivity lives in this page (served from GitHub Pages, or opened from a file) and the README shows the static card that links to it. The page holds the enzyme as a structure that can be rotated
and zoomed, the six tiles that load their complex into the viewer when clicked, the distance from the heme iron to the coordinating nitrogen measured on the loaded coordinates, labels on hover, the
numbers of the card with their source, and the dark and light themes. Everything is inside the file: the structures, the drawings of the compounds, the numbers and the viewer (3Dmol.js, BSD-3-Clause,
``docs/card/vendor``); the page makes no request when it is opened.

Two modes, as for the card. ``public`` (the default) holds crystal structures of the PDB only: the hero is CYP3A4 with fluconazole (6MA7) and the six tiles are public complexes with imidazole and pyridine
ligands. ``--full`` is built only after the deadline, by ``scripts/release_after_deadline.py``, and holds the submitted structures (the hero and six of the twenty poses, as on the card).

The numbers are read from ``docs/video/facts.py`` (the files that feed the card and the film), the structures from ``data/structure_track/inputs/pdb_cache``, the compounds from
``data/structure_track/inputs/restraint_expansion_ligands.csv``. The output is deterministic, so that the pre-submission gate can compare it with the content recorded in ``audit/artifacts.json``.
"""

import hashlib
import html
import json
import re
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "docs/figures/gen"), str(ROOT / "docs/video"), str(ROOT / "docs/video/anime"), str(HERE)]

import build_card as card
import facts as facts_module
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from cypmode.validation.build_input import heme_donor_atom

PDB_CACHE = ROOT / "data/structure_track/inputs/pdb_cache"
VIEWER = HERE / "vendor/3Dmol-min.js"
OUT = HERE / "interactive"
POCKET_SHELL = 4.0  # residues with an atom this close to the ligand are drawn as sticks, in angstrom
LIGAND_NAME = "LIG"  # the residue name given to the ligand in the page (a PDB code of five letters does not fit a PDB-format residue name)


def structure_entry(source: Path, ligand_code: str) -> dict:
    """The chain that holds the ligand, as PDB text with only its protein, the heme and the ligand (renamed LIG), the iron and the donor nitrogen, and the residues near the ligand."""
    st = gemmi.read_structure(str(source))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    model = st[0]

    def n_amino(chain) -> int:
        return sum(1 for r in chain if (info := gemmi.find_tabulated_residue(r.name)) and info.is_amino_acid())

    ligand_chain = next(chain for chain in model if any(residue.name == ligand_code for residue in chain))
    protein_chain = ligand_chain if n_amino(ligand_chain) >= 50 else max(model, key=n_amino)  # a complex file keeps the ligand and the heme either in the chain of the protein or in chains of their own
    keep = {protein_chain.name, ligand_chain.name} | ({chain.name for chain in model if any(r.name == "HEM" for r in chain)} if not any(r.name == "HEM" for r in protein_chain) else set())
    for name in [chain.name for chain in model if chain.name not in keep]:
        model.remove_chain(name)
    for chain in model:
        for i in reversed(range(len(chain))):
            residue = chain[i]
            if residue.name == ligand_code:
                residue.name = LIGAND_NAME
            elif residue.name != "HEM" and not ((info := gemmi.find_tabulated_residue(residue.name)) and info.is_amino_acid() and chain.name == protein_chain.name):
                del chain[i]
    iron = next(np.array(a.pos.tolist()) for chain in model for r in chain if r.name == "HEM" for a in r if a.element.name == "Fe")
    ligand = [(a, np.array(a.pos.tolist())) for chain in model for r in chain if r.name == LIGAND_NAME for a in r]
    nitrogens = [(float(np.linalg.norm(xyz - iron)), xyz) for a, xyz in ligand if a.element.name == "N"]
    distance, donor = min(nitrogens, key=lambda item: item[0])
    near = sorted({r.seqid.num for r in model[protein_chain.name] if r.name not in (LIGAND_NAME, "HEM") for a in r for _, xyz in ligand if float(np.linalg.norm(a.pos.tolist() - xyz)) <= POCKET_SHELL})
    text = "\n".join(line for line in st.make_pdb_string().splitlines() if line.startswith(("ATOM", "HETATM")))
    return {"pdb": text, "fe": [round(float(v), 3) for v in iron], "donor": [round(float(v), 3) for v in donor], "distance": round(distance, 2), "near": near}


def sketch(smiles: str, theme: str, donor: int | None) -> str:
    """The compound in 2D as an SVG without a background, the donor nitrogen circled."""
    colours = card.THEMES[theme]
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    canvas = rdMolDraw2D.MolDraw2DSVG(300, 240)
    options = canvas.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2.4
    options.setAtomPalette({-1: colours["bond"], 7: colours["n"], 8: colours["o"], 16: colours["s"], 9: colours["bond"], 17: colours["bond"], 35: colours["o"]})
    atoms = [donor] if donor is not None else []
    canvas.DrawMolecule(mol, highlightAtoms=atoms, highlightAtomColors={a: colours["hl"] for a in atoms}, highlightAtomRadii={a: 0.55 for a in atoms})
    canvas.FinishDrawing()
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", canvas.GetDrawingText())
    return re.sub(r"(<svg[^>]*?)\s+width='\d+px'\s+height='\d+px'", r"\1", svg, count=1)


def entries(mode: str = "public") -> list[dict]:
    """The hero first, then the six tiles; each with its structure, its two drawings and its caption."""
    if mode == "full":
        smiles = pd.read_csv(card.SMILES).set_index("Molecule_Name")["SMILES"]
        out = [{"id": card.HERO, "code": "LIG", "title": f"predicted ligand {card.HERO}", "caption": card.HERO, "sketch": None, **structure_entry(card.FINAL / f"{card.HERO}.pdb", "LIG")}]
        for name in card.STRIP:
            out.append({"id": name, "code": "LIG", "title": name, "caption": name, "sketch": {theme: sketch(smiles[name], theme, card.coordinating_n(name, smiles[name])) for theme in card.THEMES},
                        **structure_entry(card.FINAL / f"{name}.pdb", "LIG")})
        return out
    crystals = pd.read_csv(card.EXPANSION).set_index("comp_id")
    hero_pdb, hero_code = card.PUBLIC_HERO
    out = [{"id": hero_pdb, "code": hero_code, "title": f"fluconazole (PDB {hero_pdb})", "caption": f"{hero_code} &middot; PDB {hero_pdb}", "sketch": None, **structure_entry(PDB_CACHE / f"{hero_pdb}.cif", hero_code)}]
    for code, pdb_id in card.PUBLIC_STRIP:
        smiles = crystals.loc[code, "smiles"]
        donor = (heme_donor_atom(smiles) or (None,))[0]
        out.append({"id": pdb_id, "code": code, "title": f"{code} (PDB {pdb_id})", "caption": f"{code} &middot; PDB {pdb_id}",
                    "sketch": {theme: sketch(smiles, theme, donor) for theme in card.THEMES}, **structure_entry(PDB_CACHE / f"{pdb_id}.cif", code)})
    return out


PAGE = r"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CYPMode card</title>
<style>
:root{--page:#0a0f1e;--ink:#eef3ff;--muted:#9fb0cf;--panel:#0d1426;--line:#26324f;--accent:#2ee6c8;--chip:rgba(46,230,200,.12)}
:root[data-theme=light]{--page:#f4f7fd;--ink:#12203a;--muted:#566684;--panel:#fff;--line:#d5deee;--accent:#0e9f8a;--chip:rgba(14,159,138,.1)}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--page);color:var(--ink);font-family:'Avenir Next','Helvetica Neue',Arial,sans-serif}
.card{max-width:1400px;margin:0 auto;padding:28px 24px 40px;display:grid;gap:18px;grid-template-columns:minmax(0,5fr) minmax(0,7fr);grid-template-areas:"head view" "stats view" "strip strip" "note note"}
@media(max-width:900px){.card{grid-template-columns:1fr;grid-template-areas:"head" "view" "stats" "strip" "note"}}
header{grid-area:head;container-type:inline-size;min-width:0}.eyebrow{font-size:13px;letter-spacing:4px;font-weight:700;color:var(--accent)}
h1{margin:6px 0 0;font-size:min(19cqw,92px);line-height:1;font-weight:800;font-style:italic;letter-spacing:-3px}
.lede{margin:12px 0 0;font-size:19px;line-height:1.4;color:var(--muted)}
.chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}.chip{padding:7px 16px;border-radius:999px;background:var(--chip);border:1.5px solid var(--accent);font-size:15px;font-weight:700}
.stats{grid-area:stats;display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));align-content:start}
.stat{all:unset;display:block;cursor:help;padding:10px 12px;border-radius:14px;border:1.5px solid var(--line);background:var(--panel)}
.stat:focus-visible,.tile:focus-visible,.bar button:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.num{font-size:clamp(17px,2.2vw,22px);font-weight:800;color:var(--accent);overflow-wrap:anywhere}.lab{font-size:13px;line-height:1.3;color:var(--muted);margin-top:2px}
.view{grid-area:view;position:relative;border-radius:22px;overflow:hidden;border:1.5px solid var(--line);background:var(--panel);min-height:420px}
#v{width:100%;height:min(62vh,560px);min-height:420px;position:relative}
.bar{position:absolute;left:12px;top:12px;display:flex;flex-wrap:wrap;gap:6px;z-index:5}
.bar button{padding:6px 12px;border-radius:10px;border:1.5px solid var(--line);background:var(--panel);color:var(--ink);font:inherit;font-size:13px;font-weight:600;cursor:pointer}
.bar button[aria-pressed=true]{border-color:var(--accent);color:var(--accent)}
.reading{position:absolute;left:12px;bottom:10px;right:12px;z-index:5;font-size:14px;font-weight:700;pointer-events:none}.reading span{color:var(--accent)}
.strip{grid-area:strip;display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
.tile{all:unset;box-sizing:border-box;cursor:pointer;border-radius:16px;border:1.5px solid var(--line);background:var(--panel);overflow:hidden;display:flex;flex-direction:column}
.tile[aria-pressed=true]{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent)}.tile:hover{border-color:var(--accent)}
.tile .sk{height:110px;display:flex;align-items:center;justify-content:center;padding:6px}.tile .sk svg{width:100%;height:100%}
.tile .cap{padding:8px 12px;border-top:1.5px solid var(--line);font-size:14px}.tile .cap b{display:block}.tile .cap span{color:var(--accent);font-weight:700}
.note{grid-area:note;font-size:14px;line-height:1.45;color:var(--muted)}
.theme{position:fixed;right:14px;top:12px;z-index:9;padding:6px 12px;border-radius:10px;border:1.5px solid var(--line);background:var(--panel);color:var(--ink);font:inherit;font-size:13px;font-weight:600;cursor:pointer}
.sk-light{display:none}:root[data-theme=light] .sk-dark{display:none}:root[data-theme=light] .sk-light{display:block}
</style></head><body>
<button class="theme" id="theme" aria-label="Switch between the dark and the light theme">Theme</button>
<main class="card">
<header><div class="eyebrow">OPENADMET CYP INHIBITION BLIND CHALLENGE</div><h1>CYPMode</h1>
<p class="lede">Time-dependent CYP inhibition and CYP3A4&ndash;ligand structures, predicted by models whose every addition passed a pre-registered gate.</p>
<div class="chips"><span class="chip">TDI classification &middot; CYP3A4 and CYP2D6</span><span class="chip">Structure prediction &middot; CYP3A4 &middot; 20 ligands</span></div></header>
<section class="stats" aria-label="Numbers of the project">__STATS__</section>
<section class="view"><div class="bar" role="toolbar" aria-label="Viewer controls"><button id="spin" aria-pressed="false">Spin</button><button id="focus">Zoom to the ligand</button><button id="whole">Whole enzyme</button><button id="cart" aria-pressed="true">Cartoon</button></div>
<div id="v" role="img" aria-label="Interactive three-dimensional structure of CYP3A4 with a bound ligand"></div><div class="reading" id="reading"></div></section>
<nav class="strip" id="strip" aria-label="Complexes: choose one to load it">__TILES__</nav>
<p class="note">__NOTE__ Drag to rotate, scroll to zoom, hover an atom of the ligand or the heme to name it. Structures: wwPDB. Drawings: RDKit. Viewer: 3Dmol.js (BSD-3-Clause), bundled in this file; the page makes no request.</p>
</main>
<script>__VIEWER__</script>
<script>
const ENTRIES = __DATA__;
const PALETTE = {dark: ["#2c4a9e", "#4f80d9", "#6ac7eb", "#dbf2ff"], light: ["#2c4a9e", "#4f80d9", "#3aa6cf", "#8ec9e8"]};
const docEl = document.documentElement;
const viewer = $3Dmol.createViewer(document.getElementById("v"), {backgroundAlpha: 0});
let current = 0, cartoon = true, spinning = false;
const theme = () => docEl.dataset.theme;
const lerp = (a, b, t) => a + (b - a) * t;
function colour(resi, lo, hi) {
  const stops = PALETTE[theme()].map(h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  const x = Math.min(1, Math.max(0, (resi - lo) / Math.max(1, hi - lo))) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), t = x - i;
  return "#" + [0, 1, 2].map(k => Math.round(lerp(stops[i][k], stops[i + 1][k], t)).toString(16).padStart(2, "0")).join("");
}
function load(i) {
  current = i;
  const e = ENTRIES[i];
  viewer.clear();
  viewer.addModel(e.pdb, "pdb");
  const resis = viewer.selectedAtoms({hetflag: false}).map(a => a.resi), lo = Math.min(...resis), hi = Math.max(...resis);
  viewer.setStyle({hetflag: false}, cartoon ? {cartoon: {colorfunc: a => colour(a.resi, lo, hi), opacity: 0.92}} : {line: {colorfunc: a => colour(a.resi, lo, hi)}});
  viewer.addStyle({resi: e.near, hetflag: false}, {stick: {radius: 0.12, colorscheme: "whiteCarbon"}});
  viewer.setStyle({resn: "HEM"}, {stick: {radius: 0.2, colorscheme: "greenCarbon"}});
  viewer.setStyle({resn: "__LIG__"}, {stick: {radius: 0.24, colorscheme: "orangeCarbon"}});
  viewer.addStyle({elem: "Fe"}, {sphere: {radius: 0.8, color: "#ff7a00"}});
  const mid = {x: (e.fe[0] + e.donor[0]) / 2, y: (e.fe[1] + e.donor[1]) / 2, z: (e.fe[2] + e.donor[2]) / 2};
  viewer.addCylinder({start: {x: e.fe[0], y: e.fe[1], z: e.fe[2]}, end: {x: e.donor[0], y: e.donor[1], z: e.donor[2]}, radius: 0.07, dashed: true, dashLength: 0.25, gapLength: 0.15, color: "#2ee6c8", fromCap: 1, toCap: 1});
  viewer.addLabel(e.distance.toFixed(2) + " Å", {position: mid, fontSize: 13, fontColor: "#06251f", backgroundColor: "#2ee6c8", backgroundOpacity: 0.95, showBackground: true, inFront: true});
  viewer.setHoverable({resn: ["__LIG__", "HEM"]}, true,
    (atom, v) => { if (!atom.label) { atom.label = v.addLabel(atom.resn === "__LIG__" ? "ligand " + atom.atom : "heme " + atom.atom, {position: atom, fontSize: 12, backgroundColor: "#000", backgroundOpacity: 0.78, fontColor: "#fff", showBackground: true}); } },
    (atom, v) => { if (atom.label) { v.removeLabel(atom.label); delete atom.label; } });
  document.querySelectorAll(".tile").forEach((t, k) => t.setAttribute("aria-pressed", String(k === i)));
  document.getElementById("reading").innerHTML = e.title + " &middot; iron&ndash;nitrogen <span>" + e.distance.toFixed(2) + " Å</span>, measured on the coordinates shown";
  viewer.render();
  focus(i === 0 ? "whole" : "ligand");
}
function focus(what) {
  if (what === "ligand") { viewer.zoomTo({resn: "__LIG__"}); viewer.zoom(0.55, 400); } else { viewer.zoomTo(); }
  viewer.render();
}
document.getElementById("strip").addEventListener("click", ev => { const t = ev.target.closest(".tile"); if (t) load(Number(t.dataset.i)); });
document.getElementById("spin").onclick = ev => { spinning = !spinning; viewer.spin(spinning ? "y" : false); ev.currentTarget.setAttribute("aria-pressed", String(spinning)); };
document.getElementById("focus").onclick = () => focus("ligand");
document.getElementById("whole").onclick = () => focus("whole");
document.getElementById("cart").onclick = ev => { cartoon = !cartoon; ev.currentTarget.setAttribute("aria-pressed", String(cartoon)); ev.currentTarget.textContent = cartoon ? "Cartoon" : "Lines"; load(current); };
const stored = (() => { try { return localStorage.getItem("cypmode-card-theme"); } catch (e) { return null; } })();
const query = new URLSearchParams(location.search).get("theme");
docEl.dataset.theme = query || stored || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
document.getElementById("theme").onclick = () => { docEl.dataset.theme = theme() === "dark" ? "light" : "dark"; try { localStorage.setItem("cypmode-card-theme", theme()); } catch (e) {} load(current); };
const wanted = ENTRIES.findIndex(e => e.id === location.hash.slice(1).toUpperCase());
load(wanted >= 0 ? wanted : 0);
window.addEventListener("resize", () => viewer.resize());
</script></body></html>
"""


def tile_html(entry: dict, index: int) -> str:
    sketches = "" if entry["sketch"] is None else "".join(f'<span class="sk-{theme}">{svg}</span>' for theme, svg in entry["sketch"].items())
    sk = f'<div class="sk">{sketches}</div>' if sketches else '<div class="sk"><b>Whole enzyme</b></div>'
    return (f'<button class="tile" data-i="{index}" aria-pressed="false" aria-label="Load {html.escape(entry["title"])}">{sk}'
            f'<div class="cap"><b>{entry["caption"]}</b><span>Fe&ndash;N {entry["distance"]:.2f} &Aring;</span></div></button>')


def page(mode: str = "public") -> str:
    facts = facts_module.load()
    full = mode == "full"
    stats = [(f"{facts.auc['CYP3A4']:.3f} / {facts.auc['CYP2D6']:.3f}", "nested-CV AUC, CYP3A4 / CYP2D6", "The area under the ROC curve of the final stack in the nested cross-validation (repeats of outer folds, every choice inside the loop), at the stricter confidence cut."),
             (f"{facts.poses - facts.pose_failures} / {facts.poses}", "poses passing all 22 PoseBusters checks", "How many of the submitted structure-track poses pass every PoseBusters check; the poses themselves are published after the challenge closes."),
             (f"{facts.gates}", f"ideas judged by a pre-registered gate, {facts.gates - facts.gates_not_met} met", "Every idea tried was given its pass rule before its result existed; this counts them and those that passed.")]
    stat_html = "".join(f'<button class="stat" title="{html.escape(why)}" aria-label="{html.escape(label)}: {html.escape(n)}. {html.escape(why)}"><div class="num">{n}</div><div class="lab">{html.escape(label)}</div></button>' for n, label, why in stats)
    data = entries(mode)
    note = (f"Six of the {facts.poses} submitted poses, picked by hand: the circled nitrogen of each drawing is the one nearest the heme iron in the submitted structure, the dashed line in the viewer joins it to the iron, "
            f"and all {facts.poses - facts.pose_failures} poses pass the PoseBusters checks."
            if full else "Public crystal structures of CYP3A4 from the PDB: the enzyme with fluconazole and six complexes with imidazole and pyridine ligands; the circled nitrogen of each drawing is the Type II donor, "
            f"and the dashed line in the viewer joins it to the heme iron. The project's own {facts.poses} predicted structures are published after the challenge closes.")
    payload = json.dumps([{k: v for k, v in e.items() if k not in ("sketch", "caption")} for e in data], separators=(",", ":")).replace("</", "<\\/")
    text = (PAGE.replace("__LIG__", LIGAND_NAME).replace("__STATS__", stat_html).replace("__TILES__", "".join(tile_html(e, i) for i, e in enumerate(data)))
            .replace("__NOTE__", note).replace("__DATA__", payload))
    return text.replace("__VIEWER__", VIEWER.read_text())  # the library last, so that nothing in it is touched by the placeholders above


def main() -> int:
    mode = "full" if "--full" in sys.argv else "public"
    OUT.mkdir(exist_ok=True)
    text = page(mode)
    target = OUT / "index.html"
    target.write_text(text)
    (OUT / ".nojekyll").write_text("")
    from cypmode.audit.artifacts import record

    record("interactive_card", [target], {"html_sha256": hashlib.sha256(text.encode()).hexdigest(), "mode": mode})
    print(f"wrote {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
