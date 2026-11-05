# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI, the structure track: what a complex file holds (11.1), noise on real coordinates (11.2), the input of a prediction (11.3), an RMSD (11.4), the stored poses of the benchmark (11.5), the engines (11.9), the submitted structures (11.10), the stereoisomers of a ligand (11.12)
and the experiments of the ledger behind the other sections."""

import altair as alt
import numpy as np
import pandas as pd

from cypmode.validation.build_input import build_boltz_yaml, find_coordinating_cysteine
from simulators import molecule, replay, store, structure


def c_11_1(mo):
    names = structure.structure_names()
    return {"name": mo.ui.dropdown(options=names, value="validation_ketoconazole" if "validation_ketoconazole" in names else names[0], label="Structure (the pocket of a prediction)"), "protein": mo.ui.checkbox(value=True, label="protein"), "heme": mo.ui.checkbox(value=True, label="heme"), "distance": mo.ui.checkbox(value=True, label="iron-nitrogen distance")}


def v_11_1(mo, v):
    text = structure.pdb_text(v["name"])
    atoms = structure.parse(text)
    contact = structure.iron_nitrogen(atoms)
    where = f"The nearest nitrogen of the ligand to the iron is {contact['atom']} at **{contact['distance']:.2f} A**." if contact else "No iron-nitrogen pair."
    return mo.vstack([mo.md(f"A file of a complex lists atoms with their coordinates, grouped in chains: the protein, the heme (the cofactor that holds the iron) and the ligand. Only the residues within 10 A of the ligand are kept here. {where} Drag to turn it, scroll to zoom."),
                      mo.ui.table(structure.residue_table(atoms), selection=None, pagination=False)])


def e_11_1(mo, v):
    return mo.iframe(structure.viewer_html(structure.pdb_text(v["name"]), bool(v["protein"]), bool(v["heme"]), show_distance=bool(v["distance"])), height="460px")


def c_11_2(mo):
    return {"name": mo.ui.dropdown(options=structure.structure_names(), value=structure.structure_names()[0], label="Structure"), "sigma": mo.ui.slider(start=0.0, stop=8.0, step=0.1, value=2.0, label="noise (angstrom)", show_value=True), "seed": mo.ui.number(start=0, stop=99, step=1, value=0, label="seed")}


def v_11_2(mo, v):
    xyz = structure.ligand_xyz(structure.parse(structure.pdb_text(v["name"])))
    now = structure.noisy(xyz, float(v["sigma"]), int(v["seed"]))
    frame = pd.concat([pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "state": "the structure"}), pd.DataFrame({"x": now[:, 0], "y": now[:, 1], "state": "with noise"})])
    return mo.vstack([mo.md(f"A co-folding model of the diffusion kind starts from atoms in random places and removes the noise step by step until a structure is left; it is trained by taking real structures, adding noise of every size and learning to remove it. Here the real coordinates of the ligand ({len(xyz)} heavy atoms) with a noise of {v['sigma']:.1f} A: "
                            f"the RMSD between the two is **{structure.rmsd(xyz, now):.2f} A** (about the noise times the square root of 3). This only draws the noise; the model that removes it is not run here."),
                      alt.Chart(frame).mark_circle(size=60, opacity=0.8).encode(x="x:Q", y="y:Q", color="state:N").properties(width=420, height=360)])


def c_11_3(mo):
    return {"smiles": mo.ui.text(value="Cc1ncc(n1CCO)c1ccccc1", label="Ligand (SMILES)", full_width=True), "restraint": mo.ui.dropdown(options=["none", "bond", "contact"], value="none", label="restraint of the donor nitrogen to the iron")}


def v_11_3(mo, v):
    sequence = (structure.HERE / "data/cyp3a4_sequence.txt").read_text().strip()
    cys = find_coordinating_cysteine(sequence)
    restraint = None if v["restraint"] == "none" else v["restraint"]
    try:
        text = build_boltz_yaml(sequence, v["smiles"], cys, predict_affinity=restraint is None, donor_restraint=restraint)
    except Exception as error:  # a SMILES that the builder cannot use is a message and not a traceback
        return mo.md(f"The input could not be built: {error}")
    short = "\n".join(line if len(line) < 100 else line[:97] + "..." for line in text.splitlines())
    return mo.vstack([mo.md(f"The input of the prediction is a short file: the sequence of the protein, the heme with its covalent bond to the cysteine at position **{cys}** (found as the conserved signature FxxGxxxCxG in the sequence), the ligand as a SMILES, and, if asked, a constraint that puts the donor nitrogen on the iron."), mo.md(f"```yaml\n{short}\n```")])


def c_11_4(mo):
    return {"smiles": mo.ui.text(value="CC(C)Cc1ccc(cc1)C(C)C(=O)O", label="Molecule (SMILES)", full_width=True), "shift": mo.ui.slider(start=0.0, stop=10.0, step=0.5, value=3.0, label="rigid shift of the second pose (angstrom)", show_value=True), "angle": mo.ui.slider(start=0, stop=180, step=10, value=60, label="rotation of the second pose (degrees)", show_value=True)}


def v_11_4(mo, v):
    r = structure.rmsd_demo(v["smiles"], float(v["shift"]), float(v["angle"]))
    if r is None:
        return mo.md("No pair of conformers could be made for this molecule.")
    return mo.vstack([mo.md("The RMSD of a docking pose is measured in the frame of the protein, so a ligand that is in the wrong place is wrong even if its shape is right; the RMSD after an alignment only compares shapes. Two conformers of one molecule, the second moved as a rigid body:"),
                      mo.ui.table(pd.DataFrame([{"measure": k, "RMSD (A)": round(x, 3)} for k, x in r.items()]), selection=None, pagination=False)])


def c_11_5(mo):
    return {"threshold": mo.ui.slider(start=0.5, stop=6.0, step=0.1, value=2.0, label="a pose is right when its RMSD to the crystal pose is within (angstrom)", show_value=True)}


def v_11_5(mo, v):
    poses = store.table("poses")
    table = structure.success(poses, float(v["threshold"]))
    grid = pd.concat([structure.success(poses, t).assign(threshold=t) for t in np.linspace(0.5, 6.0, 23)]).melt(["engine", "threshold"], ["first sample", "best of the samples"], "rule", "share of ligands")
    return mo.vstack([mo.md("A replay of the stored predictions for the ligands of the crystal-anchored benchmark: every sample of each engine with its RMSD to the crystal pose (nothing is predicted here). The first sample is what one gets without knowing the answer; the best of the samples is what a perfect choice among them would give, and the distance between the two is the room for a rule that chooses."),
                      mo.ui.table(table.round(3), selection=None, pagination=False), alt.Chart(grid).mark_line().encode(x="threshold:Q", y=alt.Y("share of ligands:Q", scale=alt.Scale(domain=[0, 1])), color="engine:N", strokeDash="rule:N").properties(width=520, height=260)])


def c_11_9(mo):
    return {"tokens": mo.ui.slider(start=200, stop=900, step=10, value=590, label="tokens (residues + atoms of the heme and the ligand)", show_value=True), "channels": mo.ui.slider(start=32, stop=256, step=32, value=128, label="channels of the pair representation", show_value=True),
            "bytes": mo.ui.radio(options=[2, 4], value=4, label="bytes per number", inline=True), "copies": mo.ui.slider(start=2, stop=16, step=1, value=8, label="pair tensors alive at once", show_value=True), "heads": mo.ui.slider(start=1, stop=8, step=1, value=4, label="attention heads", show_value=True)}


def v_11_9(mo, v):
    r = structure.pair_tensor_memory(int(v["tokens"]), int(v["channels"]), int(v["bytes"]), int(v["copies"]), int(v["heads"]))
    grid = pd.DataFrame([{"tokens": n, "GB": structure.pair_tensor_memory(n, int(v["channels"]), int(v["bytes"]), int(v["copies"]), int(v["heads"]))["total (GB)"]} for n in range(200, 910, 10)])
    limit = alt.Chart(pd.DataFrame({"y": [16]})).mark_rule(color="#c62828", strokeDash=[4, 3]).encode(y="y:Q")
    return mo.vstack([mo.md(f"The pair representation of a co-folding network has one vector for every pair of tokens, so its memory grows with the square of the size, and a triangle attention over it with the cube. An order-of-magnitude estimate for {int(v['tokens'])} tokens: pair tensors **{r['pair tensors (GB)']:.1f} GB**, "
                            f"the logits of one triangle attention **{r['logits of one triangle attention (GB)']:.1f} GB**, together about **{r['total (GB)']:.1f} GB**; the dashed line is the 16 GB of the machine of the project. The estimate leaves out the weights, the diffusion module and every temporary, so it is a floor, not a forecast."),
                      (alt.Chart(grid).mark_line().encode(x="tokens:Q", y="GB:Q") + limit).properties(width=520, height=240), replay.page(mo, {"enzyme": "both", "alpha": 0.05}, ["ost_vs_ours"], "The comparison of the project with the yardstick of OpenStructure:")])


def c_11_10(mo):
    names = sorted(store.table("posebusters")["compound"])
    return {"compound": mo.ui.dropdown(options=names, value=names[0], label="A submitted structure"), "protein": mo.ui.checkbox(value=True, label="protein")}


def v_11_10(mo, v):
    table = store.table("posebusters")
    row = table[table["compound"] == v["compound"]].iloc[0]
    return mo.vstack([mo.md(f"The file that is submitted for each of the twenty ligands is the first pose that passes the validity checks of PoseBusters with the margin rule, relaxed under a rule written beforehand. {v['compound']}: **{'all' if row['all passed'] else 'not all'}** of the {int(row['checks made'])} checks passed ({int(row['checks failed'])} failed). The table has the twenty."),
                      mo.ui.table(table, selection=None, pagination=False)])


def e_11_10(mo, v):
    return mo.iframe(structure.viewer_html(structure.pdb_text(f"submitted_{v['compound']}"), bool(v["protein"])), height="460px")


def c_11_12(mo):
    return {"smiles": mo.ui.text(value="CC(C)C[C@H](N)C(=O)NC(C)C(O)C", label="A ligand (SMILES; leave a centre without its @ to see the ambiguity)", full_width=True)}


def v_11_12(mo, v):
    mol = molecule.parse(v["smiles"])
    if mol is None:
        return mo.md("That is not a SMILES that RDKit can read.")
    r = structure.stereoisomers(v["smiles"])
    return mo.vstack([mo.md(f"The molecule has **{len(r['centres'])}** stereocentres, **{len(r['unassigned'])}** of them without a configuration. A SMILES without it describes every stereoisomer at once (a racemate), and a co-folding model given it draws one of them without saying which; the three ligands of the official set that came this way were given their stereochemistry from the supplied information. "
                            f"The {len(r['isomers'])} stereoisomer(s):"), mo.ui.table(pd.DataFrame({"SMILES": r["isomers"]}), selection=None, pagination=False)])


def _r(families, intro):
    return (replay.controls, lambda mo, v: replay.page(mo, v, families, intro))


SIMS = {"11.1": (c_11_1, v_11_1, e_11_1), "11.2": (c_11_2, v_11_2), "11.3": (c_11_3, v_11_3), "11.4": (c_11_4, v_11_4), "11.5": (c_11_5, v_11_5),
        "11.6": _r(["restraint_expansion", "structure_gates"], "Section 11.6 put the donor nitrogen on the iron with a restraint in the input, and asked what that adds."),
        "11.7": _r(["structure_gates", "heme_frame_prior", "ei_sampling"], "Section 11.7 collects the levers that did not pass: seeds, templates, a prior on the heme frame and sampling by expected improvement."),
        "11.9": (c_11_9, v_11_9), "11.10": (c_11_10, v_11_10, e_11_10),
        "11.11": _r(["pose_pharmacophore_model"], "Section 11.11 built a pharmacophore model of the poses and looked at the two biases that such a model carries."),
        "11.12": (c_11_12, v_11_12), "11.13": _r(["ligand_dynamics"], "Section 11.13 let the ligand move in the frozen pocket and asked whether the movement ranks the poses.")}
