# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX (fourth part): the compound relative to the pocket, the one pharmacophore representation that is not about the ligand alone (9.13)."""

from notebooks.notebook_builder import code, figure, md

POCKET_FACTS = r'''
# The pocket block of 9.13: the sizes of the two reference sets, the gate lines of the four runs (the addition on its own enzyme, the pocket of the other enzyme as the control) and the number of
# compounds with each mode of the block. Every number is read from the files that the queue jobs 262 to 264 left.
import json
import re

import pandas as pd

pocket_logs = ROOT / "data/structure_track/outputs/cyp3a4/logs"
pocket_sets = {}
for pocket_name in ("cyp3a4", "cyp2d6"):
    pocket_ligands = json.loads((ROOT / f"data/structure_track/inputs/{pocket_name}_reference_ligands.json").read_text())["ligands"]
    pocket_sets[pocket_name] = (len(pocket_ligands), len({r["comp_id"] for r in pocket_ligands}), len({r["pdb_id"] for r in pocket_ligands}))
pocket_stacked = re.compile(r"z0=([\d.]+)\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.]+), better (\d+)/(\d+)\)")
pocket_alone = re.compile(r"z0=([\d.]+)\s+alone\s+(\S+)\s+vs (\S+)\s*: AUC ([\d.]+) -> ([\d.]+) \(([+-][\d.]+), p=([\d.]+)")
pocket_gate = re.compile(r"^(CYP\w+) GATE \((alone|stacked)[^)]*\): (MET|NOT MET)", re.MULTILINE)
pocket_rows = []
for enzyme, own in (("CYP3A4", "pocket_cyp3a4"), ("CYP2D6", "pocket_cyp2d6")):
    for role, log_name in (("own pocket (the addition)", f"aux_{own}_{enzyme}.log"), ("other pocket (the control)", f"exploratory_pocket_control_{enzyme}.log")):
        text = (pocket_logs / log_name).read_text()
        cuts = {float(z0): (float(gain), float(p), int(better), int(folds)) for z0, gain, p, better, folds in pocket_stacked.findall(text)}
        verdicts = {kind: verdict for iso, kind, verdict in pocket_gate.findall(text) if iso == enzyme}
        pocket_rows.append({"enzyme": enzyme, "pocket": role, "stacked gain, z0 1.0": cuts[1.0][0], "p, z0 1.0": cuts[1.0][1], "stacked gain, z0 1.96": cuts[1.96][0], "p, z0 1.96": cuts[1.96][1],
                            "better folds, z0 1.0": f"{cuts[1.0][2]}/{cuts[1.0][3]}", "gate, alone": verdicts.get("alone", "n/a"), "gate, stacked": verdicts.get("stacked", "n/a")})
pocket_table = pd.DataFrame(pocket_rows)
pocket_own = pocket_table[pocket_table["pocket"].str.startswith("own")].set_index("enzyme")
pocket_met = [enzyme for enzyme in pocket_own.index if pocket_own.loc[enzyme, "gate, stacked"] == "MET" and pocket_own.loc[enzyme, "gate, alone"] == "MET"]
pocket_stack_met = [enzyme for enzyme in pocket_own.index if pocket_own.loc[enzyme, "gate, stacked"] == "MET"]
pocket_cache = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/pocket_pharmacophore_cache.csv", index_col="SMILES")
pocket_n = len(pocket_cache)
pocket_share_donor = float(pocket_cache["pp_cyp3a4_donor_best"].notna().mean())
pocket_share_substrate = float(pocket_cache["pp_cyp3a4_substrate_best"].notna().mean())
pocket_table.round(4)
'''

CELLS = [
    md("""
## 9.13 The compound relative to the pocket

> **Field:** Cheminformatics · Structural biology · Computational chemistry

*Where this section sits:* 3.5 gave a compound a three-point pharmacophore fingerprint (7.8 tested it), 10.6 gave it a pharmacophore level of its own in the physics block, and 11.8 used a pharmacophore term to choose among poses. All three describe the ligand alone, and all three failed their gates. A representation of the ligand alone cannot differ between CYP3A4 and CYP2D6, although their pockets do. This section describes the one pharmacophore representation that is *relative to a pocket*, declared before any code ran (the checklist item of the same name), and what became of it.

**The pocket as a set of fields.** A **pharmacophore** is the arrangement in space of the features of a molecule that matter for binding (a hydrogen-bond donor, an acceptor, an aromatic ring, a hydrophobic patch, a charged group). The public crystal structures of an enzyme with ligands in its active site say where such features sit in that pocket. To compare structures of different crystals they have to be put in one **heme frame**: a coordinate system fixed to the heme, with the iron at the origin, the axis $z$ along the normal of the porphyrin plane (the best-fit plane through the four pyrrole nitrogens, found as the direction in which those four points vary least) and pointing to the side where the ligand lies, and the axis $x$ pointing to one named pyrrole nitrogen. In that frame every ligand feature has coordinates, and the features of all ligands of one enzyme give a **density field**: a number at every point of a grid that is large where many ligands have a feature of that family. With $x_i$ the position of the $i$-th feature of a family, $w_i$ its weight and $\\sigma$ a width,

$$F(x) = \\frac{1}{F_{\\max}} \\sum_i w_i \\, e^{-\\lVert x - x_i \\rVert^2 / (2\\sigma^2)}, \\qquad w_i = \\frac{1}{n_{\\mathrm{copies}}},$$

where $F_{\\max}$ is the largest value of the sum on the grid, so that the field has a maximum of 1, and $n_{\\mathrm{copies}}$ is the number of crystal copies of the ligand to which feature $i$ belongs, so that a ligand with fifty entries counts as much as a ligand with one. The width is 1.5 Å for the six feature families (donor, acceptor, aromatic ring, hydrophobe, positive and negative ionisable) and 1.2 Å for a seventh field, the occupancy of the heavy atoms. CYP3A4 rests on {{=pocket_sets['cyp3a4'][0]}} ligand copies of {{=pocket_sets['cyp3a4'][1]}} distinct ligands in {{=pocket_sets['cyp3a4'][2]}} entries of the PDB, and CYP2D6 on {{=pocket_sets['cyp2d6'][0]}} copies of {{=pocket_sets['cyp2d6'][1]}} ligands in {{=pocket_sets['cyp2d6'][2]}} entries, so that the CYP2D6 fields are much the less certain. No label of the challenge is used anywhere in them.
"""),
    figure("pharm_pocket_fields"),
    md("""
**From a compound to a handful of numbers.** A compound has no position in the frame; it is given many. Up to twelve conformers within 5 kcal/mol of the lowest (the conformers of 3.4) are each placed by an **anchor atom** and a fixed set of orientations: sixty directions in a cone of 70 degrees around $z$ and eighteen spins about each, 1,080 in all. In the *donor mode*, which applies to a compound with a nitrogen that can bind the iron (the Type II donor of 1.9), that nitrogen sits 2.2 Å above the iron; in the *substrate mode* each non-aromatic carbon that bears a hydrogen is the anchor in turn, 4.5 Å above the iron, the middle of the distance at which a carbon is oxidised. A pose is refused when another heavy atom lies lower than 1.5 Å above the porphyrin plane or closer than 2.0 Å to the iron. A pose scores $S = \\bar O + \\frac{1}{|P|}\\sum_{f \\in P} \\bar F_f$, where $\\bar O$ is the mean occupancy at its heavy atoms, $P$ is the set of families that the compound has and $\\bar F_f$ is the mean value of the field of family $f$ at the compound's features of that family, each read at the nearest grid point; no weight is fitted. For each mode that applies the block keeps the best score, its occupancy part, its six family parts, the share of poses that were accepted, the height of the centroid and the distance of the nearest heavy atom at the best pose: eleven columns, twenty-two for the two modes, and the same twenty-two for each pocket. Of the {{=pocket_n}} compounds of the two files, {{=pocket_share_donor:.0%}} have the donor mode and {{=pocket_share_substrate:.0%}} the substrate mode for the CYP3A4 pocket.

**The test, and the control.** The block is added to the two deep boosted-tree candidates on feature sets A and A+fps (7.2), and the gate is the one of every addition: alone and inside the stack, corrected resampled t below 0.05 at both confidence cuts with a positive mean AUC difference, 3 repeats of 5 outer folds, and a replicate on 10 outer folds that must be met as well before the group can join the stack. Because the pocket is the whole point, each enzyme is also given the pocket of the *other* enzyme as a control that enters no gate: a gain that the control shares is not about the pocket.
"""),
    code(POCKET_FACTS),
    md("""
The table gives, for each enzyme, the change in AUC when the block of its own pocket is added to the default stack and when the block of the other pocket is. The gate of the own pocket is **{{=', '.join(pocket_met) if pocket_met else 'not met for either enzyme'}}** (alone and stacked at both cuts{{=''.join([' for ' + e for e in pocket_met]) if pocket_met else ''}}); the stacked comparison alone is met for {{=', '.join(pocket_stack_met) if pocket_stack_met else 'neither enzyme'}}. {{=('A group that is met on 15 folds still has to be met on 10, and the final build reads that verdict, not this one.' if pocket_met or pocket_stack_met else 'A group that fails its gate is written down as a negative result and is not tuned until it passes: the representation was not changed after the first run, and nothing of it enters the final stack.')}}
"""),
    figure("pharm_forest"),
    md("""
**What the figure shows.** Every pharmacophore representation of the project, whichever of the three levels it describes, and what happened to it. The ligand-only representations (the three-point fingerprint, the quantum-pharmacophore block, the physics block with its fourth level) and the pose terms are on the same page as the pocket block, so that the claim "pharmacophore information does not help" can be read for what it covers: a family of attempts, each on its own measure, and not one failed idea.

*On what actually ran.* `cypmode/challenge/pocket_pharmacophore.py` builds the fields (`build_fields`), places the conformers (`score_poses`, `compound_columns`) and caches the block for every compound of both files (job `262_pocket_pharmacophore_extract`); the reference set of CYP2D6 is collected by `cypmode/validation/reference_ligands.py` with the builder that made the one of CYP3A4. The gated runs are the jobs `263_pocket_pharmacophore_CYP3A4` and `264_pocket_pharmacophore_CYP2D6`, which also run the control, and `265_pocket_pharmacophore_replicate` runs the 10-fold comparison only for an enzyme whose own gate was met. The placement is rigid and sees no protein beyond two refusal rules, the field is read at the nearest grid point and there are at most twelve conformers, so the block is a cheap proxy for where a compound could sit in the pocket and not a docking score; its tests are in `tests/test_pocket_pharmacophore.py`. The CYP2D6 reference set has few distinct ligands, and any result for that pocket carries that limit.
"""),
]
