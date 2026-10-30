# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX (fourth part): the compound relative to the pocket, the one pharmacophore representation that is not about the ligand alone (9.13), and the two pharmacophore models, one from the pocket and one from the ligands (9.14)."""

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
bias_line = re.search(r"a member within ([\d.]+) A of the crystal pose for (\d+)% of the (\d+) copies", (pocket_logs / "conformer_bias.log").read_text())
ensemble_share = int(bias_line.group(2))  # the share of crystal ligand copies whose bound conformation the ensemble of the block contains (job 273, section 11.11)
ensemble_ok = ensemble_share >= 80
pocket_cache = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/pocket_pharmacophore_cache.csv", index_col="SMILES")
pocket_n = len(pocket_cache)
pocket_share_donor = float(pocket_cache["pp_cyp3a4_donor_best"].notna().mean())
pocket_share_substrate = float(pocket_cache["pp_cyp3a4_substrate_best"].notna().mean())
'''

POCKET_TABLE = r'''
pocket_table.round(4)
'''

MODEL_FACTS = r'''
# The two pharmacophore models of 9.14: the sizes of what was built (the sites of the pockets, the hypotheses of the vocabulary) and the gate lines of the runs (the model on its own enzyme, its control),
# every number read from the files that the queue jobs 262, 266 and 267 to 271 left.
import re

import numpy as np
import pandas as pd

from cypmode.challenge import pocket_pharmacophore as pocket_module

site_counts = {name: sum(len(v) for v in pocket_module.sites_of(pocket_module.load_fields(name)).values()) for name in ("cyp3a4", "cyp2d6")}
extract_line = re.search(r"(\d+) compounds, (\d+) realised hypotheses in all, (\d+) in the vocabulary \(at least (\d+) compounds\), (\d+) compounds realise none", (ROOT / "data/structure_track/outputs/cyp3a4/logs/pharmacophore_hypotheses_extract.log").read_text())
hyp_compounds, hyp_total, hyp_vocabulary, hyp_min_compounds, hyp_none = (int(v) for v in extract_line.groups())
model_logs = ROOT / "data/structure_track/outputs/cyp3a4/logs"
model_stacked = re.compile(r"z0=([\d.]+)\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.]+), better (\d+)/(\d+)\)")
model_gate = re.compile(r"^(CYP\w+) GATE \((alone|stacked)[^)]*\): (MET|NOT MET)", re.MULTILINE)
model_runs = (
    ("site model", "CYP3A4", "own pocket (the addition)", "aux_site_cyp3a4_CYP3A4.log"),
    ("site model", "CYP3A4", "other pocket (the control)", "exploratory_site_control_CYP3A4.log"),
    ("site model", "CYP2D6", "own pocket (the addition)", "aux_site_cyp2d6_CYP2D6.log"),
    ("site model", "CYP2D6", "other pocket (the control)", "exploratory_site_control_CYP2D6.log"),
    ("hypothesis model", "CYP3A4", "real labels (the addition)", "aux_hypothesis_CYP3A4.log"),
    ("hypothesis model", "CYP3A4", "permuted labels (the control)", "exploratory_hypothesis_permuted_CYP3A4.log"),
    ("hypothesis model", "CYP2D6", "real labels (the addition)", "aux_hypothesis_CYP2D6.log"),
    ("hypothesis model", "CYP2D6", "permuted labels (the control)", "exploratory_hypothesis_permuted_CYP2D6.log"),
)
model_rows = []
for model, enzyme, role, log_name in model_runs:
    text = (model_logs / log_name).read_text()
    cuts = {float(z0): (float(gain), float(p), int(better), int(folds)) for z0, gain, p, better, folds in model_stacked.findall(text)}
    verdicts = {kind: verdict for iso, kind, verdict in model_gate.findall(text) if iso == enzyme}
    model_rows.append({"model": model, "enzyme": enzyme, "run": role, "stacked gain, z0 1.0": cuts[1.0][0], "p, z0 1.0": cuts[1.0][1], "stacked gain, z0 1.96": cuts[1.96][0], "p, z0 1.96": cuts[1.96][1],
                       "better folds, z0 1.0": f"{cuts[1.0][2]}/{cuts[1.0][3]}", "gate, alone": verdicts.get("alone", "n/a"), "gate, stacked": verdicts.get("stacked", "n/a")})
model_table = pd.DataFrame(model_rows)
is_addition = model_table["run"].str.contains("addition")
model_met = {m: [e for e in ("CYP3A4", "CYP2D6") if len(model_table[is_addition & (model_table["model"] == m) & (model_table["enzyme"] == e) & (model_table["gate, alone"] == "MET") & (model_table["gate, stacked"] == "MET")])] for m in ("site model", "hypothesis model")}
model_stack_met = {m: [e for e in ("CYP3A4", "CYP2D6") if len(model_table[is_addition & (model_table["model"] == m) & (model_table["enzyme"] == e) & (model_table["gate, stacked"] == "MET")])] for m in ("site model", "hypothesis model")}
control_rows = model_table[~is_addition]
control_shares = [f"{r.model} on {r.enzyme}" for r in control_rows.itertuples() if r[4] > 0 and r[6] > 0 and r[5] < 0.05 and r[7] < 0.05]
'''

MODEL_TABLE = r'''
model_table.round(4)
'''

CELLS = [
    code(POCKET_FACTS, hide_input=True),
    md("""
## 9.13 The compound relative to the pocket

> **Field:** Cheminformatics · Structural biology · Computational chemistry

*Where this section sits:* 3.5 gave a compound a three-point pharmacophore fingerprint (7.8 tested it), 10.6 gave it a pharmacophore level of its own in the physics block, and 11.8 used a pharmacophore term to choose among poses. All three describe the ligand alone, and all three failed their gates. A representation of the ligand alone cannot differ between CYP3A4 and CYP2D6, although their pockets do. This section describes the one pharmacophore representation that is *relative to a pocket*, declared before any code ran, as the header of its module records, and what became of it.

**The pocket as a set of fields.** A **pharmacophore** is the arrangement in space of the features of a molecule that matter for binding (a hydrogen-bond donor, an acceptor, an aromatic ring, a hydrophobic patch, a charged group). The public crystal structures of an enzyme with ligands in its active site say where such features sit in that pocket. To compare structures of different crystals they have to be put in one **heme frame**: a coordinate system fixed to the heme, with the iron at the origin, the axis $z$ along the normal of the porphyrin plane (the best-fit plane through the four pyrrole nitrogens, found as the direction in which those four points vary least) and pointing to the side where the ligand lies, and the axis $x$ pointing to one named pyrrole nitrogen. In that frame every ligand feature has coordinates, and the features of all ligands of one enzyme give a **density field**: a number at every point of a grid that is large where many ligands have a feature of that family. With $x_i$ the position of the $i$-th feature of a family, $w_i$ its weight and $\\sigma$ a width,

$$F(x) = \\frac{1}{F_{\\max}} \\sum_i w_i \\, e^{-\\lVert x - x_i \\rVert^2 / (2\\sigma^2)}, \\qquad w_i = \\frac{1}{n_{\\mathrm{copies}}},$$

where $F_{\\max}$ is the largest value of the sum on the grid, so that the field has a maximum of 1, and $n_{\\mathrm{copies}}$ is the number of crystal copies of the ligand to which feature $i$ belongs, so that a ligand with fifty entries counts as much as a ligand with one. The width is 1.5 Å for the six feature families (donor, acceptor, aromatic ring, hydrophobe, positive and negative ionisable) and 1.2 Å for a seventh field, the occupancy of the heavy atoms. CYP3A4 rests on {{=pocket_sets['cyp3a4'][0]}} ligand copies of {{=pocket_sets['cyp3a4'][1]}} distinct ligands in {{=pocket_sets['cyp3a4'][2]}} entries of the PDB, and CYP2D6 on {{=pocket_sets['cyp2d6'][0]}} copies of {{=pocket_sets['cyp2d6'][1]}} ligands in {{=pocket_sets['cyp2d6'][2]}} entries, so that the CYP2D6 fields are much the less certain. No label of the challenge is used anywhere in them.
"""),
    figure("pharm_pocket_fields"),
    md("""
**From a compound to a handful of numbers.** A compound has no position in the frame; it is given many. Up to twelve conformers within 5 kcal/mol of the lowest (the conformers of 3.4) are each placed by an **anchor atom** and a fixed set of orientations: sixty directions in a cone of 70 degrees around $z$ and eighteen spins about each, 1,080 in all. In the *donor mode*, which applies to a compound with a nitrogen that can bind the iron (the Type II donor of 1.9), that nitrogen sits 2.2 Å above the iron; in the *substrate mode* each non-aromatic carbon that bears a hydrogen is the anchor in turn, 4.5 Å above the iron, the middle of the distance at which a carbon is oxidised. A pose is refused when another heavy atom lies lower than 1.5 Å above the porphyrin plane or closer than 2.0 Å to the iron. A pose scores $S = \\bar O + \\frac{1}{|P|}\\sum_{f \\in P} \\bar F_f$, where $\\bar O$ is the mean occupancy at its heavy atoms, $P$ is the set of families that the compound has and $\\bar F_f$ is the mean value of the field of family $f$ at the compound's features of that family, each read at the nearest grid point; no weight is fitted. For each mode that applies the block keeps the best score, its occupancy part, its six family parts, the share of poses that were accepted, the height of the centroid and the distance of the nearest heavy atom at the best pose: eleven columns, twenty-two for the two modes, and the same twenty-two for each pocket. Of the {{=pocket_n}} compounds of the two files, {{=pocket_share_donor:.0%}} have the donor mode and {{=pocket_share_substrate:.0%}} the substrate mode for the CYP3A4 pocket.

**The test, and the control.** The block is added to the two deep boosted-tree candidates on feature sets A and A+fps (7.2), and the gate is the one of every addition: alone and inside the stack, corrected resampled t below 0.05 at both confidence cuts with a positive mean AUC difference, 3 repeats of 5 outer folds, and a replicate on 10 outer folds that must be met as well before the group can join the stack. Because the pocket is the whole point, each enzyme is also given the pocket of the *other* enzyme as a control that enters no gate: a gain that the control shares is not about the pocket.
"""),
    code(POCKET_TABLE),
    md("""
The table gives, for each enzyme, the change in AUC when the block of its own pocket is added to the default stack and when the block of the other pocket is. The gate of the own pocket is **{{=', '.join(pocket_met) if pocket_met else 'not met for either enzyme'}}** (alone and stacked at both cuts{{=''.join([' for ' + e for e in pocket_met]) if pocket_met else ''}}); the stacked comparison alone is met for {{=', '.join(pocket_stack_met) if pocket_stack_met else 'neither enzyme'}}. {{=('A group that is met on 15 folds still has to be met on 10, and the final build reads that verdict, not this one.' if pocket_met or pocket_stack_met else 'A group that fails its gate is written down as a negative result and is not tuned until it passes: the representation was not changed after the first run, and nothing of it enters the final stack.')}}
"""),
    md("""
**A first run that was void.** The block was first written with the conformer generator seeded with 0. In the version of RDKit used here, `EmbedMultipleConfs` with seed 0 returns the same conformer for every copy (the raw embeddings are identical and the energy span of the ensemble is zero), so the twelve conformers of that run were one conformer, and the hypothesis cache of 9.14 had the same defect. The first result of the block, NOT MET on both enzymes, is void: its logs are kept as `exploratory_pocket_first_run_seed0_own_CYP3A4.log` and the three others of that name (own and control pocket, for each enzyme), and no gate parser reads them. Both were rebuilt on 2026-10-23 with a conformer seed of 42, and `test_the_ensemble_of_a_flexible_compound_holds_distinct_conformers` requires distinct geometries. The older code that places conformers (the physics block, the pose terms, the basic-centre block and the fingerprint) uses seeds 42 or 7 and is not affected.
"""),
    md("""
*On what actually ran.* `cypmode/challenge/pocket_pharmacophore.py` builds the fields (`build_fields`), places the conformers (`score_poses`, `compound_columns`) and caches the block for every compound of both files (job `262_pocket_pharmacophore_extract`); the reference set of CYP2D6 is collected by `cypmode/validation/reference_ligands.py` with the builder that made the one of CYP3A4. The gated runs are the jobs `263_pocket_pharmacophore_CYP3A4` and `264_pocket_pharmacophore_CYP2D6`, which also run the control, and `265_pocket_pharmacophore_replicate` runs the 10-fold comparison only for an enzyme whose own gate was met. The placement is rigid and sees no protein beyond two refusal rules, the field is read at the nearest grid point and there are at most twelve conformers, so the block is a cheap proxy for where a compound could sit in the pocket and not a docking score (**docking** is the search for the pose of a molecule in a pocket that a scoring function ranks best); its tests are in `tests/test_pocket_pharmacophore.py`. The same extraction (job 262) also writes the six columns of the site model of 9.14, which are computed from the same poses. The CYP2D6 reference set has few distinct ligands, and any result for that pocket carries that limit. The ensemble of up to twelve conformers is drawn with the seed 42 (a seed of 0 returns the same conformer every time in the installed RDKit, which turned a first run into a run with one conformer per compound; that run is kept in the logs and is not a verdict), and {{=('it contains a conformer within 1.0 Å of the bound pose for only ' + str(ensemble_share) + '% of the crystal ligand copies (11.11), so every verdict of this section and of 9.14 is a verdict for compounds placed in their low-energy conformers, which the bound state often is not.' if not ensemble_ok else 'it contains a conformer within 1.0 Å of the bound pose for ' + str(ensemble_share) + '% of the crystal ligand copies (11.11).')}}
"""),
    code(MODEL_FACTS, hide_input=True),
    md("""
## 9.14 Pharmacophore models: sites from the pocket, hypotheses from the ligands

> **Field:** Cheminformatics · Structural biology · Statistics · Machine learning

*Where this section sits:* 3.5, 7.8, 10.6, 11.8 and 9.13 turned a compound into numbers that describe its pharmacophore, and every gate they met was failed or not yet decided. None of them built a *model*. This section builds two, one from each kind of evidence that pharmacophore modelling has always used, and gates them like every other addition. It rests on 9.13 (the heme frame and the fields), on 7.2 (the boosted trees to which columns are added) and on 6.3 (nested validation, which decides here how a model may be fitted).

**What a pharmacophore model is.** A **pharmacophore hypothesis** is a statement about the arrangement of chemical features, such as "an acceptor, an aromatic ring and a hydrophobic atom group with these distances between them", that a molecule must offer to bind. A *model* is a set of such hypotheses that was derived from data and is then used to judge a molecule: to say how well it matches. There are two sources of data. A **structure-based** model is read from the crystal structures of the target: where in the pocket the ligands that were seen place their donors, acceptors and rings. A **ligand-based** model is learned from molecules whose activity is known, by asking which arrangements the actives share and the inactives lack. Both are built here.

**The structure-based site model.** The six feature fields of 9.13 already say where the crystal ligands of an enzyme put each family of features. The model keeps the peaks. In each field the points that are higher than all of their 26 neighbours and at least 0.35 (the maximum of a field is 1) are the candidates, taken in order of height; a candidate is kept as a **site** when it is at least 2.5 Å from every kept site of the same family, and at most four are kept per family. The weight of a site is the height of the field at it, and its tolerance is a sphere of 1.5 Å, the width of the fields. CYP3A4 ends with {{=site_counts['cyp3a4']}} sites and CYP2D6 with {{=site_counts['cyp2d6']}}. A placed compound *matches* a site when one of its features of that family lies inside the sphere. For each mode of 9.13 that applies, and over the same poses, the model gives three numbers per pocket: the **recall**, the weight of the matched sites over the weight of all sites at the best pose; the number of sites matched at that pose; and the **precision**, the share of the compound's typed features that lie in a sphere of a site of their own family at that pose. The best pose is the one with the largest recall and, among equals, the larger precision. These six numbers are the block `st3a4` or `st2d6` of the feature store.
"""),
    figure("pharm_site_model"),
    md("""
**The ligand-based hypothesis model.** The hypotheses are those of the pmapper program: a subset of two or three pharmacophore features with the binned distances between them (bins of 1 Å), identified by the md5 signature that pmapper computes for the subset. Unlike the fingerprint of 3.5, nothing is folded into a shared bit, so two hypotheses never share a column. A compound *realises* a hypothesis when one of its conformers, the same up to twelve of 9.13, offers the subset with those distances. Over the {{=hyp_compounds}} compounds of the two files the conformers realise {{=hyp_total}} hypotheses in all (counting each compound once per hypothesis); the vocabulary keeps the {{=hyp_vocabulary}} that at least {{=hyp_min_compounds}} compounds realise, and {{=hyp_none}} compounds realise none. No label is read for this step.

*Fitting inside a training split.* Let $n_1$ and $n_0$ be the numbers of positive and negative rows of the training split, and for a hypothesis $h$ let $a_h$ and $b_h$ be the numbers of positive and of negative rows that realise it. A hypothesis is considered only when $a_h + b_h \\ge 10$. Its **log-likelihood ratio** compares how common it is among the positives with how common it is among the negatives, with one pseudo-observation added to each count so that a hypothesis seen in no negative row does not get an infinite ratio:

$$\\lambda_h = \\ln\\frac{a_h + 1}{n_1 + 2} - \\ln\\frac{b_h + 1}{n_0 + 2}.$$

A positive $\\lambda_h$ says the hypothesis favours the positive class. Hypotheses are ranked by the **chi-square statistic** of the two-by-two table of presence against class, which measures how far the table is from one in which presence and class are unrelated:

$$\\chi^2_h = \\frac{n\\,\\bigl(a_h (n_0 - b_h) - (n_1 - a_h)\\, b_h\\bigr)^2}{(a_h + b_h)\\,(n - a_h - b_h)\\, n_1 n_0}, \\qquad n = n_1 + n_0.$$

The model keeps the 60 hypotheses of largest $\\chi^2_h$ among those with $\\lambda_h > 0$ (the pharmacophore of the positives) and the 60 among those with $\\lambda_h < 0$ (the exclusion hypotheses: arrangements that active molecules tend to lack). A compound $c$ is then described by five numbers: the sum $s_c = \\sum_{h \\in K} \\lambda_h r_{ch}$ over the kept set $K$, where $r_{ch}$ is 1 when $c$ realises $h$ and 0 otherwise; the number of positive and of exclusion hypotheses it realises; the largest $\\lambda_h$ among those it realises (0 when none); and the share of the five strongest positive hypotheses that it realises. The sum is the log-odds of a Bayes classifier that treats the hypotheses as independent given the class. They are not independent (overlapping subsets share features), so the sum is a ranking score, not a probability, and the boosted trees that receive it take it for what it is.
"""),
    figure("pharm_hypotheses"),
    md("""
**Keeping the columns free of labels.** A model that chooses its hypotheses with the labels of the rows it then scores will score those rows too well, and a boosted tree trained on such a column learns to trust it. The training rows of a task therefore receive the five columns **out of fold**: from a model, including the choice of the 120 hypotheses, fitted on the other four of five stratified folds of the training rows. The evaluation rows receive them from the model fitted on all training rows, so no evaluation label is read (the tests flip every evaluation label and the predictions must not move, as for every runner). A test with labels that carry no signal shows the need: the out-of-fold column has an AUC below 0.58, the in-sample column above 0.75.

**The test, the controls and the rule.** Each model is added to the two deep boosted-tree candidates on feature sets A and A+fps, and the gate is the one of every addition: alone and inside the stack, corrected resampled t below 0.05 at both confidence cuts with a positive mean AUC difference, 3 repeats of 5 outer folds, then a replicate on 10 outer folds that must be met as well before a group can join the stack. The site model of an enzyme is tested on that enzyme only, with the site model of the *other* enzyme as the control: a gain that the other pocket shares is not about the pocket. The hypothesis model is tested on both enzymes, with the same model fitted on **permuted labels** (the labels shuffled within the training rows before any step) as the control: a gain that survives the permutation is not about the hypotheses. Neither control enters a gate.
"""),
    code(MODEL_TABLE),
    md("""
The table gives, for each model and enzyme, the change in AUC when its columns are added to the default stack, at the two confidence cuts, and for the control. The gate (alone and stacked, both cuts) is met for **{{=', '.join(f'{m} on {e}' for m, es in model_met.items() for e in es) if any(model_met.values()) else 'no model on either enzyme'}}**; the stacked comparison alone is met for {{=', '.join(f'{m} on {e}' for m, es in model_stack_met.items() for e in es) if any(model_stack_met.values()) else 'none'}}. {{=('A control that shares the gain (' + ', '.join(control_shares) + ') says the gain is not about the model.' if control_shares else 'No control shares a gain that is significant at both cuts.')}} {{=('A group that is met on 15 folds still has to be met on 10, and the final build reads that verdict, not this one.' if any(model_stack_met.values()) else 'A group that fails its gate is written down as a negative result and is not tuned until it passes: neither model was changed after the first run, and nothing of them enters the final stack.')}}
"""),
    figure("pharm_forest"),
    md("""
**What the last figure shows.** Every pharmacophore representation and model of the project, whichever of the three levels it describes, and what happened to it. The ligand-only representations (the three-point fingerprint, the quantum-pharmacophore block, the physics block with its fourth level), the pose terms, the pocket block and the two models with their controls are on one page, so that the claim "pharmacophore information does not help" can be read for what it covers: a family of attempts, each on its own measure, and not one failed idea. Where a model and its control look alike, the figure shows it.

*On what actually ran.* `cypmode/challenge/pharmacophore_hypotheses.py` finds the hypotheses of a compound (`realised`), caches the vocabulary for every compound of both files (job `266_hypothesis_extract`), fits the model (`fit_model`), computes its columns (`model_columns`) and the out-of-fold and permuted variants (`hypothesis_columns`); the candidates are `hypothesis_candidates` and the site candidates of `nested_runners.py`. The gated runs are the jobs `267_site_model_CYP3A4` and `268_site_model_CYP2D6` (each with the control of the other pocket) and `269_hypothesis_model_CYP3A4` and `270_hypothesis_model_CYP2D6` (each with the permuted-label control), and `271_pharmacophore_models_replicate` runs the 10-fold comparison only for a group whose stacked gate was met. The tests are in `tests/test_pharmacophore_hypotheses.py` (a planted hypothesis is found, the out-of-fold columns carry no selection bias, the evaluation columns do not move when evaluation labels are flipped, the permuted control carries no label information) and in `tests/test_pocket_pharmacophore.py` (the sites of a field, the matching of a pose). Limits: the hypotheses use conformers from one force field and a bin of 1 Å, so two arrangements that differ by a fraction of an ångström on a bin edge are different hypotheses; the site model inherits the rigid placement of 9.13 and the small crystal set of CYP2D6; and a hypothesis that distinguishes a series of one chemotype is a property of that series, which the nested validation, whose folds keep similar compounds apart, is built to expose.
"""),
]
