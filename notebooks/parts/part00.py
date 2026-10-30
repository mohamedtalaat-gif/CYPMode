# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part 0: orientation. The problem, how the project grew, the challenge, the whole system, the ledger of everything tried, how the notebook is built."""

from notebooks.notebook_builder import code, figure, md

PANEL = '''
# The motif screen's first measurement: of the molecules that the Veith panel labels as inhibitors of an enzyme, how many carry a nitrogen that could reach the heme iron?
# (The panel loader prints a line per file it reads; the output is discarded here.)
import contextlib, io

from cypmode.data.panel import load_panel, summarize_panel

with contextlib.redirect_stdout(io.StringIO()):
    panel = load_panel()
summary = summarize_panel(panel)
summary["share with a heme-binding nitrogen"] = (100 * summary["frac_with_type_ii_motif"]).round(1).astype(str) + " %"
show_table(
    summary.rename(columns={"isoform": "enzyme", "n_labeled_inhibitors": "labelled inhibitors", "n_with_type_ii_motif": "with a heme-binding nitrogen"})[
        ["enzyme", "labelled inhibitors", "with a heme-binding nitrogen", "share with a heme-binding nitrogen"]
    ]
)
'''

FOUR = '''
# The four compounds whose structures in CYP3A4 were solved by cryo-EM, re-predicted by Boltz-2: the distance from the heme iron to the nearest nitrogen of the motif.
import json

import pandas as pd

summary_path = ROOT / "data/structure_track/outputs/cyp3a4/validation_summary.json"
reference = json.loads(summary_path.read_text())
rows = []
for name, entry in reference.items():
    motifs = sorted({m["motif"] for m in entry["motif_nitrogen_distances"]})
    rows.append({"compound": name, "motif found": ", ".join(motifs), "iron to nearest motif nitrogen (angstrom)": entry["closest_motif_nitrogen_distance_angstrom"],
                 "bonded to the iron": "yes" if entry["coordinated"] else "no"})
show_table(pd.DataFrame(rows), floatfmt=".2f")
'''

FILES = '''
# What the two challenge files hold. A column is named "<enzyme>_<quantity>"; the quantities are grouped below.
import pandas as pd

train = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
test = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
enzymes = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
quantities = {
    "pIC50 after pre-incubation (the TDI condition)": ["_pIC50_TDI_condition"],
    "its lower and upper confidence limits": ["_pIC50_TDI_condition_conf_low", "_pIC50_TDI_condition_conf_high"],
    "its standard deviation": ["_pIC50_TDI_condition_std"],
    "pIC50 without pre-incubation (direct inhibition)": ["_pIC50_direct_inhibition"],
    "its lower and upper confidence limits ": ["_pIC50_direct_inhibition_conf_low", "_pIC50_direct_inhibition_conf_high"],
    "its standard deviation ": ["_pIC50_direct_inhibition_std"],
    "yes/no label": ["_is_TDI"],
}
rows = []
for label, suffixes in quantities.items():
    filled = {e: int(train[e + suffixes[0]].notna().sum()) for e in enzymes if e + suffixes[0] in train.columns}
    rows.append({"quantity": label, "columns": len(suffixes) * len(filled), "enzymes": ", ".join(filled), "filled rows per enzyme": ", ".join(f"{e} {n}" for e, n in filled.items())})
print(f"training file: {len(train)} rows x {train.shape[1]} columns; blinded test file: {len(test)} rows x {test.shape[1]} columns ({', '.join(test.columns)})")
show_table(pd.DataFrame(rows))
'''

LABELLED = '''
# How many training molecules can be used for each enzyme, read with the project's own view of the file (section 2.3 explains the rows that are left out).
labels = pd.DataFrame({
    "enzyme": list(F.labelled),
    "molecules with both measurements and a label": list(F.labelled.values()),
    "of which labelled positive": list(F.positives.values()),
    "share positive": [F.positives[e] / F.labelled[e] for e in F.labelled],
    "molecules without a usable label": [F.unlabelled[e] for e in F.labelled],
})
show_table(labels)
'''

LEDGER = '''
# The ledger: every logged verdict of a pre-declared gate, grouped by experiment. The counts are read from the experiments' own logs.
import re

from cypmode.audit.ledger_kinds import ledger_rows

DESCRIPTION = {
    "selection_aware_v2": ("selection-aware CYP2D6 model: P(label) = P(hit) x P(label | hit)", "9.1"),
    "hedged_blend": ("hedged rank blend of the hits-only and the selection-aware CYP2D6 stacks", "9.1"),
    "pair_transfer": ("structure-activity (SAR) pair model on descriptor differences", "9.2"),
    "mmp_transfer": ("matched-molecular-pair transfer", "9.2"),
    "committee_label_model": ("label-process committee: P(shift > log10 2) from two regressors", "9.3"),
    "embeddings": ("frozen CheMeleon foundation-model embeddings", "9.4"),
    "test_likeness": ("training weights that follow similarity to the test structures", "9.5"),
    "arms_experiment": ("assay-arm regressors floor-clipped at the assay floor", "9.6"),
    "xiso_transfer": ("transfer of the assay arms across enzymes", "9.7"),
    "physics_experiment": ("GFN2-xTB physics features, tier A", "10.6"),
    "physics_compact": ("the 20 most useful physics columns", "10.6"),
    "physics_tierb": ("physics features, tier B (Boltzmann-weighted conformers)", "10.6"),
    "population_physics": ("physics inside the selection-aware model", "10.6"),
    "pair_transfer_physics": ("structure-activity (SAR) pair model on physics differences", "10.6"),
    "replicate_verdict": ("10-fold replicate of the physics additions", "10.6"),
    "structure_gates": ("structure levers: Fe-donor bond and contact restraints, seeds, conformer templates, step scale", "11.6, 11.7"),
    "restraint_expansion": ("confirmation of the bond restraint on 20 more donor ligands", "11.6"),
    "selection_validation": ("pose choice by minimum Bayes risk", "11.8"),
    "pose_awareness": ("physics and pharmacophore terms in pose choice", "11.8"),
    "series_consensus": ("structure-activity (SAR) term: agreement with the poses of analogs", "11.8"),
    "heme_frame_prior": ("pose prior from 148 crystal ligands in the heme frame", "11.8"),
    "ei_sampling": ("extra poses allocated by pool disagreement", "11.8"),
    "mbr_lddt_reevaluation": ("minimum Bayes risk re-scored with LDDT-PLI (exploratory, not declared beforehand)", "11.8"),
}
# The section that explains each experiment that is not described above; its name comes from the titles of the README (cypmode/audit/docs_state.py).
SECTIONS = {
    "aux_chemprop": "9.9", "aux_mlp": "9.9", "aux_neural": "9.9", "permutation_control": "6.8, 9.10", "aux_atomjepa": "9.10", "aux_atomjepa_head": "9.10", "exploratory_atomjepa_head": "9.10", "aux_basic_centre": "9.10",
    "aux_chemprop_foundation": "9.10", "aux_emax": "9.10", "aux_ext_adme": "9.10", "aux_ext_cont": "9.10", "aux_ext_tox": "9.10", "aux_gate_shift": "9.10", "aux_mlp_aux": "9.10", "aux_nodirect": "9.10", "aux_pxr": "9.10", "aux_screen": "9.10",
    "aux_tabicl": "9.10", "aux_tabicl_chemeleon": "9.10", "aux_tabicl_monroe": "9.10", "aux_tabpfn": "9.10", "aux_tabpfn_chemeleon": "9.10", "monroe_embeddings": "9.10", "nonneg_stack": "9.10", "population_screen": "9.10",
    "softmax_selector": "9.10", "aux_pocket_cyp2d6": "9.13", "aux_pocket_cyp3a4": "9.13", "exploratory_pocket_control": "9.13", "exploratory_pocket_first_run_seed0_control": "9.13", "exploratory_pocket_first_run_seed0_own": "9.13",
    "aux_hypothesis": "9.14", "aux_site_cyp2d6": "9.14", "aux_site_cyp3a4": "9.14", "exploratory_hypothesis_permuted": "9.14", "exploratory_site_control": "9.14",
    "ost_vs_ours": "11.4, 11.6", "ost_regate": "11.6", "ost_pool_gates": "11.8, 11.9", "selection_validation_bioemu2": "11.9", "relaxation_gate": "11.10", "ligand_dynamics": "11.13", "pose_pharmacophore_model": "11.11",
}
from cypmode.audit.docs_state import TITLES


def section_of(stem):
    return DESCRIPTION[stem][1] if stem in DESCRIPTION else SECTIONS.get(stem, "?")


def section_key(stem):
    """The first section of a stem as numbers, so that 9.9 comes before 9.10."""
    first = section_of(stem).split(",")[0].strip()
    return tuple(int(part) for part in first.split(".")) if first[:1].isdigit() else (99,)


rows = {}
frame = ledger_rows()
for stem, verdict in zip(frame["stem"], frame["verdict"], strict=True):
    entry = rows.setdefault(stem, {"met": 0, "not met": 0})
    entry["met" if verdict == "MET" else "not met"] += 1
ledger = pd.DataFrame([
    {"idea": DESCRIPTION[stem][0] if stem in DESCRIPTION else TITLES.get(stem, stem), "section": section_of(stem), "verdicts met": v["met"], "verdicts not met": v["not met"]}
    for stem, v in sorted(rows.items(), key=lambda kv: (section_key(kv[0]), kv[0]))
])
show_table(ledger)
met_rows = frame[frame["verdict"] == "MET"]
met_rows = met_rows.assign(order=(met_rows["kind"] == "structure levers").astype(int) * 2 + met_rows["name"].str.endswith("_ost").astype(int)).sort_values("order", kind="stable")  # the classification model first, then the structure track, the repeats under the second scorer last
MET_LABEL = {  # a verdict that is met is named by what it passed, not by the family of the experiment (a family also holds verdicts that were not met)
    "restraint_bond": "the restraint of the iron-nitrogen bond",
    "restraint_contact": "the restraint of the iron-nitrogen contact",
    "restraint_bond_ost": "the same bond restraint under the OpenStructure scorer",
    "restraint_contact_ost": "the same contact restraint under the OpenStructure scorer",
    "force_field_relaxation": "the force-field relaxation of poses",
}


def met_label(stem, name):
    if name in MET_LABEL:
        return MET_LABEL[name]
    replicated = re.search(r"replicate (\\w+)", name)
    if replicated:
        return f"the 10-fold replicate of the {replicated.group(1)} group"
    return TITLES.get(stem, stem)


met = list(dict.fromkeys(met_label(stem, name) for stem, name in zip(met_rows["stem"], met_rows["name"], strict=True)))
met_structure = int((met_rows["kind"] == "structure levers").sum())
met_model = len(met_rows) - met_structure
print("verdicts met:", "; ".join(met))
'''

INNOVATION = '''
# The claims of the innovation summary of the method report, with the number that carries each one; the rows are read from the logs, the manifests and the audit by cypmode/audit/method_report.py.
from cypmode.audit.method_report import INNOVATION_COLUMNS, innovation_rows

claims = pd.DataFrame(innovation_rows(), columns=INNOVATION_COLUMNS)
show_table(claims)
'''

BUILD = '''
# What the repository holds and what the build of this notebook enforces.
print(f"{F.modules} modules in cypmode/, {F.scripts} files in scripts/, {F.jobs} queue jobs, {F.test_files} test files with {F.test_functions} test functions")
'''

SCIENCES = '''
# Where each science of the project is taught: the sections, and the worked example (a cell that computes the idea on a small case or on the project's own data). The table is written by hand; the build checks that
# every section it names exists and holds a code cell, and links each number to the heading exactly as the table of contents does.
import re

from notebooks.notebook_builder import headings, section_of_cells, slug
from notebooks.notebook_cells import ALL_CELLS

MATH, BIO, CHEM, CHEMINF, COMP, STRUCT, ML, ENG = (
    "Mathematics and statistics", "Biology and pharmacology", "Chemistry", "Cheminformatics", "Computational and quantum chemistry", "Structural biology and bioinformatics", "Machine learning", "Engineering of the project")
taught = [
    (MATH, "Rates of change: elimination, binding and inactivation", "1.1 1.2 1.6", "one dose with first-order elimination (1.1); the schemes E + S <-> ES -> E + P (1.2) and E + I <-> EI -> E-I* (1.6) integrated as differential equations"),
    (MATH, "Probability, likelihood and the log-loss", "4.1", "the log-loss of one compound and its derivative p - y, checked against a finite difference"),
    (MATH, "Standard deviation, confidence limits and measurement noise", "2.5 1.7", "what the confidence limits and the standard deviation of a pIC50 are (2.5); a simulated dose-response experiment with 6 percent noise (1.7)"),
    (MATH, "Rank test and permutation test", "5.4 6.8", "the rank test written out on a tiny example and on the real groups, and a permutation test that shuffles the group labels (5.4); a label-permutation control on data with nothing to find (6.8)"),
    (MATH, "The corrected resampled t-test", "6.5", "two classifiers that are equal in truth, compared by cross-validation on many simulated data sets, and why the naive test fails"),
    (MATH, "Multiple comparisons: Holm's adjustment", "6.5 9.11", "a family of comparisons from the project's own logs (6.5); the ledger of the gates with the correction for the number of attempts (9.11)"),
    (MATH, "Logistic regression and its penalties", "4.1 8.2", "the weights fitted by gradient descent and compared with the library (4.1); a toy in which the truth is known, to show why the blend is penalised (8.2)"),
    (MATH, "Optimisation: gradient descent", "4.1 11.10", "the loop that fits the logistic regression (4.1); the same rule lowering an energy with four step sizes, against L-BFGS (11.10)"),
    (MATH, "Linear algebra: eigenvectors and principal components", "4.9", "principal component analysis by hand on the 217 descriptors of all 6895 compounds, compared with scikit-learn"),
    (MATH, "Linear algebra: the Kabsch fit", "11.4", "the rotation that best superposes two sets of points, on a small example checked by hand"),
    (BIO, "The liver and the elimination of a drug", "1.1", "one dose with first-order elimination, C(t) = C0 exp(-kt)"),
    (BIO, "Enzymes and the Michaelis-Menten equation", "1.2", "E + S <-> ES -> E + P integrated as differential equations and compared with the Michaelis-Menten rate"),
    (BIO, "Cytochrome P450, the heme and the catalytic cycle", "1.3", "the conserved heme-binding signature F..G...C.G found in the sequence of CYP3A4 by a pattern, with the cysteine that binds the iron; the cycle drawn in a figure"),
    (BIO, "Drug-drug interactions", "1.4", "the increase in exposure (the AUC ratio) that a competitive inhibitor causes"),
    (BIO, "Reversible inhibition", "1.5", "how the IC50 of a competitive inhibitor depends on the substrate concentration, IC50 = Ki (1 + [S] / KM)"),
    (BIO, "Time-dependent inhibition", "1.6", "E + I <-> EI -> E-I* integrated as differential equations"),
    (BIO, "Dose-response, the Hill equation, the IC50 and the pIC50", "1.7", "a simulated dose-response experiment, ten concentrations in three-fold steps, with its fit"),
    (BIO, "The two assay arms and the potency shift", "1.8 2.3", "the shift that a time-dependent inhibitor shows after a pre-incubation (1.8); the label rule and its single threshold (2.3)"),
    (CHEM, "Organic structure: atoms, bonds, aromaticity and stereochemistry", "2.1", "an example compound read as a graph; aromaticity by counting electrons; one molecule written by many random walks; the two mirror forms of a carbon with four different neighbours"),
    (CHEM, "Functional groups and structural alerts", "1.10 9.15", "the structural alerts of the project against the real labels (1.10); the functional groups of the table against the base rate (9.15)"),
    (CHEM, "Basic amines, pKa and protonation", "5.2 9.10", "the trialkylamine pattern of the project (5.2); the share of charged molecules from the pKa, checked against Dimorphite-DL (9.10)"),
    (CHEM, "Nitrogen-iron coordination, Type I and Type II binding", "1.9 5.1 5.5", "the iron-to-ligand distances in crystal structures of CYP3A4 (1.9); the six patterns of the motif screen (5.1); the iron-nitrogen distance computed by hand and by the library (5.5)"),
    (CHEM, "Reactive metabolites and mechanism-based inactivation", "1.6 1.10", "the kinetic scheme of inactivation (1.6); the structural alerts against the real labels (1.10)"),
    (CHEMINF, "SMILES and molecular graphs", "2.1", "one molecule written by 2000 random walks over its graph"),
    (CHEMINF, "Standardising structures", "2.9", "what the standardised parent does to a salt, on three real compounds"),
    (CHEMINF, "Fingerprints and the Morgan algorithm", "3.1", "the Morgan algorithm written out on a toy that uses nothing but the graph"),
    (CHEMINF, "Tanimoto similarity", "3.2", "the Tanimoto similarity by hand and by RDKit"),
    (CHEMINF, "Descriptors", "3.3", "some of the 217 RDKit descriptors, computed for the example compound"),
    (CHEMINF, "Pharmacophore fingerprints", "3.5", "the example compound as the project's pharmacophore fingerprint sees it"),
    (CHEMINF, "Scaffolds, clusters and similarity-based validation", "3.6 6.4", "Bemis-Murcko scaffolds and Taylor-Butina clusters (3.6); how close a validation compound is to the training compounds, against how close a test compound is (6.4)"),
    (CHEMINF, "Matched molecular pairs", "9.2", "two compounds that differ by one cut bond, found with the matched-pair fragmentation of RDKit"),
    (CHEMINF, "Feature blocks and the feature store", "3.4 3.7", "every block with its real width (3.4); the two promises of the store checked on the real functions (3.7)"),
    (COMP, "Force fields", "11.10", "the terms of a force field put into numbers on one pair of atoms"),
    (COMP, "The Lennard-Jones and Coulomb terms", "11.10", "both terms for one pair of atoms: the bottom of the well at 2^(1/6) sigma found by the formula and by a search, and the Coulomb constant computed from the physical constants"),
    (COMP, "Energy minimisation", "11.10", "gradient descent with four step sizes against L-BFGS, on the Lennard-Jones energy"),
    (COMP, "The relaxation of a predicted pose", "11.10", "one ligand atom too close to a frozen receptor atom, minimised for several strengths of the tether"),
    (COMP, "Partial charges", "10.5", "the quantities of the physics block, with the charge of every atom, computed on three small molecules"),
    (COMP, "The Schrödinger equation and the variational principle", "10.1", "the Schrödinger equation as a matrix eigenvalue problem on a grid; the variational principle on the hydrogen atom"),
    (COMP, "Hartree-Fock and basis sets", "10.2", "Hartree-Fock written out for the smallest molecules, and H2 at several distances"),
    (COMP, "Density functional theory", "10.3", "density functional theory against Hartree-Fock on the water molecule"),
    (COMP, "GFN2-xTB and tight binding", "10.4", "one GFN2-xTB single point on toluene, with the program's own report"),
    (COMP, "Molecular dynamics", "11.13", "none is run in the notebook: the settings and the gate of the experiment are read from the module and from the log of the job, and the integrator is explained in the text"),
    (COMP, "Solvent-accessible surface and cavities", "5.6", "the Shrake-Rupley surface written out; cavity detection on a toy in two dimensions"),
    (STRUCT, "Proteins and the PDB file format", "11.1", "what a predicted complex file holds, read from one of the submitted files"),
    (STRUCT, "mmCIF files", "5.5", "the first atom of a real file laid against the names of the columns, field by field"),
    (STRUCT, "Sequence alignments and co-evolution", "11.2", "a toy protein of 12 positions: what 400 related sequences tell about its shape"),
    (STRUCT, "Co-folding: the pair representation and the triangle update", "11.2", "one triangle update with random weights: relabelling the positions gives the relabelled result"),
    (STRUCT, "The Pairformer", "11.2", "the order of the operations of a block read from the code of OpenFold3, and attention with a bias from the pairs on six tokens"),
    (STRUCT, "Diffusion models", "11.2", "a diffusion sampler in two dimensions whose ideal denoiser is known exactly, for several step scales"),
    (STRUCT, "Boltz-2 and OpenFold3", "11.3 11.9", "the input of a prediction built by the project's own function (11.3); OpenFold3 as it was run for the benchmark (11.9)"),
    (STRUCT, "Structure templates", "11.7", "how different the pockets of the crystal structures are"),
    (STRUCT, "Scoring a pose: RMSD, LDDT-PLI and PoseBusters", "11.4", "the three ideas on small examples checked by hand, and the physical-validity checks as the scorer calls them"),
    (STRUCT, "Pose pools and the choice of a pose", "11.8 11.10", "minimum Bayes risk selection on a toy complex (11.8); the pools of the twenty ligands and the margin rule (11.10)"),
    (ML, "Supervised learning, bias and variance", "4.1", "a polynomial fit whose error on new points is split into bias, variance and noise on a case where the truth is known"),
    (ML, "Cross-validation and nested cross-validation", "4.1 6.2 6.3", "folds and held-out data (4.1); what choosing does to a score on pure noise (6.2); the nested plan on a toy of 24 compounds (6.3)"),
    (ML, "Decision trees", "4.2", "the split search written out for every descriptor and compared with the tree of depth 1 of scikit-learn"),
    (ML, "Gradient boosting", "4.3", "second-order gradient boosting written out in about forty lines"),
    (ML, "Random forests", "4.4", "the variance of an average of correlated estimates, simulated"),
    (ML, "Nearest neighbours and support vector machines", "4.5", "the Tanimoto kernel on the 2334 compounds and the two models built on it"),
    (ML, "Neural networks and backpropagation", "4.7", "a network with one hidden layer written out with its gradient"),
    (ML, "Message passing on graphs", "4.8", "a directed message-passing network, the form that Chemprop uses, with random weights"),
    (ML, "Attention", "4.8 11.2", "self-attention written out on the SMILES of the example compound (4.8); attention with a bias from the pairs on six tokens (11.2)"),
    (ML, "Embeddings and pretrained models", "7.9 9.4", "the ChemBERTa embedding of every labelled compound (7.9); the frozen CheMeleon block (9.4)"),
    (ML, "In-context tabular models", "4.9", "none is run in the notebook: the model needs a separate environment, so the section describes it and the nested evaluation uses it as a candidate"),
    (ML, "Metrics: AUC, MCC and the confusion matrix", "4.10", "the MCC by the formula, by the library and as a correlation coefficient: the same number"),
    (ML, "Calibration", "8.5", "calibration for both enzymes, and isotonic regression by hand with the pool-adjacent-violators algorithm"),
    (ML, "Stacking", "8.2", "a toy in which the truth is known, to show why the blend is penalised"),
    (ML, "Thresholds and calls", "7.5 8.4", "four ways of choosing the threshold on the same probabilities (7.5); the smoothing of the threshold (8.4)"),
    (ENG, "Reproducibility: seeds, threads and solvers", "12.1", "why the same numbers added in a different order give a different total"),
    (ENG, "The nested validation harness", "6.3", "the nested plan on a toy of 24 compounds, and the real plan checked by code"),
    (ENG, "The feature store and the caches", "3.7", "the two promises of the feature store checked on the real functions"),
    (ENG, "Tests and continuous integration", "12.3", "the test suite counted from its files, and one small test file run in a process of its own"),
    (ENG, "The job queue", "12.4", "the helpers of queue/lib.sh run for real in child processes, and the log of the runner read back"),
    (ENG, "The audit", "12.2", "the audit report read as it stands: the counts of its summary and the checks that did not pass"),
    (ENG, "The generated documents", "12.5", "the two generated reports and the generated blocks of the other documents"),
]
titles = {}
for level, text_of_heading, _ in headings(ALL_CELLS):
    found = re.match(r"(\\d+\\.\\d+) ", text_of_heading)
    if level == 2 and found:
        titles[found.group(1)] = text_of_heading
has_code = {label for label, cell in zip(section_of_cells(ALL_CELLS), ALL_CELLS) if cell["cell_type"] == "code"}
named = [s for _, _, sections, _ in taught for s in sections.split()]
assert all(s in titles for s in named), [s for s in named if s not in titles]  # every section named exists
assert all(s in has_code for s in named), [s for s in named if s not in has_code]  # and holds a code cell
science_table = pd.DataFrame([{"Field": field, "Science": science, "Taught in": ", ".join(f"[{s}](#{slug(titles[s])})" for s in sections.split()), "Worked example": example} for field, science, sections, example in taught])
n_sciences, n_fields, n_sections_named = len(taught), len({row[0] for row in taught}), len(set(named))
print(f"{n_sciences} sciences in {n_fields} fields, taught in {n_sections_named} different sections")
show_table(science_table)
'''

CELLS = [
    md("""
# Part 0: Orientation

This part gives the whole picture before any detail: what problem is being solved and why, how the project came to be, what the challenge asks, how the two finished systems are laid
out, what was tried and what came of it, and how this notebook is built. Each box of each system is named here and given a pointer to the section that explains it.
"""),
    md("""
## 0.1 The problem: medicines in the body and why their interactions matter

> **Field:** Pharmacology · Biology

A **drug** is a molecule that changes how something in the body works. After it is swallowed or injected, the body treats it as a foreign substance and deals with it in four stages,
remembered by the abbreviation **ADME**: **absorption** (the drug gets from the gut or the injection site into the blood), **distribution** (the blood carries it to the tissues),
**metabolism** (the body chemically alters it) and **excretion** (the altered drug leaves in urine or bile). This project is about the third stage.
"""),
    figure("bio_adme_route"),
    md("""
Most metabolism is done by **enzymes**, which are proteins that speed up one particular chemical reaction by many orders of magnitude and are not used up by it, so that a small
number of enzyme molecules can process a very large number of drug molecules. In the liver a family of enzymes called **cytochrome P450**, written **CYP**, does most of the chemical
modification of drugs. (The name comes from a pigment, hence "P", that absorbs light of wavelength 450 nanometres when it is bound to carbon monoxide; section 1.3 gives the origin.)
Individual enzymes have names such as CYP3A4 and CYP2D6: the first digit names a family, the letter a subfamily, and the last digit one enzyme within it. These two enzymes are the
subject of this project. A recent preprint on this enzyme (section 0.2) states that CYP3A4 alone metabolises roughly half of all marketed drugs.

Every CYP enzyme has at its centre a **heme**, a flat ring-shaped molecule that holds one iron atom. The chemical reaction happens at that iron atom, and Part I explains how. What
matters at this point is that a molecule which sits on or next to the iron, or which reacts with the enzyme, can stop the enzyme from working.

To **inhibit** an enzyme is to reduce its activity. This matters because one drug can inhibit the enzyme that normally removes another. The second drug is then cleared more slowly, its
concentration in the blood rises, and a dose that was safe on its own can become harmful. This is a **drug-drug interaction**. Drug developers are therefore expected to find out whether
a new candidate inhibits the main CYP enzymes before it reaches patients.

Two kinds of inhibition are distinguished.

- In **reversible inhibition** the effect lasts only while the inhibitor is present. When the inhibitor has been cleared from the body, the enzyme works as before.
- In **time-dependent inhibition**, abbreviated **TDI**, the effect grows with the time the enzyme and the molecule spend together, and it outlasts the molecule. The enzyme, or a
  reactive product that the enzyme itself makes from the molecule, ends up bound so tightly or so permanently that the enzyme stays out of action until the body builds new enzyme,
  which takes days. A drug that is a time-dependent inhibitor can interact with other drugs long after its last dose.

Time-dependent inhibition is also hard to see in a simple test. A reversible inhibitor shows its effect at the moment it is mixed with the enzyme. A time-dependent inhibitor has to be
mixed with the enzyme, left for a while, and only then tested. It is therefore measured in a separate, longer experiment (section 1.8). A computer program that could say beforehand
which molecules are likely to behave this way would let chemists drop or redesign them before spending weeks on making and testing them.

This project asks two questions about molecules given as chemical structures:

1. **Classification.** Is the molecule a time-dependent inhibitor of CYP3A4, yes or no? Is it one of CYP2D6, yes or no?
2. **Structure.** Where do the atoms of the molecule sit when it is bound inside CYP3A4?

Both are posed by a public competition, described in section 0.3.
"""),
    md("""
## 0.2 How this project grew

> **Field:** Overview

The project did not begin as a competition entry. Its first version is described here because its tools are still in the repository and several of its ideas return in later parts.
It was a **heme-coordination motif screen**.

**The starting point: a label that hides the mechanism.** The Therapeutics Data Commons (TDC) distributes yes/no labels saying whether a molecule inhibits each of five CYP enzymes
(the measurements of Veith and co-workers, published in 2009). A "yes" says that the molecule inhibits the enzyme. It does not say how. Pharmacologists distinguish two ways in which
a molecule can bind to the heme region of a CYP enzyme. In **Type I** binding the molecule sits in the active site near the heme, and the iron keeps the water molecule that is
normally attached to it. In **Type II** binding a nitrogen atom of the molecule reaches the iron and takes the water's place: the nitrogen's **lone pair** of electrons (a pair it holds without sharing it in a bond) forms a bond with
the iron. A Type II interaction is much tighter and is often far more significant clinically (de Groot, 2006; the chemistry is in section 1.9).

**The screen.** A heme-binding nitrogen sits in a recognisable group of atoms: an imidazole, a triazole, a pyridine or a thiazole ring. A fast program (`cypmode/metrics/motifs.py`)
searches a molecule for such groups using **SMARTS** patterns, a text notation for chemical substructures (section 5.1). The first measurement was how many of the labelled inhibitors
in the panel carry one.
"""),
    code(PANEL),
    md("""
Only about a quarter to a third of the labelled inhibitors of any enzyme carry such a group. Most of what the yes/no label calls "inhibitor" is therefore something a motif screen
would not expect to bind the iron directly, which is exactly the information a plain yes/no label discards. The screen was kept as a cheap, interpretable signal and not claimed as a
mechanism predictor.

**The structural check.** A screen that finds a nitrogen-containing group says only that binding to the iron is possible. Whether it happens can be checked on a predicted structure.
**Boltz-2** is a **co-folding** model: a neural network that predicts the three-dimensional structure of a protein together with the molecules bound to it, starting from the protein's
sequence and the molecule's structure (section 11.2). It was run for four drugs whose structures bound to CYP3A4 had been determined experimentally by **cryo-EM**, a technique that
images frozen molecules with an electron microscope: ketoconazole, ritonavir, azamulin and vardenafil (the preprint "Structural Plasticity and Ligand Promiscuity of CYP3A4 Revealed by
Cryo-EM", bioRxiv, DOI 10.64898/2026.09.01.748687). The input explicitly included the heme, attached to the enzyme through its cysteine, because a first attempt without the heme
could not have confirmed or refuted anything. No restraint forced the molecule toward the iron, so a short iron-to-nitrogen distance is a prediction and not something the input
imposed. Distances are given in **ångström**, written Å, a unit of 0.1 nanometre; a bond between a nitrogen and an iron atom is about 2 Å long.
"""),
    code(FOUR),
    md("""
Ketoconazole and ritonavir place a nitrogen within bonding distance of the iron, as the screen expected. Azamulin and vardenafil do not, although both match a motif, and each failure has a
documented reason. Azamulin is known to inactivate CYP3A4 by a mechanism that needs the enzyme to turn it into a reactive product (Lim and co-workers, 2005, DOI
10.1124/dmd.104.003475; Sevrioukova, 2019, DOI 10.3390/ijms20174245), so it is a real inhibitor with a mechanism that a screen of the Type II kind mislabels. In vardenafil the nitrogen
group sits inside a fused ring system and does not behave like a free ring nitrogen. One caution applies to both: the cryo-EM preprint itself argues that co-folding models struggle
with CYP3A4 because the enzyme reshapes its pocket on binding, so a single predicted pose cannot settle the question. Section 5.5 returns to these four compounds in detail.

**From a screen to a challenge.** In 2026 the OpenADMET project ran a public challenge on exactly this subject (section 0.3). It offered what the screen had lacked: a large,
consistent set of time-dependent inhibition measurements, a hidden test set that makes evaluation without leakage possible, and a structure-prediction task for CYP3A4. The motif screen became
one block of features for the classification model (sections 3.4 and 7.2); the structure tools built for the four-compound check, which write Boltz-2 inputs with the heme attached,
became the base of the structure entry (Part XI).

**Three habits formed along the way**, and they shape the whole notebook. Every number in a report is copied by code from a log or a result table, never typed. Every proposed
improvement to a model is judged by a **gate**, a pass/fail rule written down before its result exists (section 6.7). And a result that does not help is kept and explained, not dropped (section 0.5).
"""),
    md("""
## 0.3 The challenge

> **Field:** Overview · Statistics

The **OpenADMET CYP blind challenge** is a public competition run by the OpenADMET project, hosted as a Hugging Face Space
(<https://huggingface.co/spaces/openadmet/cyp-challenge>) with its data in a public dataset (<https://huggingface.co/datasets/openadmet/cyp-challenge-train-test>). A **blind** test
means that the organisers keep the correct answers for the test molecules and score every submission themselves; participants cannot see how well they are doing except through the
limited scores the organisers show. Three tracks exist. The facts in this section come from the challenge's FAQ and from the evaluation configuration published in its Space, which the
challenge states is ported from its scoring backend.

| track | task | scored by | entered here |
| --- | --- | --- | --- |
| Direct inhibition | predict how strongly a molecule inhibits each of four enzymes, as a number (regression) | a soft-threshold relative error | no |
| Time-dependent inhibition (TDI) | say yes or no for CYP3A4 and for CYP2D6 | the Matthews correlation coefficient (MCC), averaged over the two enzymes | yes |
| Structure | predict the pose of 20 ligands bound to CYP3A4 | LDDT-PLI, with the ligand RMSD (root-mean-square deviation: the typical distance between the predicted and the true position of the ligand's atoms) after pocket superposition as a secondary score | yes |

The **Matthews correlation coefficient** (**MCC**) is a score for yes/no predictions that is 0 for guessing and 1 for perfect answers, and that stays informative when "yes" is rare
(section 4.10 derives it). **LDDT-PLI** is a score from 0 to 1 for how well the distances between a ligand's atoms and the protein atoms near it are reproduced (section 11.4). A
submitted structure that fails any of the physical-validity checks of the **PoseBusters** program is given the worst possible scores, however close it is to the truth; the program
checks, for example, that no two atoms overlap and that bond lengths are sensible (section 11.4).

The rules that matter for this project are these.

- The test set has 750 molecules and its labels are withheld until the challenge closes. A submission may be made once every 12 hours for each track, and only the latest valid one counts.
- The final deadline is 3 November 2026, 23:59 UTC.
- A submission carries a declaration of whether proprietary data were used. This project uses public data only.
- Publishing the code is encouraged, and it is required for a separate award, the "Innovation in ML" award, which recognises architectural novelty, creative use of data, novel
  uncertainty quantification and clever engineering independently of leaderboard rank. A written method report must be linked for the final leaderboard.

**The data.** The training file holds {{=F.n_train:,}} molecules and the blinded test file {{=F.n_test}}. A molecule is given as a **SMILES** string, a line of text that spells out its
atoms and bonds (section 2.1). The next cell shows what the two files contain.
"""),
    code(FILES),
    md("""
A **pIC50** is a number that says how strongly a molecule acts on an enzyme: the negative base-10 logarithm of the concentration, in moles per litre, that halves the enzyme's
activity, so that a larger pIC50 means a stronger effect (section 1.7). Each molecule has up to two such numbers per enzyme, and many are missing because not every molecule was
measured on every enzyme (section 2.6). The yes/no label for the classification task is not typed in by a person: it follows from the two pIC50 values by a rule that this project
recovered from the data and checked exactly, and that the challenge's own pages confirm (section 2.3). The cell below counts, for the two enzymes of the TDI track, how many training
molecules carry a usable label.
"""),
    code(LABELLED),
    md("""
For CYP3A4 and CYP2D6 the positive molecules are a minority of the labelled ones, and the labelled molecules are themselves a selected subset of the whole file. Both facts shape
every later decision: they are the reason a plain **accuracy** score (the fraction of compounds called correctly) would be misleading, the reason the test set's positive rate matters (section 2.10), and the reason that the project
spends much of its effort on how a model is evaluated (Part VI) and not only on the model.

**The structure track.** The 20 ligands are given as SMILES strings; the answer is a three-dimensional structure file for each ligand bound to CYP3A4. The reference structures are
new cryo-EM structures of CYP3A4 bound to drug-like molecules. Fifteen of the 20 come from the training set of the TDI track, two from its test set and three are additional compounds, a fact
used in section 11.5 to attach what is known about a ligand to its structure.
"""),
    md("""
## 0.4 The whole system on two pages

> **Field:** Overview · Machine learning · Structural biology

Two systems were built, one for each track. Both are drawn here as a row of boxes, and the tables that follow name the section that explains each box. The figures are generated
from the repository's files, so the counts on them are the current ones.

**The classification system** turns a training file into calls on the test molecules.
"""),
    figure("arch_tdi_pipeline"),
    md("""
| box | what it does | explained in |
| --- | --- | --- |
| Training file | {{=F.n_train:,}} molecules with two assay arms per enzyme (measured without and with a head start for the inhibitor) | 2.2, 2.6 |
| Label rule | the yes/no label follows from the two measurements, `s = t - max(d, 4) > log10 2`, and each label gets a confidence from the measurement noise | 2.3, 2.5 |
| Features | every molecule becomes {{=F.n_columns:,}} numbers in {{=F.n_blocks}} blocks: fingerprints of its substructures, calculated properties, alerts for reactive groups, and physics from quantum chemistry | Part III, Part X |
| Candidate models | {{=F.n_models}} models of {{=F.n_families}} kinds (boosted trees, nearest neighbours, support vector machines, extremely randomised trees, a random forest and two boosted regressors of the assay arms), each reading some of the features | Part IV, 8.1 |
| Nested validation | the whole procedure is evaluated on folds that no choice has seen; the evaluation weights match the test set | Part VI |
| Stacked blend | a logistic regression combines the candidates' scores; its penalty is chosen inside the inner folds | 8.2 |
| Gates | each new idea must beat the stack with corrected p < 0.05 at two confidence cut-offs before it is used | 6.7, Parts IX and X |
| Submission | the test molecules are scored, the top fraction is called positive, the files are checked with the portal's own rules | 8.4 to 8.6 |

In plain words: every molecule is first turned into a long list of numbers that describe it. A **fingerprint**, one kind of such numbers, is a list of yes/no answers to the question "does
the molecule contain this particular small group of atoms?". Many different models then each give a score for how likely the molecule is to be a time-dependent inhibitor, and a final
blending step, called **stacking**, learns how much to trust each model. The molecules with the highest blended scores are called positive. Because the true test answers are hidden, the
whole procedure is practised on the training molecules in a way that mimics the hidden test, and every idea that might improve it has to pass a gate first.

**The structure system** turns a ligand into a validated pose.
"""),
    figure("arch_structure_pipeline"),
    md("""
| box | what it does | explained in |
| --- | --- | --- |
| Target | CYP3A4 with its heme; templates built from cryo-EM structures were tried | 11.1, 11.7 |
| Input per ligand | the ligand's SMILES, the heme, the enzyme sequence and, for ligands with a donor nitrogen, a restraint that keeps the nitrogen near the iron | 11.3, 11.6 |
| Boltz-2 | co-folding on the laptop's graphics processor, several samples per ligand | 11.2, 11.3 |
| Validity | PoseBusters is run on every pose, and an invalid pose is never handed in | 11.4 |
| Selection | the first valid pose is taken, with the restrained run first for donor ligands; more elaborate choices were tested and not adopted | 11.8, 11.10 |
| OpenFold3 | a second co-folding model was run for real, benchmarked, and did not help | 11.9 |
| Final zip | {{=F.poses}} structure files, {{=F.pose_failures}} validity failures, rebuilt byte for byte | 11.10, 12.1 |

Underneath both systems sit the engineering parts: the tests, the job queue that ran everything in order, the audit that runs the whole pipeline twice and compares the results
byte for byte, and the reports generated from the logs (Part XII).
"""),
    md("""
## 0.5 The ledger in one table

> **Field:** Overview · Statistics

A **gate** is a pass/fail rule for an experiment, written down before the experiment's result exists. For a proposed addition to the classification model the rule is: it must beat the
reference with a corrected p value below 0.05 at both confidence cut-offs, both when added alone and when added to the assembled stack (section 6.7). For a structure lever the rule is
stated on a benchmark of ligands with experimentally known poses. Fixing the rule first matters because a rule chosen after seeing a result can always be chosen so that the result
passes. The table lists every logged verdict, grouped by experiment. One idea usually yields several verdicts, because it is judged for each enzyme, alone and stacked, and sometimes
again in a replicate with 10 folds; the verdicts of a replicate are counted with the experiment it replicates. The terms in the table are explained in the sections named in its last column.
"""),
    code(LEDGER),
    md("""
{{=F.gates}} verdicts in {{=F.families}} families of ideas were recorded for rules declared beforehand, and {{=F.gates_met}} of them are "met": {{=met_model}} for the classification model and {{=met_structure}} for the structure track. The verdicts
that are met are {{='; '.join(met)}}. All the others, including every more elaborate way of choosing a pose and most additions to the classification model, ended "not met". The table holds ideas only: the check of the pipeline on permuted labels
(which must give chance) and the exploratory second looks, among them the re-scoring of minimum-Bayes-risk pose selection with LDDT-PLI whose rule was not declared beforehand, are checks and not ideas, and they are not in it.

A ledger that is almost entirely "not met" is not a failure of the method. It is what a strict method is for. The classification model that results is deliberately plain: a blend of
{{=F.n_models}} standard models (the default stack; the final CYP3A4 stack adds {{=F.n_final['CYP3A4'] - F.n_models}} members from groups that passed their gates) on a large set of fingerprints and calculated properties. Its distinguishing features are in how it is evaluated, in how the label is modelled, and in the
record of what was tried. Sections 9.10 and 13.1 return to what the ledger does and does not show.
"""),
    md("""
## 0.6 How this notebook is built and how to run it

> **Field:** Software engineering

The notebook is both a text and a program. Its source is a set of Python modules in `notebooks/parts/`, assembled by `notebooks/build_notebook.py`. The builder runs every code cell in
order in one namespace with the repository's root as the working directory, records the printed text, tables and figures as the cell's output, and writes the executed notebook. Four
checks must pass before it is written.

1. **No cell may fail.** A failing cell stops the build, so a number can never be left over from an earlier version of the code.
2. **Every defined term is defined where it is first used.** The list of terms is in `notebooks/parts/glossary.py`; the builder finds the first section that uses each one and fails if
   that section does not contain its bold definition. This is how the rule "no background assumed" is enforced and not merely intended.
3. **Every file of the repository is explained.** The builder lists every module of `cypmode/`, every file of `scripts/`, every job of `queue/jobs/` and every test file, and fails if
   any is not named in the text. Appendix D prints the resulting map from file to section, so nothing in the repository can go unexplained, and anything added later has to be
   explained before the notebook builds.
4. **Every number in a sentence comes from a file.** A sentence such as "{{=F.n_train:,}} molecules" is a template filled in from `notebooks/facts.py`, which reads the repository's own
   files, at build time.
"""),
    code(BUILD),
    md("""
**Where each science is taught.** The notebook assumes no background, so every science that the project uses is taught from its first principles, in the section of the part where it is first needed, with a worked example: a cell that computes the idea on a small case or on the project's own data. The table below is the index. Each row names a science, the sections that teach it (every number is a link to the heading) and what the worked example computes, so that a reader who meets an unfamiliar subject anywhere in the text can look it up here. The build checks that every section named in the table exists and holds a code cell; the description of the example is written by hand. Two sciences have no computed example because the notebook does not run them: the in-context tabular models, which need an environment of their own, and molecular dynamics, whose settings and gate are read from the log of the job. The table says so in those rows.
"""),
    code(SCIENCES),
    md("""
The notebook is rebuilt whenever results, models or tools change, and a last time when everything has finished running, so that the text, the figures and the files of the
submission describe the same state of the work. To rebuild it, run the following from the repository root, with the environment of section A.1 active.

```bash
python -m notebooks.build_notebook     # run every cell, check, write notebooks/cypmode_notebook.ipynb
python -m notebooks.build_exports      # write the single-file HTML and the PDF from the executed notebook
```

The equations in the HTML and the PDF are typeset by KaTeX, a mathematics typesetting library included in `notebooks/_vendor/katex/` with its licence, so that no network connection is
needed. The cells can also be run in an ordinary Jupyter session started in the repository root: the first code cell imports the helpers that put figures and tables into a cell's output.
"""),
    md("""
## 0.7 What the work adds, claim by claim

> **Field:** Overview · Statistics

The method report opens, before its results, with an innovation summary, and the README and the architecture specification carry the same table. A claim of novelty is easy to write and hard to check, so each row of the table
names the number that carries the claim and the log that holds it. The rows are about five things: how the label is modelled (through the two measurements that define it), how the evaluation is made to resemble the scored population (Part VI),
a model of the selection that decided which CYP2D6 compounds were measured (Part IX), how the reliability of a prediction is stated (similarity tiers with intervals over chemical series, a check that probabilities match frequencies, and a committee that was tested and did
not help), and the simple ideas of the structure track (the iron-nitrogen restraint, the validity margin and the relaxation under a registered rule). One row is an attempt that came back inconclusive, the estimate of the positive rate without a
leaderboard (section 2.11); it is in the table because a claim list that holds only successes is not a record.
"""),
    code(INNOVATION),
    md("""
Two things are deliberately absent from the table. It makes no claim about the score on the hidden labels, which is not known until the challenge closes, and it makes no claim that a rule for choosing among predicted poses works: every such rule
failed its gate (section 11.8), and the structure track keeps the model's own ranking among the poses that are valid.
"""),
]
