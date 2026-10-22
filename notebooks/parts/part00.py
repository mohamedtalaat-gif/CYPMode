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

from cypmode.audit.method_report import gate_rows

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
rows = {}
for log, gate, verdict in gate_rows():
    stem = re.sub(r"(_f10)?\\.log$", "", re.sub(r"_(CYP3A4|CYP2D6)(?=(_f10)?\\.log)", "", log))
    entry = rows.setdefault(stem, {"met": 0, "not met": 0})
    entry["met" if verdict == "MET" else "not met"] += 1
ledger = pd.DataFrame([
    {"idea": DESCRIPTION.get(stem, (stem, ""))[0], "section": DESCRIPTION.get(stem, ("", "?"))[1], "verdicts met": v["met"], "verdicts not met": v["not met"]}
    for stem, v in sorted(rows.items(), key=lambda kv: (DESCRIPTION.get(kv[0], ("", "99"))[1], kv[0]))
])
show_table(ledger)
met = [name for _, name, verdict in gate_rows() if verdict == "MET"]
print("gates met:", ", ".join(met))
'''

BUILD = '''
# What the repository holds and what the build of this notebook enforces.
print(f"{F.modules} modules in cypmode/, {F.scripts} files in scripts/, {F.jobs} queue jobs, {F.test_files} test files with {F.test_functions} test functions")
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
consistent set of time-dependent inhibition measurements, a hidden test set that makes honest evaluation possible, and a structure-prediction task for CYP3A4. The motif screen became
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
| Structure | predict the pose of 20 ligands bound to CYP3A4 | LDDT-PLI, with the ligand RMSD after pocket superposition as a secondary score | yes |

The **Matthews correlation coefficient** (**MCC**) is a score for yes/no predictions that is 0 for guessing and 1 for perfect answers, and that stays honest when "yes" is rare
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
| Candidate models | {{=F.n_models}} models of five kinds (boosted trees, nearest neighbours, support vector machines, extremely randomised trees, a random forest), each reading some of the features | Part IV, 8.1 |
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
{{=F.gates}} verdicts were recorded for rules declared beforehand, and {{=F.gates_met}} of them are "met": {{=', '.join(met)}}. Both belong to the structure track. All the others, every
addition to the classification model and every more elaborate way of choosing a pose, ended "not met". The re-scoring of minimum-Bayes-risk pose selection with LDDT-PLI is marked
exploratory because its rule was not declared beforehand, so it is shown but not counted.

A ledger that is almost entirely "not met" is not a failure of the method. It is what a strict method is for. The classification model that results is deliberately plain: a blend of
{{=F.n_models}} standard models on a large set of fingerprints and calculated properties. Its distinguishing features are in how it is evaluated, in how the label is modelled, and in the
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
The notebook is rebuilt whenever results, models or tools change, and a last time when everything has finished running, so that the text, the figures and the files of the
submission describe the same state of the work. To rebuild it, run the following from the repository root, with the environment of section A.1 active.

```bash
python -m notebooks.build_notebook     # run every cell, check, write notebooks/cypmode_notebook.ipynb
python -m notebooks.build_exports      # write the single-file HTML and the PDF from the executed notebook
```

The equations in the HTML and the PDF are typeset by KaTeX, a mathematics typesetting library included in `notebooks/_vendor/katex/` with its licence, so that no network connection is
needed. The cells can also be run in an ordinary Jupyter session started in the repository root: the first code cell imports the helpers that put figures and tables into a cell's output.
"""),
]
