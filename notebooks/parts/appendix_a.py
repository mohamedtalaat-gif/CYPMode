# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Appendix A: the practical walkthrough. The environments, the map of the repository, every stage as a runnable path, the list of the queue jobs (generated from the jobs themselves and repeated job by job in the text), how a candidate is reproduced, and how a model or a structure lever is added through a gate."""

from notebooks.notebook_builder import code, md

ENVIRONMENTS = '''
# The environments, read from the files that pin them (environment/). A pinned entry is a package with the exact version it was installed at (name==version in a pip file, name=version in a conda file). The file SHA256SUMS holds the SHA-256 of
# every pin file, so that a file that changed since the checksums were written is found, and pyproject.toml lists the optional groups of packages (extras) that pip install -e ".[...]" takes.
import hashlib
import tomllib

env_dir = ROOT / "environment"
recorded_sums = {line.split()[1]: line.split()[0] for line in (env_dir / "SHA256SUMS").read_text().splitlines() if line.strip()}
env_rows = []
for path in sorted(p for p in env_dir.iterdir() if p.name != "SHA256SUMS"):
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if path.suffix == ".txt":
        pinned = sum(1 for line in lines if "==" in line)
    else:
        pinned = sum(1 for line in lines if line.lstrip().startswith("- ") and "=" in line)
    env_rows.append({"file": path.name, "pinned entries": pinned, "SHA-256 equals the recorded one": hashlib.sha256(path.read_bytes()).hexdigest() == recorded_sums.get(path.name)})
env_table = pd.DataFrame(env_rows)
show_table(env_table, floatfmt=".0f")
extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"].get("optional-dependencies", {})
extras_text = ", ".join(f"{name} ({len(packages)} packages)" for name, packages in extras.items())
n_env_files, all_sums_match = len(env_table), bool(env_table["SHA-256 equals the recorded one"].all())
assert all_sums_match  # a changed pin file fails the build
'''

REPO_MAP = '''
# The map of the repository, with the number of source files in each place that holds code. The purpose of each place is written here; the count is made when the cell runs.
repo_places = [
    ("cypmode/challenge", "the TDI track: labels, features, the nested evaluation, every candidate model, the gates, the final build", "*.py"),
    ("cypmode/validation", "the structure track: Boltz-2 and OpenFold3 runs, scoring of poses, the gates, the final recipe", "*.py"),
    ("cypmode/metrics", "motif and amine detection used by the first version of the project", "*.py"),
    ("cypmode/data", "loaders of the public data sets of the first version", "*.py"),
    ("cypmode/audit", "the audit, the method report, the negative-results ledger, the queue check, the readiness scan", "*.py"),
    ("scripts", "the queue runner and its guards, the audit and reproduction scripts, the release script", "*"),
    ("queue/jobs", "one shell script per experiment or build step, run in the order of its number", "*.sh"),
    ("tests", "the test suite", "*.py"),
    ("notebooks/parts", "the text and the code cells of this notebook", "*.py"),
    ("docs", "figures, the card and the film, with the programs that draw them", "*.py"),
]
repo_rows = []
for place, purpose, pattern in repo_places:
    folder = ROOT / place
    files = [p for p in folder.rglob(pattern) if p.is_file() and "__pycache__" not in p.parts] if folder.exists() else []
    repo_rows.append({"place": place, "what it holds": purpose, "source files": len(files)})
repo_table = pd.DataFrame(repo_rows)
show_table(repo_table, floatfmt=".0f")
n_python_files = int(repo_table.loc[repo_table["place"].isin(["cypmode/challenge", "cypmode/validation", "cypmode/metrics", "cypmode/data", "cypmode/audit"]), "source files"].sum())
n_test_files = int(repo_table.loc[repo_table["place"] == "tests", "source files"].iloc[0])
'''

JOB_LIST = '''
# The list of queue jobs, generated from the jobs themselves: the number, the name and the first comment line of every script in queue/jobs, which by the rules of the repository says what the job does and why. Nothing here is typed in; a job
# that is added to the directory is in the table the next time the notebook is built.
import re

job_rows = []
for path in sorted((ROOT / "queue/jobs").glob("*.sh")):
    comment = next((line.lstrip("# ").strip() for line in path.read_text().splitlines()[1:] if line.startswith("#")), "")
    number, _, name = path.stem.partition("_")
    comment = re.sub(r"OCNT-[0-9]+", "an official ligand", comment)  # the table says what a job does and names no compound of the blinded sets
    job_rows.append({"job": number, "name": name, "what it does": comment[:230] + (" ..." if len(comment) > 230 else "")})
job_table = pd.DataFrame(job_rows)
show_table(job_table, floatfmt=".0f")
job_numbers = job_table["job"].astype(int)
ranges = [("000 to 099", 0, 99), ("100 to 199", 100, 199), ("200 to 299", 200, 299), ("900 to 999", 900, 999)]
range_counts = {label: int(((job_numbers >= lo) & (job_numbers <= hi)).sum()) for label, lo, hi in ranges}
n_jobs = len(job_table)
'''

PARSE_TOY = '''
# How a gate reaches the final build, on logs made for the purpose in a temporary folder. The final build (final_tdi_build.parse_gates) reads the line "CYP3A4 GATE (stacked, ...): MET" or "NOT MET" from the log of each group, under the name
# the group's pattern gives. A group of CONFIRMED_GROUPS (the neural pair) counts as passed only if the log of its 10-outer-fold replicate says MET as well; a missing replicate is a failure. A log under any other name is read by nobody.
import tempfile
from pathlib import Path

from cypmode.challenge.final_tdi_build import CONFIRMED_GROUPS, GROUPS as BUILD_GROUPS, parse_gates

MET_LINE = "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET\\n"
NOT_MET_LINE = "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): NOT MET\\n"
REPLICATE_MET = "CYP3A4 GATE (replicate neural, stacked): MET\\n"
gate_rows = []
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    cases = [
        ("a group with a MET line", {"physics_experiment_CYP3A4.log": MET_LINE}),
        ("a group with a NOT MET line", {"mmp_transfer_CYP3A4.log": NOT_MET_LINE}),
        ("the neural pair, MET, no replicate log", {"aux_neural_CYP3A4.log": MET_LINE}),
        ("the neural pair, MET, replicate MET", {"aux_neural_CYP3A4.log": MET_LINE, "replicate_verdict_CYP3A4.log": REPLICATE_MET}),
        ("a MET line in a log with a name that no group reads", {"exploratory_physics_f10_CYP3A4.log": MET_LINE}),
    ]
    for label, logs in cases:
        for old in folder.iterdir():
            old.unlink()
        for name, text in logs.items():
            (folder / name).write_text(text)
        verdicts = parse_gates(folder)
        gate_rows.append({"case": label, "groups the build sees": len(verdicts), "groups that pass": ", ".join(f"{group} ({isoform})" for (group, isoform), v in verdicts.items() if v.get("stacked")) or "none"})
parse_table = pd.DataFrame(gate_rows)
show_table(parse_table, floatfmt=".0f")
n_build_groups = len(BUILD_GROUPS)
confirmed_text = ", ".join(CONFIRMED_GROUPS)
assert parse_table.loc[0, "groups that pass"].startswith("physics") and parse_table.loc[1, "groups that pass"] == "none"
assert parse_table.loc[2, "groups that pass"] == "none" and parse_table.loc[3, "groups that pass"].startswith("neural")
assert parse_table.loc[4, "groups the build sees"] == 0
'''

LEVER_NAMES = '''
# The gates the structure recipe reads, with their recorded verdicts (final/recipe.json). A new lever for the structure track becomes one more line "STRUCTURE GATE <name>: MET|NOT MET" in the log of its own job, and one more entry in the
# mapping of cypmode/validation/final_recipe.py, which is the only place that turns verdicts into choices.
import json

structure_gate_names = list(json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/recipe.json").read_text())["gates"])
n_structure_gates = len(structure_gate_names)
structure_gate_text = ", ".join(structure_gate_names)
'''

JOB_TABLE = r"""
The same list again, written into the text rather than generated when the page is built: one row per job, its number and name, and the first comment line of its script. By the rules of the repository that comment says what the job does and why, so the description is the job's own (a line cut short is marked with " ... "); the table is the same order as the one above, and it is written into the text so that every job of the queue is named in the notebook and the generated table can be checked against the scripts.

| job | the first comment line of its script |
| --- | --- |
| `010_pop_2d6_15fold` | Selection-aware CYP2D6 (panel-block pool): 3 repeats x 5 folds, the gate for the pre-registered decision. |
| `012_pop_2d6_sensitivity` | Same caches: sensitivity to the share of non-hits in the scored set. |
| `013_test_nonhit_share` | Label-shift estimate of how hit-like the blinded test structures are (input to the CYP2D6 A/B decision). |
| `014_hedged_blend` | Rank blend of hits-only and selection-aware CYP2D6 stacks, judged by the worst case across regimes (cache-only, fast). |
| `015_pop_3a4_control` | CYP3A4 control: the test is analogs of 3A4 hits, so no selection-aware gain is expected here. |
| `030_calibration` | Runs the calibration experiment (calibration_experiment.py); the log is calibration.log. |
| `040_selection_aware_v2` | Stronger selection-aware model: cross-isoform hit-profile gate, Heckman-style correction term, gated vs the v1 sets. |
| `045_block_holdout` | Most test-like validation available: hold out the analog-expansion block of the training ids and score the stack on it. |
| `046_pair_conditional_transfer` | SAR-aware pair-conditional label transfer (descriptor deltas); the physics-delta variant runs in job 075. |
| `047_mmp_transfer` | SAR made explicit: matched-molecular-pair transformation features, gated against the base stack. |
| `050_frozen_embeddings` | Frozen CheMeleon embeddings: chemprop extraction in .venv-ml (no LightGBM), evaluation in .venv (no torch). |
| `060_test_likeness_training` | Importance-weighted training from the unlabeled test structures; gated alone and stacked. |
| `070_physics_xtb_block` | GFN2-xTB atom/site/pharmacophore physics block: tblite-only extraction first, then the LightGBM evaluation in separate processes (tblite and LightGBM deadlock in one process). |
| `071_population_physics` | Physics block inside the selection-aware CYP2D6 model (P(hit) and P(label \| hit)), panel-block pool. |
| `072_physics_compact` | Compact physics: the 20 most useful physics columns chosen inside every task. |
| `073_physics_tierb_extract` | Tier-B physics extraction (tblite only, own process): Boltzmann conformers and relaxed hydrogen abstraction. |
| `074_physics_tierb_experiment` | The tier-B physics block for both isoforms, alone and added to the stack (3 repeats); the gate lines are in physics_tierb_&lt;isoform&gt;.log. |
| `075_pair_transfer_physics` | Physics-aware SAR: the pair model whose deltas include the xTB block (needs job 070's cache). |
| `076_xiso_arm_transfer` | I3: arms of the other isoforms, predicted from structure (cross-fitted), as features of the target-isoform classifier; alone and stacked. |
| `078_arm_variants` | Floor-clipped two-arm regressors vs the raw-arm ones, alone and stacked. |
| `080_committee_ei_model` | Probabilistic label-process model: noise-aware committee of two-arm regressors -> P(label) and expected improvement. |
| `081_replicate_grid_f10` | 10-outer-fold replicate: the default candidate grid on its own plan (outer 10, inner 4, 3 repeats), deployment fits included. |
| `082_replicate_physics_f10` | The physics comparisons again on the 10-outer-fold plan (tier A, compact, tier B). |
| `083_replicate_verdict` | The verdict of the 10-outer-fold replicate for both isoforms (replicate_verdict.py), written to replicate_verdict_&lt;isoform&gt;.log. |
| `084_tdi_stack_reevaluation` | Every comparison that goes through the stack (StackedLogit) again, after its logistic meta-learner was made deterministic (tight tolerance, fixed seed; before, liblinear's random coordinate order changed the stack from process to process). Nothing is refitted: the prediction caches are complete, `--run` only evaluates. The verdict lines the final build parses are rewritten, then the replicate verdicts and the final build are redone. |
| `090_final_tdi_build` | Gate-driven TDI candidate files, once per design (5 outer folds, then the 10-fold replicate). Never submits. |
| `096_applicability_domain` | The applicability domain of the 750 blinded compounds, generated from the final build: the reliability of the final stack by nearest-training similarity (held-out outer folds of the nested validation) |
| `097_why_no_gain` | Diagnostics of why each idea that did not join the final stack added nothing: paired change with its corrected p value, redundancy with the final stack, the weight the stack gave it and the smallest effect the nested design could detect. Generated by code; the notebook reads the table. |
| `100_boltz_unguided_resume` | Boltz-2 on the ligands of the crystal-anchored benchmark without a restraint (5 diffusion samples each, one sample at a time), resuming after the failed entries are cleared. |
| `110_boltz_restraint_bond` | Boltz-2 on the benchmark ligands with the iron-nitrogen bond restraint (3 samples each); exits non-zero when a ligand failed, so that the runner tries again. |
| `120_boltz_restraint_contact` | Boltz-2 on the benchmark ligands with the iron-nitrogen contact restraint (3 samples each); exits non-zero when a ligand failed, so that the runner tries again. |
| `130_boltz_diversity_pilot` | Boltz-2 on the ligands of the diversity pilot, one sample at a time; exits non-zero when a ligand failed, so that the runner tries again. |
| `133_series_benchmark_runs` | Boltz-2 on three crystal-anchored analog pairs (6 ligands, 4 samples each): the benchmark of the series-consensus rule. |
| `135_boltz_cryoem_pilot` | S3: Boltz-2 with the cryo-EM CYP3A4 entries as protein templates, on the three pilot ligands (3 templates each, 1 sample each). |
| `140_score_restraint_and_pilot` | Scores the poses of the unrestrained, bond-restraint, contact-restraint and diversity-pilot runs on the crystal-anchored benchmark and writes the comparison of the restraints and the pilot report. |
| `142_series_scoring` | True ligand RMSD of every series pose against its crystal pose. |
| `144_score_cryoem_pilot` | S3: true ligand RMSD of the cryo-EM template runs, then the pool comparison against the unguided pools (same thresholds as the other pilot levers). |
| `149_openfold3_memory_probe` | OpenFold3 memory probe. The largest benchmark ligand (bench-YNV, 46 heavy atoms; its first run stopped with "Invalid buffer size: 9.27 GiB") |
| `150_openfold3_benchmark` | OpenFold3 on the 10 benchmark ligands, 3 samples each, in the memory mode the probe (job 149) found. Ligands that already have structures are skipped, a ligand without one is retried, and the job fails unless every ligand ends with a structure. |
| `160_score_all_and_report` | Scores every pose of the crystal-anchored benchmark and writes the table of selection outcomes. |
| `165_structure_gate_verdicts` | Machine-readable restraint and pilot verdicts (STRUCTURE GATE lines) from the scored pose tables of job 140. |
| `168_pose_awareness` | Physics, molecule and pharmacophore terms of the pose pools, judged on the pooled crystal-anchored benchmark. |
| `169_series_consensus` | SAR-aware selection: poses that agree with the poses of the ligand's analogs. |
| `170_mbr_selection_validation` | Validation of the minimum-Bayes-risk selection on the scored benchmark poses (selection_validation.py). |
| `172_ei_sampling` | Expected-improvement sampling: extra Boltz samples go to the ligands whose pose pool disagrees most (needs the pooled benchmark poses). |
| `173_mbr_lddt_reevaluation` | Minimum Bayes risk and the restraint re-scored with LDDT-PLI against the crystal (exploratory: the registered gate used RMSD). No compute beyond scoring. |
| `175_heme_frame_prior` | The heme-frame prior on the scored benchmark poses (heme_frame_prior.py). |
| `176_restraint_expansion_prepare` | The confirmation of the restraint on crystal ligands that were not used to decide it (plan registered in data/structure_track/inputs/restraint_expansion_plan.json). |
| `177_boltz_restraint_expansion_primary` | Boltz-2 on the primary stratum of the confirmation ligands, both conditions, two samples each (about 13 minutes per sample on this Mac). |
| `178_score_restraint_expansion` | Scores the confirmation runs (RMSD with the existing rescoring harness, LDDT-PLI with lddt_to_crystal) and writes the registered verdicts the final recipe reads. |
| `179_requeue_final_recipe` | Once the restraint confirmation is scored (178), the final recipe and its update (180, 185) must run again so that the structures follow the confirmed (or withdrawn) restraint: their old "done" markers are cleared and the runner, which rescans the jobs after every job, runs them next. |
| `180_structure_final_recipe_run` | Run the frozen recipe on the 20 official ligands and validate every structure. Needs the recipe module. |
| `185_structure_recipe_update` | The final recipe once more after the engine-consensus and selection verdicts exist (jobs 149-170): it reads the verdicts again, adds what they call for to the Boltz runs of job 180 (OpenFold3 samples for the official ligands, MBR or rank-term selection) and rewrites the candidate zip. |
| `186_boltz_restraint_expansion_secondary` | The secondary stratum of the confirmation ligands (31 to 34 heavy atoms): reported next to the primary one, never used for the decision, so it runs after the final recipe. |
| `187_score_restraint_expansion_secondary` | The scoring again with the secondary stratum included (the verdicts of the primary stratum do not change). |
| `188_requeue_final_chain` | After the confirmation, the final recipe and the secondary stratum are finished (180 to 187): the chain that describes and packages the final state runs again, in order, on a quiet repository (audit and tests, paired summaries, method report, public-readiness scan, figures, notebook, the real-structure clips and the film, and the final sync of submissions/). |
| `189_nodirect_labels` | The 1,249 CYP3A4 compounds without a direct curve, used through what their TDI-arm pIC50 implies (194 certain negatives; soft labels for the rest from a first-stage model on the pool rows of each task), alone and inside the stack. |
| `190_official_extra_audit` | The official files the first release did not read (single-concentration screen, Emax, direct inhibition, reaction phenotyping): identifiers, overlap with the blinded test set, agreement with the TDI file. Fails when any check fails. |
| `191_basic_centre_extract` | Basic-centre block (Dimorphite-DL protonation states, centre classes, charge-to-ring geometry) for the training and blinded test compounds. |
| `192_structure_margin_aware_zip` | Margin-aware pose choice: every pool pose is checked with the organisers' own PoseBusters call, the final recipe keeps only valid poses whose relative distance to the receptor is at least 0.80 (the scorer's threshold is 0.75) and rewrites the candidate zip, and the zip is verified the way the scorer reads it. |
| `193_nonneg_stack` | Non-negative stacking as an alternative selector, from the cached predictions of the default stack (same folds, same yardstick). |
| `194_gate_shift` | The two-part form of the label, P(t > 4.301) x P(t - d > log10 2): the gate reads every compound with a TDI-arm curve. Validated alone and inside the stack. |
| `195_screen_block` | The official single-concentration screen as a predicted block (cross-fitted inside every task); validated alone and inside the stack, both isoforms. |
| `196_screen_selection_2d6` | The real screen inside the selection-aware CYP2D6 model (gate learned from the screen; predicted screen in the end-to-end classifier), on the panel-block pool. |
| `197_emax_block` | The official Emax readings of both arms as a predicted block, validated alone and inside the stack. |
| `198_mlp_addition` | The multitask network (cache complete) added to the default stack, both isoforms. |
| `199_basic_centre_experiment` | The basic-centre block added to feature sets A and A+fps, validated alone and inside the stack (it matters most for CYP2D6), both isoforms. |
| `200_permutation_control` | Negative control of the whole nested pipeline: the pool labels are permuted at random and the procedure is repeated; the outer AUC must be 0.5. |
| `201_tabicl_CYP3A4` | TabICL candidate completed for every task of the nested plan (time-boxed chunks, resumable; the memory guard stays on). |
| `202_chemeleon_speed` | The speed test and the decision on the fine-tuned CheMeleon (cypmode/challenge/chemeleon_speed.py, decided 2026-10-16 before any measurement): one task that is already in the cache from the processor is trained again on the Apple GPU, in a scratch directory, and the rule of the module decides whether stage 1 for CYP3A4 runs on the GPU, on the processor, or is not run for cost. It starts once the cheap candidates have settled, so that nothing else is heavy while it is timed. CYP2D6 was dropped from the CheMeleon design on the same date. |
| `202_tabicl_CYP2D6` | TabICL candidate completed for every task of the nested plan (time-boxed chunks, resumable; the memory guard stays on). |
| `203_chemprop_CYP3A4` | CheMeleon, stage 1 of its staged design (cypmode/challenge/chemeleon_stage.py): the 25 cross-validation tasks of repeat 0 for CYP3A4, time-boxed chunks, resumable. The plain Chemprop cache is complete. Whether it runs, and on which device, is decided by job 202 (cypmode/challenge/chemeleon_speed.py): "mps", "cpu" or "not run: cost" (then nothing is run here and the ledger says so). |
| `204_chemprop_CYP2D6` | CheMeleon stage 1 for CYP2D6 is not run: the enzyme was dropped from the CheMeleon design on 2026-10-16 (plain Chemprop is at chance on CYP2D6, outer AUC 0.54 and 0.53; the frozen CheMeleon block reaches 0.63; see cypmode/challenge/chemeleon_speed.py). The four tasks already in the cache stay there and are not used. |
| `205_tabicl_addition` | TabICL added to the default stack once its cache is complete (jobs 201 and 202). |
| `206_chemprop_addition` | Plain Chemprop (cache complete on the nested plan) added to the default stack, both enzymes; CheMeleon has its own staged jobs (203, 204, 236 to 239). |
| `207_ost_vs_ours` | OpenStructure's own LDDT-PLI (the organisers' scorer, in the conda environment `ost`) against the local values, pose by pose, on the confirmation ligands. |
| `208_mlp_aux_CYP3A4` | Multitask network with the official screen and Emax readings as auxiliary heads, completed for every task of the nested plan (time-boxed chunks, resumable). |
| `209_mlp_aux_CYP2D6` | Multitask network with the official screen and Emax readings as auxiliary heads, completed for every task of the nested plan (time-boxed chunks, resumable). |
| `210_mlp_aux_addition` | The multitask network with auxiliary heads added to the default stack once its cache is complete (jobs 208 and 209). |
| `211_monroe_extract` | Frozen Monroe (GRIT) embeddings of the training and blinded test compounds, in the isolated environment .venv-monroe (weights read as plain tensors). |
| `212_monroe_block_experiment` | The Monroe embedding block added to feature sets A and A+fps (and alone), validated alone and inside the stack, both isoforms. |
| `213_tabicl_monroe_CYP3A4` | TabICL on the frozen Monroe embedding, completed for every task of the nested plan (time-boxed chunks, resumable). |
| `214_tabicl_monroe_CYP2D6` | TabICL on the frozen Monroe embedding, completed for every task of the nested plan (time-boxed chunks, resumable). |
| `215_tabicl_monroe_addition` | TabICL on Monroe embeddings added to the default stack once its cache is complete (jobs 213 and 214). |
| `217_pxr_block` | The PXR-induction readings of the same compound library (pEC50 and Emax; training set and the two unblinded test phases of the PXR challenge) as a predicted block, alone and inside the stack. |
| `218_relaxation_gate` | Force-field relaxation of predicted ligand poses in the frozen pocket (OpenMM, in the conda environment openmm), tested on the poses of the crystal-anchored ligands before any use on the official ones: the scorer's PoseBusters verdict and OpenStructure's LDDT-PLI, before and after, with the registered gate of cypmode/validation/relax_gate.py. |
| `219_ext_cont_extract` | The continuous responses and potencies of the NCATS qHTS panel (PubChem AID 1851) and of the Tox21 CYP screen as label-free blocks: models trained on external compounds only predict them for every challenge compound. |
| `220_ext_cont_experiment` | The continuous external block, and then the Tox21 block as a separate test, added to the feature sets that carry the binary external block, alone and inside the stack (pre-registered gate of addition_experiment), both isoforms. |
| `221_tabpfn_CYP3A4` | TabPFN (Prior Labs) completed for every task of the nested plan (time-boxed chunks, resumable; the memory guard stays on). The weights are under Prior Labs' non-commercial licence and are fetched by the package after the licence has been accepted at ux.priorlabs.ai; the job waits until the file data/tdi_track/inputs/tabpfn_licence_accepted.txt exists (it is created once the licence has been accepted). |
| `222_tabpfn_CYP2D6` | TabPFN (Prior Labs) completed for every task of the nested plan (time-boxed chunks, resumable; the memory guard stays on). The weights are under Prior Labs' non-commercial licence and are fetched by the package after the licence has been accepted at ux.priorlabs.ai; the job waits until the file data/tdi_track/inputs/tabpfn_licence_accepted.txt exists (it is created once the licence has been accepted). |
| `223_tabpfn_addition` | TabPFN added to the default stack once its cache is complete (jobs 221 and 222); the pre-registered gate of addition_experiment decides. |
| `227_openadmet_surrogates` | logD and mouse brain protein binding predicted for every challenge compound by OpenADMET's surrogate Chemprop models (two label-free columns), then the gated test of adding them to feature sets A and A+fps, both isoforms. |
| `228_atomjepa_extract` | Frozen Atom-JEPA embeddings (all 8 blocks, mean over atoms) of every challenge compound, in chunks that are saved as they finish (an interrupted run resumes), then the gated test of adding them to feature sets A and A+fps. About 0.5 s per molecule on the Apple GPU: under an hour. The full fine-tuning of the authors is not affordable here (7.4 s per molecule per step on the CPU; see the module docstring). |
| `229_atomjepa_head_CYP3A4` | A small attention readout trained on the cached per-atom features of the frozen Atom-JEPA encoder, completed for every task of the nested plan (time-boxed chunks, resumable). The encoder does not run again. |
| `230_atomjepa_head_CYP2D6` | A small attention readout trained on the cached per-atom features of the frozen Atom-JEPA encoder, completed for every task of the nested plan (time-boxed chunks, resumable). The encoder does not run again. |
| `231_atomjepa_head_addition` | The Atom-JEPA attention readout added to the default stack once its cache is complete (jobs 229 and 230); the pre-registered gate decides. |
| `232_neural_f10_fill_CYP3A4` | Confirmatory replicate of the neural pair on CYP3A4 (checklist, group neural): plain Chemprop and the multitask network on the 10-outer-fold plan (155 tasks each, deployment fits included), time-boxed chunks, resumable. |
| `233_neural_addition` | The neural pair (multitask network and plain Chemprop, caches complete) added to the default stack on the nested plan, both enzymes; the ledger rows of the 15-fold design. |
| `234_neural_addition_f10` | The neural pair on the 10-outer-fold plan (CYP3A4): the confirmatory comparison against the default stack, declared before any 10-fold result existed. |
| `235_neural_replicate_verdict` | The replicate verdicts again, now with the neural pair (CYP3A4 only; the physics lines are recomputed from the same logs). |
| `236_chemeleon_stage1_screen` | Stage 1 of CheMeleon: the stacked comparison on repeat 0 (5 outer folds) for CYP3A4. Written to a .txt file on purpose: a stage 1 screen is not a pre-registered gate and must not enter the ledger. |
| `237_chemeleon_stage2_CYP3A4` | Stage 2 of CheMeleon (CYP3A4): repeats 1 and 2 and the deployment fits, only if the rule of cypmode/challenge/chemeleon_stage.py asks for it after stage 1; otherwise nothing is run. |
| `238_chemeleon_stage2_CYP2D6` | Stage 2 of CheMeleon (CYP2D6; the enzyme was dropped from the design on 2026-10-16, so no stage 1 file exists and the rule below always says stop): repeats 1 and 2 and the deployment fits, only if the rule of cypmode/challenge/che ... |
| `239_chemeleon_addition` | CheMeleon added to the default stack (15 folds, the ordinary gate) for each enzyme whose stage 2 ran; the others stay "screened on 5 folds, not continued". |
| `240_softmax_selector` | The softmax-temperature selector (T small: the best model, T large: an equal-weight committee) against the logistic blend, from the cached predictions of the default stack; also the exploration-exploitation curve. |
| `241_ost_regate` | The registered consequence of job 207's verdict: the confirmatory restraint gates repeated with OpenStructure's LDDT-PLI (a few seconds, no model run). |
| `242_data_checker` | The organisers' data checker (OpenADMET/challenge-data-checker, Apache-2.0, in its own environment .venv-checker with Python 3.12): the official training, Emax and direct-inhibition files against the blinded test file. |
| `243_structure_relaxation_final` | The force-field relaxation (gate of job 218, MET) applied to the 20 poses of the final recipe under the rule registered in cypmode/validation/relax_final.py, and the final zip rebuilt from the poses that pass it; the candidate zip is checked with the portal's rules and the scorer's own PoseBusters call before it replaces anything, and the recipe's own zip is kept as structures_unrelaxed.zip. |
| `244_ost_pools_score` | OpenStructure's LDDT-PLI and BiSyRMSD for the 167 poses that the gates of the pools were judged on (about 4 s per pose, measured on three poses), in time-boxed chunks, resumable. The registered consequence of the verdict of job 20 ... |
| `245_ost_pool_gates` | The gates of the pools repeated on OpenStructure's yardstick (after a check that the same code reproduces every original verdict on unchanged tables); nothing the gates wrote before is touched. |
| `246_tabicl_chemeleon_CYP3A4` | TabICL on the frozen CheMeleon embedding (128 label-free principal components next to the quantile-transformed descriptors), completed for every task of the nested plan (time-boxed chunks, resumable). |
| `247_tabicl_chemeleon_CYP2D6` | TabICL on the frozen CheMeleon embedding (128 label-free principal components next to the quantile-transformed descriptors), completed for every task of the nested plan (time-boxed chunks, resumable). |
| `248_tabicl_chemeleon_addition` | TabICL on the frozen CheMeleon embedding added to the default stack once its cache is complete (jobs 246 and 247); the pre-registered gate of addition_experiment decides. |
| `249_tabpfn_chemeleon_CYP3A4` | TabPFN on the same frozen CheMeleon representation as job 246/247, completed for every task of the nested plan (time-boxed chunks, resumable; the licence marker of job 221 is needed, see there). |
| `250_tabpfn_chemeleon_CYP2D6` | TabPFN on the same frozen CheMeleon representation as job 246/247, completed for every task of the nested plan (time-boxed chunks, resumable; the licence marker of job 221 is needed, see there). |
| `251_tabpfn_chemeleon_addition` | TabPFN on the frozen CheMeleon embedding added to the default stack once its cache is complete (jobs 249 and 250); the pre-registered gate of addition_experiment decides. |
| `252_atomjepa_head_f10_fill_CYP3A4` | Exploratory replicate of the Atom-JEPA readout on CYP3A4 (checklist, Atom-JEPA item; declared 2026-10-17, after the registered 5-fold gate was NOT MET): the cached readout completed for every task of the 10-outer-fold plan (time-boxed chunks, resumable). The encoder does not run again. |
| `253_atomjepa_head_f10_exploratory` | The exploratory comparison of the Atom-JEPA readout with the default stack on the 10-outer-fold plan (CYP3A4 only), declared before any 10-fold number of the readout existed. It cannot bring the group into the stack: the registered 5-fold gate is NOT MET, and the log is named so that no gate parser reads it (exploratory_*); the ledger lists it under that name. |
| `285_label_shift` | An estimate of the positive rate of the scored set that reads no leaderboard (label shift, cross-fitted soft adjusted classify-and-count), compared with the rate of the final build by a rule fixed before the first run. |
| `286_ligand_dynamics` | Ligand dynamics in the frozen pocket (the rule of docs/structure_track/structure_ligand_dynamics.md, committed before any number): the pools of the static pharmacophore gate, 20 ps of the ligand in its frozen pocket for every pose, the dynamic scores and the gate. |
| `295_lint_cleanup` | Lint cleanup of the Python modules (decided 2026-10-17). It runs after every experiment job and before the final chain (job 299), so that the audit and the test suite of the chain run on the cleaned code. |
| `299_requeue_final_chain` | After the experiments of jobs 190 to 206: the final TDI build reads every gate verdict again (a group whose gate is MET joins the stack), then the chain that describes and packages the final state runs again, in order, on a quiet repository, and the pre-submission gate closes it. |
| `900_tests_and_audit` | The full audit with the test suite, twice, byte-for-byte; only when every earlier job is finished (a frozen state) and the notebook is written. |
| `905_recompute_paired_summaries` | Older significance tables re-analysed from their saved per-fold scores with the paired corrected t (no refits). |
| `910_method_report` | Method report and negative-results ledger regenerated from the final logs and audit (job 900 runs first). |
| `921_docs_state` | The generated blocks of README.md and CYPMode_Architecture_Specification.md (the final stack per isoform, the verdicts of the gates, the structure recipe): rewritten from the final build, the logs of the gates and the recipe, then checked against their sources; the pages of docs/map are written and checked with them. Runs after the method report, which reads the same logs. |
| `930_public_readiness` | Scan for private paths, credentials, foreign e-mail addresses, oversized files; findings go to the log. |
| `940_figures_build` | The illustrations (docs/figures/PLAN.md): every generator, the catalog check, light and dark SVG and PNG, the animations. Runs after every result is final, so the figures show the final numbers. |
| `941_card_build` | The README card, built after the figures from the final numbers and recorded in audit/artifacts.json (the pre-submission gate checks the record against the numbers). |
| `942_build_notebook` | The notebook (notebooks/PLAN.md): execute every cell against the final files, run the checks (cells, glossary, coverage), write the .ipynb, then the HTML and PDF exports (the build writes them itself when every check passes, so a ... |
| `943_blender_clips` | The cinematic molecule clips of the film (docs/video/STORYBOARD.md): the CYP3A4 enzyme with its heme and a Type II ligand drawn by Blender with the Molecular Nodes add-on from the real coordinates; the film falls back to the PyMOL pictures of job 944 when a clip is missing. |
| `944_pymol_clips` | The real-structure stills and turntables of the film (docs/video/STORYBOARD.md): the heme pocket of CYP3A4 and the pose pools of the official ligands, rendered with PyMOL from the coordinates. |
| `945_video_render` | The animation (docs/video/STORYBOARD.md): every frame drawn by code, encoded to mp4, plus a gif preview and a poster frame. |
| `950_submissions_final_sync` | The very last job: submissions/ is rewritten from the final TDI candidates and the final structure zip, each checked with the portal's own rules (it never uploads anything). |
| `960_pre_submission_gate` | The last gate: whether the queue, the audit, the submission files, the documents, the notebook and the film are finished, consistent and fresh. |
"""

CELLS = [
    md("""
# Appendix A. Practical walkthrough

This appendix is the path through the repository for a reader who wants to run it, check it or extend it. It is not a second document next to the notebook: every table in it is produced from the repository when the notebook is built, so that it cannot describe a repository that is not the one beside it. A.1 says how the software is installed and pinned. A.2 is the map of the directories. A.3 gives every stage of the project as the command that runs it. A.4 lists every job of the queue. A.5 shows how the final build reads the verdict of a gate, on logs made for the purpose. A.6 and A.7 say how a model for the TDI track and a lever for the structure track are added, and what must be written before either is run.

## A.1 Environments and pins

> **Field:** Software engineering

Software changes: a package that is updated a year later can give a different number from the same code. The remedy is to **pin** the environment, which means to record the exact version of every package that was installed, in a **lock file** that anyone can install from. The repository holds four pin files in `environment/`: `requirements-lock.txt` for the main environment (the TDI track and the tests), `requirements-lock-ml.txt` for the neural models (Chemprop, TabICL, TabPFN, CheMeleon), `openmm-environment.yml` for the force-field relaxation of 11.10 and `ost-environment.yml` for the scoring with OpenStructure. The last two are conda environments, which can install programs that are not Python packages. Boltz-2 and OpenFold3 run in **virtual environments** of their own (a folder with its own copy of Python and its own packages, so that installing one tool cannot break another), `.venv-boltz` and `.venv-openfold3`, which are outside the pin files; that is a gap in the record, and the versions that ran are in the manifests of the structure track.

The next cell reads the files in `environment/`, counts the pinned entries of each and checks each file against the checksum recorded in `SHA256SUMS`. The optional groups of packages that `pip install -e ".[...]"` accepts are listed from `pyproject.toml`.
"""),
    code(ENVIRONMENTS),
    md("""
There are {{=n_env_files}} pin files, and every one equals its recorded checksum (the cell requires it). The optional groups of the package are {{=extras_text}}. The quick start of the README installs the main environment with `pip install -e ".[challenge,validate]"`; the other environments are created from their own files. A job whose environment is missing does not run: its first lines (`need_file`) make it wait, so that a missing tool is a visible delay and not a wrong result.

The reason for several environments is practical. A package such as LightGBM, which runs on every thread of the processor, and a package such as PyTorch, which also uses a library for parallel loops, can disturb each other when both are loaded in one process; the repository learned this when a test of the Atom-JEPA readout, run beside the LightGBM tests, hung for hours. 12.3 tells the story; the consequence here is that the tests that load PyTorch or the quantum-chemistry package run in processes of their own, and `CONTRIBUTING.md` says which.

## A.2 The map of the repository

> **Field:** Software engineering

The repository has the code of two tracks, the tools that check them, the queue that runs them, and the text. The next cell lists the places that hold code, says in a line what each holds, and counts the files, so that the map and the repository cannot drift apart.
"""),
    code(REPO_MAP),
    md("""
The code of the five packages is {{=n_python_files}} Python files, and the test suite is {{=n_test_files}} files. The data are not in this table because they are not code: `data/tdi_track/inputs` and `data/structure_track/inputs` hold the files of the challenge and the public data that were added to it, the `outputs` folders hold what the jobs wrote (the caches of the nested evaluation, the logs the gates read, the final candidate files), and `submissions/` holds the files in the form the portal takes. Appendix D, printed by the builder, maps every one of these files to the section of this notebook that explains it.

## A.3 Every stage as a runnable path

> **Field:** Software engineering

The project is run in the order that follows; each command is resumable, which means that it keeps what it has finished and a second run continues where the first stopped.

```bash
python3 -m venv .venv && source .venv/bin/activate        # 1. the main environment
pip install -e ".[challenge,validate]"
python -m pytest tests -q                                  # 2. the tests (the isolated files as CONTRIBUTING.md says)
bash scripts/reproduce_tdi.sh                              # 3. raw challenge files -> validated TDI candidate files
bash scripts/queue_start.sh                                # 4. the queue: every other experiment, in the order of its number
bash scripts/queue_status.sh                               #    what is done, running, waiting
python -m cypmode.audit.checklist                         # 5. every queue job names modules and scripts that exist
bash scripts/run_audit.sh                                  # 6. the audit, generated twice and compared byte for byte
python -m cypmode.audit.method_report                      # 7. the method report and the negative-results ledger
python -m cypmode.audit.public_readiness                   # 8. the scan for what must not be published
python -m notebooks.build_notebook                         # 9. this notebook
```

Step 3 is the TDI path alone, from the raw files to the candidate files: it builds the nested grid for both isoforms and then runs the jobs numbered below 100. The structure track is run by the jobs of the hundreds that the table in A.4 names (the Boltz-2 and OpenFold3 runs and their scoring), and it needs the environments of A.1 and, for the runs themselves, a machine with the memory that 11.9 describes. The queue runs everything in the order of the job number, one job at a time, and the final jobs (900 to 960) wait until the notebook is written, so that the audit, the method report and the film are made from the state that the text describes.

## A.4 The jobs of the queue

> **Field:** Software engineering

The table is generated from the jobs. A job is a shell script; it starts with `source queue/lib.sh`, which gives it the helpers described in 12.4, and its first comment line says what it does.
"""),
    code(JOB_LIST),
    md(JOB_TABLE),
    md("""
The queue holds {{=n_jobs}} jobs: {{=range_counts['000 to 099']}} numbered below 100 (the TDI experiments and the final TDI build), {{=range_counts['100 to 199']}} from 100 to 199 and {{=range_counts['200 to 299']}} from 200 to 299 (the structure track, the additions of 9.9 and 9.10, the housekeeping jobs), and {{=range_counts['900 to 999']}} from 900 (the final chain). A job that is finished leaves an empty file with its name in `queue/state/done`; a job that is not ready ends with the status 75 and is tried again in the next round; any other failure is tried once more and then recorded in `queue/state/failed`. 12.4 explains why the queue is built so.

## A.5 How a gate reaches the final build

> **Field:** Software engineering

A new idea becomes part of the final model only through the line that its log writes: "CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET" or "NOT MET". The next cell shows how the final build (`cypmode/challenge/final_tdi_build.py`) reads those lines, on logs made for the purpose in a temporary folder, including the two rules that matter.
"""),
    code(PARSE_TOY),
    md("""
The final build knows {{=n_build_groups}} groups, each with the name of the log it reads; it forms the candidate sets of Part VIII, evaluates them by the same nested procedure as everything else and keeps the best eligible one. The groups in `CONFIRMED_GROUPS` ({{=confirmed_text}}) pass only with a replicate that is also MET, and a log under a name that no group reads is not read at all.

The first two rows are the plain rule. The third and fourth are the rule for a group that was found after the 15-fold results had been looked at (9.9): the gate alone is not enough, the replicate on the 10-outer-fold plan has to say MET as well. The last row is the one that makes a second look at an idea safe: a log named `exploratory_...` is written, listed in the ledger and read by no parser, so that however it comes out it cannot change the model. The queue has one such pair of jobs, the replicate of the Atom-JEPA readout (9.10), and it is written exactly so.

## A.6 Adding a model to the TDI track through a gate

> **Field:** Software engineering

A model for the TDI track is added in four steps, and the order is the point.

1. **Declare.** Before any number of the new model exists, its declaration says what it is, where it comes from, the comparison that decides it (alone, and added to the stack, on both isoforms), and the rule it must pass. The declaration is written in the header of the module that runs the comparison and in the header of its queue job. The commit that holds the declaration is dated before the commit that holds its result, and the history is never squashed, so that the order can be checked.
2. **Register.** The model is written as a candidate (`Candidate` in `cypmode/challenge/nested_runners.py`; the torch-side ones in `nested_runners_torch.py`, the cached ones in `cached_candidates.py`), added to a group, and the group gets its log name in `final_tdi_build.GROUPS`. The test suite checks that every registered block computes on a small set of molecules and the tests of the final build check the registered groups and the names of their logs.
3. **Run.** Two queue jobs: one that fills the cache of predictions for every task of the nested plan (in time-boxed chunks, resumable), and one that runs `python -m cypmode.challenge.aux_experiments --experiment <name> --isoform <iso>`, which compares the model alone and added to the default stack and writes the gate line.
4. **Let the build decide.** Nothing is done by hand: the next run of the final build reads the line, and the model joins the stack or it does not. A model that fails stays in the ledger of negative results with its numbers.

## A.7 Adding a lever to the structure track

> **Field:** Software engineering

A lever for the structure track (a restraint, a way of sampling more poses, a rule for choosing among poses) follows the same logic with different files. The lever is a change to the input or to the selection (`build_input.py`, `pose_selection.py` and their neighbours in `cypmode/validation`); a job runs it on the crystal-anchored benchmark and scores it; the scoring job writes one line "STRUCTURE GATE <name>: MET|NOT MET" with the rule that was written before the run; and `final_recipe.decide` maps the verdicts to the choices of the final recipe. The next cell lists the gates that the recipe of this project reads.
"""),
    code(LEVER_NAMES),
    md("""
The recipe reads {{=n_structure_gates}} gates: {{=structure_gate_text}}. A lever that is MET switches something on in the recipe; a lever that is NOT MET changes nothing, which is what the recipe of 11.10 shows. Every pose of every pool is audited under the scorer's own PoseBusters call (11.10), so that a lever cannot buy accuracy with a pose that the scorer would reject.

*On what actually ran.* The cells above read `environment/`, `pyproject.toml`, `queue/jobs/` and `final/recipe.json`, and call `final_tdi_build.parse_gates` on logs made in a temporary folder. The commands of A.3 are shown, not run, by the notebook; the audit (12.2) and the test suite (12.3) are what run them.
"""),
]
