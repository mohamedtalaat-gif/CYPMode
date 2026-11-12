# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XIII (second part): after the deadline. The plan written before the labels were read and the gates (13.5), the TDI track on the released labels (13.6), the structure track on the released structures (13.7) what the analysis does and does not show (13.8) and an exploratory comparison with the poses that the first entry of the structure board published (13.9). Every number is read from the tables that the analysis wrote."""

from notebooks.notebook_builder import code, md

FACTS = r'''
# The numbers of 13.5 to 13.9, read from the tables, the logs and the plan of the analysis after the deadline (cypmode/post_deadline/): the gates, the TDI questions and the structure questions. Nothing is typed in; an assertion stands
# wherever the text below reads a direction or a verdict, so that a different result fails the build and the text is rewritten.
import json
import re

import pandas as pd

from cypmode.post_deadline.report import describe_extra_calls

post_dir = ROOT / "data/post_deadline/outputs"
plan_text = (ROOT / "docs/post_deadline/PLAN.md").read_text()
final_board = json.loads((ROOT / "data/post_deadline/final_board.json").read_text())["tracks"]  # the row of the entries on the final leaderboards of the organisers (cypmode/post_deadline/board.py)
board_tdi, board_structure = final_board["classification"], final_board["structure"]
n_questions = len(re.findall(r"^## Question \d+", plan_text, re.M))
deviations_text = plan_text.split("## Deviations from the plan")[1].split("## What follows")[0]
n_deviations = len([p for p in deviations_text.split("\n\n") if p.strip().startswith("Question")])

tdi_gate = json.loads((post_dir / "tdi_gate.json").read_text())
macro_a, macro_b, macro_board = tdi_gate["macro_mcc"]["A"], tdi_gate["macro_mcc"]["B"], tdi_gate["board"]
assert tdi_gate["submitted"] == "A" and not tdi_gate["registered"]["passed"] and tdi_gate["registered"]["within_tolerance"] == ["A", "B"], "the text of 13.5 tells how the registered rule failed and the exact match succeeded"
exact_difference = abs(macro_a - macro_board)
tdi_scores = pd.read_csv(post_dir / "tdi_board_scores_A.csv").set_index("Endpoint")
tdi_diff = pd.read_csv(post_dir / "tdi_question1_difference.csv").set_index("endpoint")
tdi_rates = pd.read_csv(post_dir / "tdi_question1_rates.csv").set_index("endpoint")
rule = pd.read_csv(post_dir / "tdi_question2a_agreement.csv").set_index("isoform")
scored_set = pd.read_csv(post_dir / "tdi_question2b_scored_set.csv")
ceiling = pd.read_csv(post_dir / "tdi_question3_ceiling.csv").set_index("endpoint")
n_agree, n_disagree = int(rule["agree"].sum()), int(rule["disagree"].sum())
n_not_computable = int(rule["not_computable"].sum())
n_labelled_3a4, n_labelled_2d6 = int(rule.loc["CYP3A4", "labelled"]), int(rule.loc["CYP2D6", "labelled"])
assert n_disagree == 0 and (scored_set["only_project"] == 0).all(), "the text says that the rule never disagrees and that the project keeps no compound that the organisers leave out"
kept = scored_set.assign(share=lambda t: t["both"] / (t["both"] + t["only_organisers"])).set_index(["isoform", "z0"])["share"]
macro_diff = tdi_diff.loc["macro"]
assert tdi_diff.loc["macro", "verdict"] == "no difference shown" and tdi_diff.loc["CYP3A4_is_TDI", "mean_difference"] == 0, "the text says that the two candidates are not told apart and that they are identical for CYP3A4"
mcc_submitted, mcc_ceiling = float(ceiling["mcc_submitted"].mean()), float(ceiling["mcc_ceiling"].mean())
gap_to_first = board_tdi["first"]["MA-MCC"] - macro_board  # the macro MCC of the entry that ranks first on the final board
assert 0 < mcc_ceiling - mcc_submitted < gap_to_first / 2, "the text says that the threshold explains less than half of the distance to the first entry"

structure_gate = json.loads((post_dir / "structure_gate.json").read_text())
versions = json.loads((post_dir / "structure_versions.json").read_text())
poses = pd.read_csv(post_dir / "structure_poses_scored.csv")
arm_mean = poses[poses["arm"].isin(["submitted", "baseline", "unguided_first"])].groupby("arm")[["LDDT-PLI", "BiSyRMSD"]].mean()
q5a = pd.read_csv(post_dir / "structure_question5a.csv")
assert (q5a["verdict"] == "no difference shown").all(), "the text says that no comparison with a plain Boltz-2 pose shows a difference"
q5a_lddt = q5a[q5a["metric"] == "LDDT-PLI"].set_index("comparison")
pool = pd.read_csv(post_dir / "structure_question5b.csv").set_index("ligand")
pool_best, pool_random, pool_submitted = float(pool["best"].mean()), float(pool["random"].mean()), float(pool["submitted"].mean())
recovered = 100 * (1 - (pool_best - pool_submitted) / (pool_best - pool_random))
n_pool_poses = int(pool["poses"].sum())
coordination = pd.read_csv(post_dir / "structure_question5c.csv")
n_compounds = len(coordination)
n_truth_coordinated, n_submitted_coordinated = int(coordination["truth_coordinated"].sum()), int(coordination["submitted_coordinated"].sum())
n_both, n_false_positive = int((coordination["truth_coordinated"] & coordination["submitted_coordinated"]).sum()), int((~coordination["truth_coordinated"] & coordination["submitted_coordinated"]).sum())
n_missed = int((coordination["truth_coordinated"] & ~coordination["submitted_coordinated"]).sum())
extra_calls = describe_extra_calls(post_dir)
assert n_missed == 0 and n_false_positive > n_both, "the text says that the restraint found every coordinated compound and called more compounds coordinated than the structures show"
resolution = json.loads((post_dir / "structure_resolution.json").read_text())
n_in_training = int(coordination["in_training"].sum())
gap_to_tier_one = board_structure["last_of_first_tier"]["LDDT-PLI"] - board_structure["values"]["LDDT-PLI"]["mean"]
assert 0 < gap_to_tier_one < 0.01 and board_structure["tier"] == 2 and board_tdi["tier"] == 2, "the text says that the structure entry lies just below the first tier and that both entries are in the second tier"
resolution_min, resolution_max = float(coordination["resolution"].min()), float(coordination["resolution"].max())
assert abs(tdi_rates.loc["CYP2D6_is_TDI", "called_A"] - tdi_rates.loc["CYP2D6_is_TDI", "true_positive_rate"]) < 5e-5, "the text says that the CYP2D6 call rate equals the true share to the fourth digit"

gate_table = pd.DataFrame(
    [
        {"track": "TDI, macro MCC", "scored here": macro_a, "board": macro_board, "difference": macro_a - macro_board},
        {"track": "structure, mean LDDT-PLI", "scored here": structure_gate["scored"]["LDDT-PLI"], "board": structure_gate["board"]["LDDT-PLI"], "difference": structure_gate["difference"]["LDDT-PLI"]},
        {"track": "structure, mean BiSyRMSD (angstroms)", "scored here": structure_gate["scored"]["BiSyRMSD"], "board": structure_gate["board"]["BiSyRMSD"], "difference": structure_gate["difference"]["BiSyRMSD"]},
    ]
)

external_gate = json.loads((post_dir / "structure_external_gate.json").read_text())
external_paired = pd.read_csv(post_dir / "structure_external_paired.csv")
external_ligands = pd.read_csv(post_dir / "structure_external_ligands.csv")
external_counts = pd.read_csv(post_dir / "structure_external_counts.csv")
vs_submitted = external_paired[external_paired["comparison"] == "external minus submitted"].set_index("metric")
vs_baseline = external_paired[external_paired["comparison"] == "external minus baseline"].set_index("metric")
assert external_gate["passed"] and not external_gate["posebusters_failures"], "the text says that the score of the published poses reproduces the board value of their entry and that no pose fails PoseBusters"
assert vs_submitted.loc["LDDT-PLI", "verdict"] == "first better" and vs_submitted.loc["BiSyRMSD", "verdict"] == "no difference shown" and vs_submitted.loc["LDDT-LP", "verdict"] == "no difference shown", "the text says that only the LDDT-PLI separates the published poses from the submitted ones"
assert vs_baseline.loc["LDDT-PLI", "verdict"] == "first better" and vs_baseline.loc["BiSyRMSD", "verdict"] == "first better", "the text says that the published poses are better than the plain Boltz-2 pose in LDDT-PLI and BiSyRMSD"
n_above_pool = int(external_ligands["external_above_pool_best"].sum())
pool_low, pool_high, pool_total = int(external_ligands["pool_poses"].min()), int(external_ligands["pool_poses"].max()), int(external_ligands["pool_poses"].sum())
top3_share = float(external_ligands["external_minus_submitted"].nlargest(3).sum() / external_ligands["external_minus_submitted"].sum())
n_external_called, n_external_true = int(external_counts["True"].sum()), int(external_counts.loc[external_counts["truth_coordinated"], "True"].sum())
n_differ_calls = int((external_ligands["external_coordinated"] != external_ligands["submitted_coordinated"]).sum())
assert 0 < vs_submitted.loc["LDDT-PLI", "low"] < 0.01 and n_above_pool > len(external_ligands) / 3, "the text says that the lower end of the interval lies just above zero and that the published pose lies above the best pose of the pool in more than a third of the ligands"
external_table = external_paired.rename(columns={"mean_difference": "mean difference", "low": "2.5%", "high": "97.5%", "first_higher": "ligands higher"})[["comparison", "metric", "mean difference", "2.5%", "97.5%", "verdict", "ligands higher"]]
'''

GATE_TABLE = r'''
show_table(gate_table, floatfmt=".5f")
'''

MD_13_5 = """
## 13.5 After the deadline: a plan that was written before the labels were read

> **Field:** Statistics · Data science

*Where this section sits:* 13.1 to 13.4 end the project as it stood on the day of the deadline: everything it claims there is a claim about a procedure, because the labels of the test compounds were not visible. On 4 and 5 November 2026 the organisers published them, together with the 20 structures that the structure track is scored against and the final leaderboards, on which the TDI entry ranks {{=board_tdi['rank']}} of {{=board_tdi['entries']}} (the second of {{=board_tdi['tiers']}} tiers) and the structure entry {{=board_structure['rank']}} of {{=board_structure['entries']}} (the second of {{=board_structure['tiers']}} tiers). For the first time the project can measure what it could only argue. This section says how that was done without letting the answer steer the question, and 13.6 and 13.7 give the results.

**Pre-registration.** A **pre-registration** is a plan that is written down, with its questions, its measures and its decision rules, and made public before the data are looked at. Its purpose is to remove one freedom: once a result is known, it is easy to find a reason why the measure should have been another one, or why a subset should have been left out, and every such reason moves the result a little towards what was hoped. The plan of this analysis is `docs/post_deadline/PLAN.md`; it holds {{=n_questions}} questions, and it was pushed to the public repository before a program of the project had read a label or a released structure. The code that answers the questions was written and tested on invented tables first, and committed before the first label was read.

**What the plan promises, and what it forbids.** The analysis only measures. It fits no model, chooses no threshold and selects no candidate, it changes no submitted file and no number of an earlier release, and a question is not changed after its first result is computed. There is a reason for the last rule that goes beyond tidiness: the labels are now public, so any model whose choices were influenced by them is no longer blind. A number from such a model would measure how well the model can be tuned to a known answer, which is a different thing from how well it predicts an unknown one.

**The gate: reproduce a number that is already public.** Before any question is answered, the scoring has to reproduce a value that the organisers have already published for the submitted entries; otherwise nothing that follows could be compared with the board. The scoring is the organisers' own, the functions of the evaluation code of their tutorial repository at a named commit, run without change. For the TDI track the leaderboard value is the mean over 1,000 **bootstrap** samples (resamplings of the compounds with replacement, 4.4) of the macro MCC; the organisers fix the seed of the resampling, so every entry is scored on the same samples. A fixed seed has a consequence that the plan did not foresee: a file that is scored as the board scored it must reproduce the board value to the last digit, and a file that differs in a few calls must not. The next table shows the gates of both tracks.
"""

MD_13_5_B = """
For the TDI track the file A gives a macro MCC of {{=macro_a:.7f}} and the board shows {{=macro_board:.7f}}, a difference of {{=exact_difference:.0e}}: the file that the board scored is the file A, which is the file that the manifest of the submissions names first. The registered rule did not see it. It asked that a candidate lie within {{=tdi_gate['registered']['tolerance']}} of the board value, and the file B, which differs from A in some CYP2D6 calls only, gives {{=macro_b:.7f}}; both lie inside the tolerance, so the rule could not tell the files apart and the gate, as registered, was not met. The plan says that the analysis then stops and the difference is written down, and that is what happened: the identification of the file was changed to an exact match of the mean and of the standard deviation, which separates the two files, and nothing else was changed. The change was made after the result of the gate was seen, and the plan records it as such. It is one of {{=n_deviations}} deviations written in the plan.

For the structure track the gate is met within the tolerances that were registered (0.005 for the LDDT-PLI and 0.1 angstrom for the BiSyRMSD), but it is not exact: the mean LDDT-PLI is {{=structure_gate['scored']['LDDT-PLI']:.5f}} against {{=structure_gate['board']['LDDT-PLI']:.5f}} on the board. The organisers state, in a reply in their challenge channel on 5 November, that the board was scored with OpenStructure 2.11.1 and PoseBusters 0.6.5. The first run used OpenStructure 2.12.0; a second environment with 2.11.1 gave the same values to seven digits, so the difference does not come from the version of the program. It is reported and not explained; the poses of 13.7 were scored with {{=versions['openstructure']}}, the version that the organisers name.
"""

MD_13_6 = """
## 13.6 The TDI track on the released labels

> **Field:** Statistics · Pharmacology · Cheminformatics

*Where this section sits:* 13.5 fixed the questions and showed that the scoring reproduces the board. The released file for the TDI track holds, for each of the 750 test compounds, the label, the two measured potencies that the label is computed from (2.3) and their standard deviations. That is enough to ask five things of the submitted entry, each with its decision rule written in advance.

**The two candidates.** The final build left two files (13.4), A and B, that differ only in some CYP2D6 calls; the file A was submitted. On the released labels, the macro MCC of B minus that of A is {{=macro_diff['mean_difference']:+.4f}}, with a 95% interval from {{=macro_diff['low']:+.4f}} to {{=macro_diff['high']:+.4f}} over a **paired bootstrap** (the two files are scored on the same resampled compounds, so that the difference is not blurred by which compounds were drawn). The interval contains zero: no difference is shown. For CYP3A4 the two files are identical, so the whole difference is in CYP2D6, where {{=int(tdi_rates.loc['CYP2D6_is_TDI', 'scored'])}} compounds are scored. The alternative that was declared before any result existed cannot be shown to have done better or worse.

**The calls and the prevalence.** The share of true positives among the scored compounds is {{=tdi_rates.loc['CYP2D6_is_TDI', 'true_positive_rate']:.4f}} for CYP2D6 and {{=tdi_rates.loc['CYP3A4_is_TDI', 'true_positive_rate']:.4f}} for CYP3A4. The submitted file calls {{=tdi_rates.loc['CYP2D6_is_TDI', 'called_A']:.4f}} of the CYP2D6 compounds positive, the true share to the fourth digit, and {{=tdi_rates.loc['CYP3A4_is_TDI', 'called_A']:.4f}} of the CYP3A4 compounds, which is more than the true share by design: the final build chose the multiplier of the call rate that maximised the MCC of the nested validation (8.4), and a rate above the prevalence is what that rule gave. The prevalence itself had been recovered, before the labels existed, from the accuracy, precision and recall that the public leaderboards publish (2.10); for CYP2D6 the recovered value and the true one agree.

**The label rule.** The rule recovered from the training file (2.3) was applied to the two measured potencies in the released file, and its label was compared with the organisers' label, compound by compound. Of the compounds whose two values are present, {{=n_agree}} agree and {{=n_disagree}} disagree; another {{=n_not_computable}} compounds carry a label but lack one of the two values, so the rule cannot be applied to them. A rule that was found by looking at a training file and then reproduces every label of a file it has never seen is the strongest evidence that it is the rule.

**The scored set, and an assumption that did not hold.** The challenge FAQ says that only compounds whose label can be assigned with confidence contribute to the score. The project read this as a filter on the confidence of the label and built it into its yardstick (2.5, 6.4): a compound whose shift lies within the noise of the threshold is left out, at one of two cut-offs of the confidence $z$. The released file says what the organisers do: they score every compound that carries a label, which is {{=n_labelled_3a4}} compounds for CYP3A4 and {{=n_labelled_2d6}} for CYP2D6, nearly the whole test set. The confidence rule of the project keeps {{=100*kept[('CYP3A4', 1.0)]:.0f}}% and {{=100*kept[('CYP3A4', 1.96)]:.0f}}% of them for CYP3A4, and {{=100*kept[('CYP2D6', 1.0)]:.0f}}% and {{=100*kept[('CYP2D6', 1.96)]:.0f}}% for CYP2D6, at the two cut-offs, and it never keeps a compound that the organisers leave out. The consequence is for the cross-validated estimates of the earlier parts: they were computed on compounds that are easier to label than the compounds that are scored, so they describe an easier problem than the one the board measures. The limitation L3 of the method report says so.

**The threshold.** How much of the distance to the leading entries is a matter of where the line was drawn between a positive and a negative call? For each isoform the stored score of the submitted file can be cut at every possible threshold, and the best of them on the released labels is a **ceiling**: a value that uses the labels and cannot be reached in practice. The submitted calls give a mean MCC of the two isoforms of {{=mcc_submitted:.4f}} and the best threshold would have given {{=mcc_ceiling:.4f}}. The call rate alone accounts for at most {{=mcc_ceiling - mcc_submitted:.4f}}, while the entry that ranks first on the final board lies {{=gap_to_first:.3f}} above this one. The rest of the distance is not a matter of the threshold.
"""

MD_13_7 = """
## 13.7 The structure track on the released structures

> **Field:** Structural biology · Computational chemistry · Statistics

*Where this section sits:* 11.4 explained how a pose is scored and 11.6 told why the final recipe forces a contact between a nitrogen of the ligand and the iron of the heme. The released structures, {{=n_compounds}} models of the complexes that the organisers determined by cryo-electron microscopy, are the reference that the board scored against, so the recipe can finally be compared with what it was trying to reproduce. {{=n_in_training}} of the {{=n_compounds}} compounds are in the training file of the challenge. The models record the resolution of their maps, between {{=resolution_min:.1f}} and {{=resolution_max:.1f}} angstroms where it is recorded, and the organisers state that they are the models of the original deposits, not re-refined; at this resolution the position of a ligand carries an uncertainty that is not small.

**The recipe against plain Boltz-2.** The structure entry lies {{=gap_to_tier_one:.4f}} of LDDT-PLI below the last entry of the first tier, a distance far inside the error bar of a single entry, which is a reminder of how little 20 compounds separate. The organisers publish the Boltz-2 baseline that they ran for the 20 compounds, and the project's own runs hold the pose that unguided Boltz-2 ranks first. Their mean LDDT-PLI is {{=arm_mean.loc['baseline', 'LDDT-PLI']:.4f}} and {{=arm_mean.loc['unguided_first', 'LDDT-PLI']:.4f}}, the submitted poses give {{=arm_mean.loc['submitted', 'LDDT-PLI']:.4f}}. The paired difference of the submitted pose minus the published baseline is {{=q5a_lddt.loc['submitted minus baseline', 'mean_difference']:+.4f}}, with a 95% interval from {{=q5a_lddt.loc['submitted minus baseline', 'low']:+.4f}} to {{=q5a_lddt.loc['submitted minus baseline', 'high']:+.4f}}, and the submitted pose is the higher of the two in {{=int(q5a_lddt.loc['submitted minus baseline', 'first_higher'])}} of {{=n_compounds}} compounds; against the unguided pose the difference is {{=q5a_lddt.loc['submitted minus unguided_first', 'mean_difference']:+.4f}}, from {{=q5a_lddt.loc['submitted minus unguided_first', 'low']:+.4f}} to {{=q5a_lddt.loc['submitted minus unguided_first', 'high']:+.4f}}. The BiSyRMSD, the secondary measure, gives the same verdict. Neither interval excludes zero, so the verdict is that no difference is shown. With 20 compounds the interval of a single entry is about plus or minus 0.04, and the organisers' own post says the same; a difference smaller than that cannot be called either way. What the numbers do not show is that the recipe, with its restraint, its choice among poses and its relaxation, improved on Boltz-2 as it comes.

**Where the limit lay.** The pool of a compound is the set of poses that the final recipe chose from, {{=n_pool_poses}} poses over the 20 compounds. Over the compounds, the best pose of the pool has a mean LDDT-PLI of {{=pool_best:.4f}} and a pose drawn at random from the pool {{=pool_random:.4f}}; the submitted pose, {{=pool_submitted:.4f}}, sits between them and recovers {{=recovered:.0f}}% of the distance from a random pose to the best one. The pool held good poses; the choice among them was the limit. 11.8 is the section that spent its effort on the choice among poses, and the released structures show that this was where most of the room lay, with the reference that the selection did not have.

**The iron of the heme.** The restraint of 11.6 asks the model to place a nitrogen of the ligand next to the iron, because many inhibitors of CYP3A4 coordinate it. In the {{=n_compounds}} released structures, {{=n_truth_coordinated}} ligands have a nitrogen within the cut-off of the project of an iron; the submitted poses call {{=n_submitted_coordinated}} ligands coordinated. All {{=n_truth_coordinated}} of the true ones are among them, and {{=n_false_positive}} more are called coordinated that the structures place beyond the cut-off: {{=extra_calls}}. A distance within half an angstrom of the cut-off, in maps of this resolution, does not exclude a contact, so these compounds are not shown to make no contact. The restraint therefore found every contact that the structures show and called a contact on {{=n_false_positive}} more compounds, which is the cost of a restraint that applies to every ligand with a donor nitrogen. The rank correlation between the resolution of a map and the LDDT-PLI of the submitted pose is {{=resolution['spearman_rho']:+.3f}} over {{=resolution['compounds']}} compounds; the models of {{=resolution['without_resolution']}} compounds record no resolution.
"""

MD_13_8 = """
## 13.8 What the analysis does and does not show

> **Field:** Statistics · Data science

*Where this section sits:* 13.5 to 13.7 reported what the released labels and structures say about the two submitted entries. This section sorts what they support.

**What the released labels support.** The scoring of the project is the organisers' scoring, and for the TDI track it reproduces the board to the last digit. The label rule is the organisers' rule. The call rate of CYP2D6 was right and the call rate of CYP3A4 was a deliberate overcall. The candidate that was kept in reserve cannot be shown to have done better or worse. The confidence filter of the yardstick, an assumption of the earlier parts, was wrong in a way that matters: the cross-validated estimates are those of an easier problem than the scored one.

**What they do not support.** They do not say that the project would have ranked higher with another threshold: the ceiling is computed with the labels and no threshold chosen without them reaches it, and in any case it explains little of the distance to the first entry. They do not rank the recipe of the structure track above Boltz-2 as it comes; 20 compounds do not resolve differences of this size. They say nothing about the live half of the test set, because the list of the compounds that the live board scored is not among what the organisers published; the question of whether the file was affected by the live board is dropped, as the plan says, and no substitute split was made.

**What follows.** Nothing in this analysis changed a submitted file or a number of an earlier release. If a result points to a cause that an experiment could address, the plan says how the experiment would be written: as a new section of the plan, with its rule, before it runs, judged by cross-validation on the training data like every other experiment of the project, with the test labels used at most once at the end and reported as obtained after the deadline and with knowledge of the labels. The restraint that forces a contact on compounds that do not make one, and the selection among the poses of a pool, are the two places where such an experiment would start.

*On what actually ran.* `cypmode/post_deadline/released.py` (`fetch`, `verify`) fetches the files that the organisers released and records their SHA-256 values; `cypmode/post_deadline/tdi.py` (`board_scores`, `gate`, `paired_difference`, `rule_agreement`, `scored_set_agreement`, `threshold_ceiling`) answers the questions of 13.6 with the organisers' scoring; `cypmode/post_deadline/structure.py` (`score_submission`, `pose_items`, `score_poses`, `paired_difference`, `pool_gaps`, `iron_nitrogen_distance`, `coordination_table`) and `cypmode/post_deadline/structure_worker.py`, which runs in the environment that holds OpenStructure and PoseBusters, answer the questions of 13.7; `cypmode/post_deadline/board.py` (`track_entry`, `build`) reads the row of the entries from the leaderboard files of the organisers into `data/post_deadline/final_board.json`; `cypmode/post_deadline/report.py` writes section 15 of the method report and the block of the README from the tables. The queue jobs `907_post_deadline_released`, `908_post_deadline_tdi` and `909_post_deadline_structure` run them before the method report, and the tests are in `tests/test_post_deadline_board.py`, `tests/test_post_deadline_released.py`, `tests/test_post_deadline_tdi.py`, `tests/test_post_deadline_structure.py`, `tests/test_post_deadline_report.py` and `tests/test_post_deadline_jobs.py`. The plan, with its deviations, is `docs/post_deadline/PLAN.md`.
"""

EXTERNAL_TABLE = r'''
show_table(external_table, floatfmt="+.4f")
'''

MD_13_9 = """
## 13.9 The published poses of the first entry: an exploratory comparison

> **Field:** Structural biology · Statistics · Data science

*Where this section sits:* 13.7 compared the submitted poses with a plain Boltz-2 pose and found that the pool of poses held better ones than the choice among them. This section adds one comparison that the plan of 13.5 did not contain. The entry that ranks first on the structure board published the 20 poses of its submission with its report, and the function that reproduces the board for the submitted archive can be applied to them. The comparison is **exploratory**, which means that it is a measurement made after the answers were known. It can point at a cause but it cannot test one, because a question asked after the data were seen has had the data to shape it; for the same reason the plan records it in its section on deviations, and the six paired comparisons below are not corrected for their number.

**The gate.** The file is fetched by size and checksum and is not stored in the repository, because the repository of the entry carries no licence. Scored with the function that scored the submitted archive, it gives a mean LDDT-PLI of {{=external_gate['scored']['LDDT-PLI']:.5f}} against {{=external_gate['board']['LDDT-PLI']:.5f}} on the board and a mean BiSyRMSD of {{=external_gate['scored']['BiSyRMSD']:.4f}} against {{=external_gate['board']['BiSyRMSD']:.4f}}, with no pose failing PoseBusters. The two sets of poses are therefore scored by one function, and that function is the one of the board. A **paired bootstrap** takes the 20 ligands, draws 20 of them with replacement 10,000 times, and records the mean difference of the two sets of poses each time; the 2.5th and the 97.5th percentiles of those means are the interval, and an interval that excludes zero says that the sign of the difference survives the resampling.
"""

MD_13_9_B = """
**The comparison.** Against the submitted poses the LDDT-PLI of the published poses is higher by {{=vs_submitted.loc['LDDT-PLI', 'mean_difference']:.4f}}, with an interval of {{=vs_submitted.loc['LDDT-PLI', 'low']:+.4f}} to {{=vs_submitted.loc['LDDT-PLI', 'high']:+.4f}}, and it is higher in {{=int(vs_submitted.loc['LDDT-PLI', 'first_higher'])}} of the 20 ligands; the lower end lies only {{=vs_submitted.loc['LDDT-PLI', 'low']:.4f}} above zero, so the difference is thin. The BiSyRMSD and the LDDT-LP, the second a measure of the pocket and not of the ligand, show no difference. Against the plain Boltz-2 pose the published poses are better in the LDDT-PLI and in the BiSyRMSD, whereas 13.7 found that the submitted poses could not be told from that pose.

**Where the difference sits.** On {{=n_above_pool}} of the 20 ligands the published pose scores higher than the best pose of the pool of the project for that ligand ({{=pool_low}} to {{=pool_high}} Boltz-2 poses per ligand, {{=pool_total}} in all), so that no choice among those poses could have given it: for those ligands the pool, and not the choice, was the limit. The three ligands with the largest difference give {{=100 * top3_share:.0f}}% of the mean difference. The iron contact does not separate the two sets of poses: the published poses call {{=n_external_called}} ligands coordinated, {{=n_external_true}} of them coordinated in the released structure, the submitted poses call {{=n_submitted_coordinated}}, and the two sets of calls differ in {{=n_differ_calls}} ligand. The count of ligands above the pool depends on the score of the published pose, so it describes a limit of the pool and does not divide the difference into the generation of poses and the choice among them; that division would need the pool from which the published poses were chosen and the score of each of its poses, which this analysis does not compute.

**What the report of the entry says.** The report ({{=external_gate['source']['report']}}) describes a pool of poses from several co-folding methods that is filtered and clustered, and a choice of the cluster and of the pose of each ligand by its author, and it states no score for any step. The comparison above is a measurement by this project and not a result of that report.

*On what actually ran.* `python -m cypmode.post_deadline.structure --external`, the second command of the job `909_post_deadline_structure`, fetches the file, checks it, scores it with the function of question 0 and writes the tables that this section reads (`structure_external_*` in `data/post_deadline/outputs/`).
"""

CELLS = [
    code(FACTS),
    md(MD_13_5),
    code(GATE_TABLE),
    md(MD_13_5_B),
    md(MD_13_6),
    md(MD_13_7),
    md(MD_13_8),
    md(MD_13_9),
    code(EXTERNAL_TABLE),
    md(MD_13_9_B),
]
