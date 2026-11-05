# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XIII: closing. What the project shows and does not show (13.1), its limitations (13.2), its open questions (13.3) and the final results with the state of the submission files (13.4). Every number is read from the repository's files when the cell runs."""

from notebooks.notebook_builder import code, md

SHOWS = r'''
# The evidence behind the claims of this section, read from the files that hold it: the label rule's mismatches and the limitations from the method report, the nested AUC of the final stacks from the final report, the verdicts of the gates from their
# logs (the final build's own parser), the family of tests with its Holm adjustment (9.11), and the structure recipe with its record. Nothing is typed in.
import json
import re

from cypmode.challenge import final_tdi_build as final_build
from cypmode.challenge.multiplicity import collect as collect_family

log_dir = ROOT / "data/structure_track/outputs/cyp3a4/logs"
method_text = (ROOT / "METHOD_REPORT.md").read_text()
label_mismatch = dict(zip(("CYP3A4", "CYP2D6"), re.search(r"mismatches: CYP3A4 (\d+), CYP2D6 (\d+)", method_text).groups()))

final_text = (ROOT / "data/tdi_track/outputs/final/final_report.md").read_text()
def final_stack_row(isoform):
    block = re.search(rf"## {isoform}\n(.*?)(?=\n## |\Z)", final_text, re.S).group(1)
    stack = re.search(r"Final stack: \*\*(.+?)\*\* \((\d+) candidates\)", block)
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] == stack.group(1) and len(cells) >= 6:
            return {"isoform": isoform, "stack": stack.group(1), "candidates": int(stack.group(2)), "AUC at z0 = 1.0": float(cells[2]), "MCC at z0 = 1.0": float(cells[3]), "AUC at z0 = 1.96": float(cells[4]), "MCC at z0 = 1.96": float(cells[5])}
final_rows = pd.DataFrame([final_stack_row("CYP3A4"), final_stack_row("CYP2D6")])
show_table(final_rows, floatfmt=".4f")
auc_3a4, auc_2d6 = final_rows.loc[0, "AUC at z0 = 1.96"], final_rows.loc[1, "AUC at z0 = 1.96"]
auc_3a4_loose, auc_2d6_loose = final_rows.loc[0, "AUC at z0 = 1.0"], final_rows.loc[1, "AUC at z0 = 1.0"]

gate_verdicts = final_build.parse_gates(log_dir)
n_gate_pairs = len(gate_verdicts)
passed = sorted(f"{group} ({isoform})" for (group, isoform), v in gate_verdicts.items() if v.get("stacked"))
n_passed, passed_text = len(passed), ", ".join(passed) or "none"
family = collect_family(log_dir)
n_family, n_raw_below = len(family), int((family["p_both"] < 0.05).sum())
smallest_holm = float(family["holm_adjusted"].min())
raw_below_names = ", ".join(family.loc[family["p_both"] < 0.05, "log"]) or "none"
n_holm_survive = int((family["holm_adjusted"] < 0.05).sum())
holm_sentence = "after the Holm adjustment for the number of attempts none of them survives" if n_holm_survive == 0 else f"after the Holm adjustment for the number of attempts {n_holm_survive} of them survive"
neural_row = family[family["log"] == "aux_neural_CYP3A4.log"].iloc[0]
neural_p, neural_holm = float(neural_row["p_both"]), float(neural_row["holm_adjusted"])

recipe = json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/recipe.json").read_text())
structure_manifest = json.loads((ROOT / "data/structure_track/outputs/cyp3a4/final/manifest.json").read_text())
n_struct_gates, n_struct_met = len(recipe["gates"]), sum(recipe["gates"].values())
met_struct = ", ".join(g for g, v in recipe["gates"].items() if v)
n_ligands, n_valid_poses = len(structure_manifest["ligands"]), sum(bool(x["valid"]) for x in structure_manifest["ligands"])
n_relaxed, n_relaxable = structure_manifest["relaxation"]["adopted"], structure_manifest["relaxation"]["of"]
assert n_valid_poses == n_ligands and "neural (CYP3A4)" in passed  # the sentences below say so; if a build ever disagrees, the build fails and the text is rewritten
n_structure_failures = len(structure_manifest["posebusters_failures"])
'''

LIMITS = r'''
# The limitations of the method report (section 11), read from the report as the audit wrote it: an identifier, the area and the text with its numbers filled in from the audit's own values. The table is the report's, shown here so that the
# reader meets the limitations in the place where the claims end.
limit_pairs = re.findall(r"^- \*\*(L\d+)\*\* \((\w+)\): (.+)$", method_text, re.M)
limit_table = pd.DataFrame(limit_pairs, columns=["id", "area", "limitation"])
show_table(limit_table, floatfmt=".0f")
n_limits = len(limit_table)
limit_areas = ", ".join(sorted(set(limit_table["area"])))
'''

REPLICATE = r'''
# The exploratory replicate of the Atom-JEPA readout (9.10), if the queue has run it: the comparison of the readout added to the default stack on the 10-outer-fold plan, CYP3A4 only, declared before any 10-fold number of the readout existed. It was
# written to a log that no gate parser reads, so it could not change the model; here it is read next to the registered 5-outer-fold result.
replicate_log = log_dir / "exploratory_atomjepa_head_f10_CYP3A4.log"
registered_log = log_dir / "aux_atomjepa_head_CYP3A4.log"
pattern = r"z0=(\S+)\s+stacked[^:\n]*:\s+AUC\s+([+-]\d+\.\d+)\s+\(p=(\d+\.\d+), better (\d+)/(\d+)\)"
registered = re.findall(pattern, registered_log.read_text())
replicated = re.findall(pattern, replicate_log.read_text()) if replicate_log.exists() else []
readout_rows = [{"plan": "registered, 5 outer folds, 3 repeats", "z0": z, "change in AUC": float(d), "p": float(p), "folds better": f"{b} of {n}"} for z, d, p, b, n in registered]
readout_rows += [{"plan": "exploratory, 10 outer folds, 3 repeats", "z0": z, "change in AUC": float(d), "p": float(p), "folds better": f"{b} of {n}"} for z, d, p, b, n in replicated]
readout_table = pd.DataFrame(readout_rows)
show_table(readout_table, floatfmt=".4f")
replicate_done = bool(replicated)
if replicate_done:
    same_sign = all(float(d) > 0 for _, d, *_ in replicated)
    replicate_sentence = (f"The exploratory replicate has been read: the change in AUC is {', '.join(f'{float(d):+.4f} (p = {p})' for _, d, p, *_ in replicated)} at the two cut-offs, " + ("in the same direction as the registered result, which is what a real effect would do and also what a chance result sometimes does" if same_sign else "not in the same direction at both cut-offs, which is what chance would do") + ". It is reported and it changes nothing: the group did not pass its registered gate.")
else:
    replicate_sentence = "The exploratory replicate had not finished when this notebook was built."
'''

FINAL_FILES = r'''
# The final results and the state of the files, read from the manifests that the builds wrote: for the TDI track the two candidate files with their SHA-256 and the call rates, the stack and the share of calls of each isoform, and for the structure
# track the archive with its SHA-256, the checks it passes, the relaxation and the smallest relative distance. The files of the portal are copies of these, made by job 950 and compared with them by job 960.
tdi_manifest = json.loads((ROOT / "data/tdi_track/outputs/final/manifest.json").read_text())
file_rows = [{"track": "TDI", "file": f["file"], "SHA-256 (first 16 digits)": f["sha256"][:16], "rows": f["rows"], "CYP3A4 called positive": f["CYP3A4_positive_rate"], "CYP2D6 called positive": f["CYP2D6_positive_rate"]} for f in tdi_manifest["files"]]
file_table = pd.DataFrame(file_rows)
show_table(file_table, floatfmt=".3f")
isoform_rows = [{"isoform": iso, "final stack": v["stack"], "candidates": v["n_candidates"], "groups that joined": ", ".join(v["surviving_groups"]) or "none", "target prevalence": v["target_prevalence"], "called rate": v["called_rate"]} for iso, v in tdi_manifest["isoforms"].items()]
isoform_table = pd.DataFrame(isoform_rows)
show_table(isoform_table, floatfmt=".3f")
relaxation = structure_manifest["relaxation"]
structure_rows = pd.DataFrame({"quantity": ["archive", "SHA-256 (first 16 digits)", "ligands", "poses that pass all checks of the scorer", "poses relaxed under the rule", "smallest relative distance to the receptor, before and after the relaxation"],
                               "value": [structure_manifest["zip"].split("/")[-1], structure_manifest["sha256"][:16], n_ligands, f"{n_valid_poses} of {n_ligands}", f"{n_relaxed} of {n_relaxable}", f"{relaxation['smallest_margin_before']:.3f} to {relaxation['smallest_margin_after']:.3f}"]})
show_table(structure_rows, floatfmt=".3f")
hours_of_runs = float(recipe["estimated_hours"])
n_tdi_files = len(file_table)
'''

MD_13_INTRO = """
# Part XIII. Closing

Twelve parts have built a method, tested it, and shown how the tests were themselves tested. This part says what that adds up to. It is short on purpose: the evidence is in the parts that produced it, and the closing is sound only if it says no more than the evidence does. 13.1 separates what the project shows from what it does not. 13.2 lists the limitations the method report logs, with the ones the report does not. 13.3 names the questions that remain open and what would settle each. 13.4 gives the final results and the state of the submission files, read from the files themselves.
"""

MD_13_1 = """
## 13.1 What the project shows and what it does not

> **Field:** Overview · Statistics

*Where this section sits:* while the challenge ran the project could not see the labels of the compounds it is scored on, and it said so from the first page. Everything it claims in 13.1 to 13.4 is therefore a claim about a procedure or about a measurement on the public data, and this section sorts those claims by how well they are supported; 13.5 to 13.8 then use the labels that the organisers released after the deadline, under a plan that was written before they were read.

The next cell reads the evidence for what follows from the files that hold it.
"""

MD_13_1_B = """
**What can be said with confidence.** Five things, each with the place where it is shown.

*The label is a function of two measurements, exactly.* A compound is a time-dependent inhibitor when its potency after pre-incubation exceeds its direct potency (floored at 4) by more than a factor of two (2.3). The rule reproduces every label of both isoforms in the training file: the number of mismatches is {{=label_mismatch['CYP3A4']}} for CYP3A4 and {{=label_mismatch['CYP2D6']}} for CYP2D6. Nothing about the labels was guessed.

*A yardstick that resembles the test, with every choice inside the loop.* Part VI built an evaluation that reproduces the population the challenge scores (its similarity to the blinded set, its prevalence, only the confidently labelled compounds) and puts every selection step inside the cross-validation. Under it the final stacks reach a nested AUC of {{=auc_3a4:.4f}} for CYP3A4 and {{=auc_2d6:.4f}} for CYP2D6 at the stricter confidence cut-off, and {{=auc_3a4_loose:.4f}} and {{=auc_2d6_loose:.4f}} at the looser one. The design cannot tell a difference smaller than {{=floor_main}} in AUC from noise (6.7), so these figures are estimates with that resolution, not exact values.

*A rule for admitting ideas that was written before the ideas were tested.* {{=n_gate_pairs}} pairs of an idea and an isoform carry a registered gate whose log was read by the final build; {{=n_passed}} passed on the stack: {{=passed_text}}. The ledger of Part IX lists the rest with their numbers, and 9.12 explains, group by group, why most of them added nothing.

*A structure recipe whose every lever was decided by a gate.* Of {{=n_struct_gates}} gates, {{=n_struct_met}} are MET ({{=met_struct}}); the others, including every way of choosing among poses, are NOT MET. {{=n_valid_poses}} of the {{=n_ligands}} submitted poses pass every check of the scorer's own PoseBusters call, and the relaxation of 11.10 was applied to {{=n_relaxed}} of {{=n_relaxable}} under a rule written before the poses were relaxed.

*A procedure that can be checked.* Part XII: the numbers are reproducible, the audit regenerates them and compares, the documents are generated from the logs, the hold-back of results is enforced by a test.

**What it does not show, and does not claim.**

*It does not show how well the entries score.* The score that the portal gives an entry is a separate measurement on the hidden labels, and the limitations of 13.2 say that the offset between a cross-validated estimate and that score depends on the model. A cross-validated AUC on the training data is not the challenge score.

*It does not show whether the selection-aware idea works.* The cross-validation says that the selection-aware file gains when the scored CYP2D6 set holds many compounds that the screen would never have promoted and loses when it holds none, and nothing identifies which case the test set is in. The entry is the hits-only file, and the second file is kept to be scored on the labels if the organisers publish them (13.4).

*It does not show that the gains it kept are real.* The family of {{=n_family}} tests of Part IX has {{=n_raw_below}} with a raw p-value below 0.05 ({{=raw_below_names}}), and {{=holm_sentence}} (the smallest adjusted p-value is {{=smallest_holm:.2f}}). The neural pair passes its registered gate, with a raw p-value of {{=neural_p:.3f}} on CYP3A4 and a replicate on the 10-outer-fold plan, but its adjusted p-value is {{=neural_holm:.2f}}. The reading is that its gain is plausible and modest and has not been shown beyond what the number of attempts could produce.

*It does not show that a pose is right.* The poses are valid; nobody outside the organisers knows how close they are to the crystal. On the benchmark of ligands with deposited crystals the first pick of the model is far from the crystal on average, and the limitation on structure in 13.2 gives the figure.

*It does not claim novelty beyond a search.* The method report claims only that a component was not found in the sources it names on the date of the search; every component has published relatives.

*On what actually ran.* The cell reads the method report, the final report, the logs of the gates through `final_tdi_build.parse_gates`, the family of tests through `multiplicity.collect`, and the records of the structure recipe. It runs no experiment.
"""

MD_13_2 = """
## 13.2 Limitations

> **Field:** Statistics

*Where this section sits:* 13.1 sorted the claims. This section is the list of reasons to hold them loosely, in two parts: the limitations the method report logs, which the audit fills with its own numbers, and the ones the report does not.

The next cell shows section 11 of the method report as it stands.
"""

MD_13_2_B = """
The table has {{=n_limits}} entries, over the areas {{=limit_areas}}.

**Read by area.** The *evaluation* limitations say the same thing twice: the numbers are cross-validated estimates, they change with the population that is scored, and the baseline of the organisers scores differently on the board than in cross-validation, so the distance between the estimate and the score is not a constant. The *labels* limitation says that the rule that decides which compounds count as confidently labelled is not public, so the cut-off used here is a proxy, and every analysis was repeated at two cut-offs for that reason. The *uncertainty* and *calibration* limitations say what the committee and the calibration do not do: where the committee adds nothing beyond the margin it is reported as a negative result, and the CYP2D6 probabilities are close to a constant, so that the model ranks and transfers a call rate but does not give usable probabilities. The *structure* limitation says that the benchmark is small, noisy and made of ligands from the PDB, which the co-folding models may have seen, so that absolute accuracy is optimistic. The *reproducibility* and *repository* limitations are about the state of the work, and the audit refreshes them each time it runs. The *population* limitation is the central uncertainty of the CYP2D6 entry: the share of compounds in the scored set that the original screen would never have promoted is not identified by anything public, and only a submission can settle it.

**What the report does not list.**

*The multiplicity of the search.* The registered gates were passed by few ideas, and none of the passes survives the correction for the number of attempts (13.1). The gates control the risk of adding an idea that does not help, one idea at a time; they do not control the risk that among many ideas one passed by chance. The pre-registration is what limits the damage, because it fixed the rule before the data were seen, and the Holm column is what measures what is left.

*The environments of the structure engines are not pinned.* The pin files of Appendix A cover the main environment, the neural environment and the two conda environments; Boltz-2 and OpenFold3 run in virtual environments that are not among them. The versions that ran are in the manifests of the structure track, but a reader who wants to reinstall them has to follow the instructions of the engines.

*The cut-off proxy was checked once the labels were released.* The confidence cut-off is a proxy for a rule that the organisers did not publish. The released labels show that the organisers score nearly the whole test set (13.6), so the proxy scores fewer compounds than the official score does and the cross-validated estimates are those of an easier problem than the one that is scored.

*The held-back results were published with the release.* The files that name results for the blinded compounds came out with the release after the challenge closed, so a reader can now verify the files as well as the code, the reports and the hashes.
"""

MD_13_3 = """
## 13.3 Open questions

> **Field:** Overview

*Where this section sits:* the limitations say what the project does not know. This section names the questions whose answers would change what it would do next, with what would settle each.

**Where do the entries land?** The step from the cross-validated estimates to the leaderboard is the largest unknown, and it is not the project's to settle: it is settled by the scores, which are the organisers'. The relationship between the baseline's cross-validated score and its board score (limitation on evaluation) is the only calibration available, and it is a single point.

**How many compounds in the scored CYP2D6 set were never promoted by the screen?** The selection-aware model gains much when they are many and loses a little when they are few, and the public metrics of the baseline are consistent with hundreds of different shares. Two entries that differ only in the CYP2D6 model, submitted on the same hidden compounds, identify the regime; the rule that decides between them was written before either was submitted (the difference in MCC must exceed twice the spread of the board). Which regime holds is a fact about the challenge's design that this project could measure only in this way.

**Is the near miss real?** The Atom-JEPA readout added a small gain to the CYP3A4 stack, in most folds, that did not reach the registered gate and lies below what the design can detect (9.10). An exploratory replicate on a different split of the folds was run with the rule that it could not change the model.
"""

MD_13_3_B = """
{{=replicate_sentence}}

**Would a fine-tuned CheMeleon help?** The speed test of 9.10 could not run on the graphics unit, and the estimate on the processor was above the time that was available. The question is therefore open for cost, not for a verdict; the cheaper use of the same encoder, frozen embeddings read by TabICL and by TabPFN, was tested and did not pass.

**Does the iron-nitrogen restraint help on ligands that were not used to choose it?** On the benchmark it did (11.6). The two checks on new ligands, which would show whether the gain holds beyond the benchmark, did not reach their gates, so the direction is neither confirmed nor refuted. The organisers announced re-refined crystal structures of CYP3A4; they would allow the twenty submitted poses to be compared with something other than the scorer's checks, and the first question to ask of them is whether the poses that the restraint produced are closer to them.

**Is there a ceiling?** Most of the ideas that did not join the stack were largely redundant with it (9.12): their predictions correlated with the stack's at about 0.8. Whether the remaining error is the noise of the assay, which no model could remove, or information that no tested representation carries, is not known. The test would be a representation whose predictions are not redundant and that still adds nothing, or one that does add.

**What would a better yardstick change?** The design floor of the evaluation is about a hundredth of an AUC. A larger or differently built set of confidently labelled compounds would lower it; the effect would be to let smaller gains be seen, not to make the existing ones larger.

*On what actually ran.* The cell reads the registered log of the readout and, if the queue has produced it, the exploratory log.
"""

DECISION_FACTS = r'''
# The choice of the final CYP2D6 file of 13.4: the result of job 016 (cypmode/challenge/selection_decision.py), read from its record.
import json

decision = json.loads((ROOT / "data/tdi_track/outputs/comparisons/selection_decision.json").read_text())
decision_pac, decision_band, decision_regret = decision["evidence"]["pac"], decision["evidence"]["band"], decision["regret"]
decision_rate = decision["evidence"]["rate"]
decision_sentence = (
    f"The detector is usable, so the entry is decided by the two uploads: B if the CYP2D6 MCC of B exceeds that of A by at least {decision['threshold']:+.2f}, A otherwise."
    if decision["usable"]
    else f"The detector is not usable, so no upload of B is made to decide, and the entry is the file with the smaller largest regret over the whole grid, file {decision['fallback']} (largest regret {decision_regret['hits_only']:.3f} for the hits-only file against {decision_regret['selection_aware']:.3f} for the selection-aware file)."
)
'''

MD_13_4_C = """
**Which CYP2D6 file is the final entry.** The two files differ in 40 of the 750 CYP2D6 calls and nowhere else. Whether the selection-aware file B is better than the hits-only file A depends on a number that no label identifies: the weight $w$ of the compounds that the screen never promoted in the scored set, where 1 is their share in the all-isoform panel and 0 means that the scored set holds only compounds like those of the training file. Part IX shows that B wins when $w$ is large and loses when it is near zero. The label-shift estimate gives $w = {{=decision_pac['w']:.2f}}$ (95 percent interval {{=decision_pac['low']:.2f}} to {{=decision_pac['high']:.2f}}), under an assumption that analog expansion may violate, and the fit of the board scores of the official baseline does not identify it: the plausible region runs from $w = {{=decision_band['w_low']:.2f}}$ to ${{=decision_band['w_high']:.1f}}$, because $w$ and the rate of time-dependent inhibitors among the hits trade off.

The choice is fixed by a procedure that was written before any file was uploaded (`cypmode/challenge/selection_decision.py`, job `016_selection_decision`). Each file calls the top {{=decision_rate:.3f}} of the scored population, as the final build does, and the weighted MCC is compared under $w$ from 0 to 1. (The first version of the procedure scored the calls at the rate of the experiments of Part IX, about a third of the pool, which is not what is submitted; it was corrected before its output was used, and the log prints both values.) At the rate of the final build B gains {{=decision['delta_high']:.3f}} of MCC on average where it wins ($w \\ge 0.25$) and loses {{=decision['delta_low']:.3f}} at $w = 0$. The live board scores half of the test set, and the difference of the two uploads on it has a spread of about 0.09 to 0.10 from the choice of compounds alone; the threshold that minimises the larger of the two risks is {{=decision['threshold']:+.2f}}, with an error of {{=decision['error_low']:.2f}} at $w = 0$ and {{=decision['error_high']:.2f}} on average for $w \\ge 0.25$, and the procedure requires both to be at most 0.25. {{=decision_sentence}}

The rank blend of the two files has no file and its gate in Part IX is not met; its largest regret against the best of the three is {{=decision_regret['blend_of_three']:.3f}}, against {{=decision_regret['hits_only_of_three']:.3f}} for the hits-only file. The choice rests on cross-validation with a model of the scored population: it does not say which file scores higher on the hidden labels, and a difference of {{=decision['delta_high']:.2f}} of MCC where B wins is smaller than the interval of the bootstrap over the outer folds in most cells of the table in the log.

*On what actually ran.* The cell reads `data/tdi_track/outputs/comparisons/selection_decision.json`, which job 016 wrote from the cached predictions of the nested cross-validation; the notebook does not run the procedure.
"""

MD_13_4_D = """
**What the second file is for.** File B is not uploaded. It is kept, with its SHA-256 in the manifest and in the reports, as the alternative that was declared before any result existed (the selection-aware model of 9.1). The text of the challenge says that no ground-truth test data are unblinded until the challenge concludes. If the labels of the test set are published, both files are scored on them with the metric of the challenge: the MCC of file A, of file B and of the rank blend of the two on CYP2D6, and the share of never-promoted compounds in the scored set (the weight $w$ above), which is then measured and no longer estimated. The comparison is written here before it can be made. It tests the hypothesis of 9.1 on data that no choice of this project touched, whichever way it falls: if file B scores higher, the selection-aware model improved on the entry that the rule left out, which is a result of the exploration that the final entry did not collect; if it scores lower, it is a recorded negative result, with the regime in which it fails.

This is how exploration and exploitation are kept apart. The organisers reward exploration over mere exploitation in their award for innovation. This project explores in the open (every idea has its rule and its verdict, the ones that failed included) and exploits only what a rule written beforehand supports, which is file A. The idea that the rule left out is not dropped: it is kept as a file with a hash, to be tested. The result of that comparison is added to the method report and to the release.
"""

MD_13_4 = """
## 13.4 Final results and the state of the submission files

> **Field:** Overview

*Where this section sits:* the end of the notebook. The numbers below are not written here; they are read, when the notebook is built, from the manifests that the final builds wrote, so that they are the numbers of the files as they are.

**TDI track.** The final build chooses the stack of each isoform by the pre-registered rule, forms the call rate from the prevalence of the scored population, and writes two candidate files that differ only in the CYP2D6 model: A, the hits-only model, and B, the selection-aware one. The tables give the files with their SHA-256 and the stack, the groups that joined and the share of calls of each isoform.
"""

MD_13_4_B = """
The two files have {{=n_tdi_files}} entries in the manifest, and the figures of the second table are the stacks of 13.1. The hash of a file is what the method report quotes and what the pre-submission gate compares: a file of the portal is the right file if and only if its SHA-256 equals the one in the table.

**Structure track.** The candidate archive holds twenty complexes. The table gives its hash, how many poses pass the scorer's checks, how many were relaxed under the rule of 11.10, and the smallest relative distance to the receptor before and after the relaxation. The recipe that produced the poses was estimated at {{=hours_of_runs:.1f}} hours of engine time before it was run.

**The state of the files.** The files in `submissions/` are copies of the ones above, made by job 950 of the queue after the notebook is built, and job 960 compares them with the manifests; the pre-submission gate reports READY only if the queue is empty, the audit has no failed check and is newer than the outputs, the submission files equal the final outputs and pass the scorer's own PoseBusters call, and the documents, the notebook and the film are newer than the last output. No file is submitted by any program of the repository: the portal is used by hand, and the entry that is final is chosen by the rule declared in advance.

What has been established and what has not is in 13.1. The rest of the story is in the parts before it; the way to run it again is in Appendix A.

*On what actually ran.* The cell reads `data/tdi_track/outputs/final/manifest.json`, `data/structure_track/outputs/cyp3a4/final/manifest.json` and the recipe record. It checks nothing about the portal: that is the work of jobs 950 and 960.
"""

CELLS = [
    md(MD_13_INTRO),
    md(MD_13_1),
    code(SHOWS),
    md(MD_13_1_B),
    md(MD_13_2),
    code(LIMITS),
    md(MD_13_2_B),
    md(MD_13_3),
    code(REPLICATE),
    md(MD_13_3_B),
    md(MD_13_4),
    code(FINAL_FILES),
    md(MD_13_4_B),
    code(DECISION_FACTS, hide_input=True),
    md(MD_13_4_C),
    md(MD_13_4_D),
]
