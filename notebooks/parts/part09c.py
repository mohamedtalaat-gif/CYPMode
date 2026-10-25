# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX (third part): the neural pair (9.9), the other additions of the last days (9.10) and the ledger of Part IX (9.11)."""

from notebooks.notebook_builder import code, md

NEURAL_MEMBERS = '''
# The members of the neural pair as the nested design holds them. The multitask network (cypmode/challenge/nested_runners_torch.py, fit_predict_multitask) reads the logarithm of the Morgan counts and the quantile-normalised
# descriptors, passes them through a shared trunk of two layers (1,024 and 512 units, batch normalisation, ReLU, dropout 0.35), and has one regression head per assay arm (the direct and the TDI pIC50 of the four enzymes, a missing
# value masked out of the loss) and one classification head for the label; it is trained for 60 epochs with AdamW and a one-cycle learning rate, three seeds averaged, and gives two scores: the label probability and the predicted
# shift t_hat - max(d_hat, 4) of the target enzyme. Plain Chemprop (cypmode/challenge/nested_runners_chemprop.py) is the message-passing network of 4.8 on the molecular graph, multitask over the same eight arms, 30 epochs, with
# 10 percent of its training rows held out for early stopping, scored by the predicted shift. The cell reads what each gave alone (the diagnostic lines of the logs) and how closely each agrees in ranking with the best single
# tree model of the default stack, lgbm_A+fps_deep, on the outer folds of the main design: a member can add to a stack only if it does not simply repeat the members that are already there.
from scipy import stats

member_stems = {"mlp_multitask_label": "aux_mlp", "mlp_multitask_shift": "aux_mlp", "chemprop_mt_shift": "aux_chemprop"}
reference_member = "lgbm_A+fps_deep"
def mean_fold_spearman(cache, plan, name_a, name_b):
    values = []
    for repeat, outer in plan.outer_folds():
        task = plan.task(repeat, outer, OUTER_REFIT)
        values.append(stats.spearmanr(cache.get(name_a, task), cache.get(name_b, task))[0])
    return float(np.mean(values))
member_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    plan_i, cache_i = setups8[iso][0], setups8[iso][1]
    for name, stem in member_stems.items():
        diag = read_gate_log(stem, iso)["diagnostic"]
        auc = diag[diag["candidate"] == name].set_index("z0")["outer AUC alone"]
        other = "chemprop_mt_shift" if name != "chemprop_mt_shift" else "mlp_multitask_shift"
        member_rows.append({"enzyme": iso, "member": name, "outer AUC alone at z0 1.0": float(auc.loc[1.0]), "outer AUC alone at z0 1.96": float(auc.loc[1.96]),
                            "rank correlation with lgbm_A+fps_deep": mean_fold_spearman(cache_i, plan_i, name, reference_member), "rank correlation with " + other: mean_fold_spearman(cache_i, plan_i, name, other)})
member_table = pd.DataFrame(member_rows)
show_table(member_table, floatfmt=".3f")
'''

NEURAL_GATES = '''
# The gate of the main design for each member alone added to the default stack, and for the pair, both enzymes, read from the logs of the experiments (aux_mlp, aux_chemprop, aux_neural). Every figure is the paired change in
# the outer-fold AUC of "default stack plus the addition" against "default stack", with the corrected resampled t of 6.5 (J = 15 folds); AP is the average precision of the same comparison.
AP_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+stacked base\\+added vs base: AUC [+-][\\d.]+ \\(p=[\\d.]+, better \\d+/\\d+\\) AP ([+-][\\d.]+) \\(p=([\\d.]+)")
gate_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    for label, stem in (("multitask network", "aux_mlp"), ("plain Chemprop", "aux_chemprop"), ("the pair", "aux_neural")):
        result = read_gate_log(stem, iso)
        stacked = result["stacked"].set_index("z0")
        ap = {float(z): (float(g), float(p)) for z, g, p in AP_LINE.findall((LOGS9 / f"{stem}_{iso}.log").read_text(errors="ignore"))}
        gate_rows.append({"enzyme": iso, "addition": label, "AUC gain at z0 1.0": float(stacked.loc[1.0, "gain"]), "p at z0 1.0": float(stacked.loc[1.0, "p"]), "folds better at z0 1.0": stacked.loc[1.0, "better"],
                          "AUC gain at z0 1.96": float(stacked.loc[1.96, "gain"]), "p at z0 1.96": float(stacked.loc[1.96, "p"]), "AP gain at z0 1.0": ap[1.0][0], "gate": "MET" if result["verdicts"].get("stacked") else "NOT MET"})
gate_table = pd.DataFrame(gate_rows)
show_table(gate_table, floatfmt=".4f")
def gate_row(iso, label):
    return gate_table[(gate_table["enzyme"] == iso) & (gate_table["addition"] == label)].iloc[0]
mlp_3a4, chemprop_3a4, pair_3a4, pair_2d6 = gate_row("CYP3A4", "multitask network"), gate_row("CYP3A4", "plain Chemprop"), gate_row("CYP3A4", "the pair"), gate_row("CYP2D6", "the pair")
'''

NEURAL_REPLICATE = '''
# The replicate (cypmode/challenge/replicate_verdict.py): the same comparison on the 10-outer-fold plan, J = 30 folds, n_test / n_train = 1/9, so that the variance factor of the corrected t of 6.5 falls from 1/15 + 1/4 to
# 1/30 + 1/9. The rule, declared on 2026-10-16 after the fifteen-fold result and before any ten-fold one: corrected p < 0.05 and a positive mean AUC difference at both cut-offs in the ten-fold design, and a positive mean
# difference at both cut-offs in the five-fold design. Scope: CYP3A4 only (the pair showed nothing on CYP2D6, so no replicate was declared for it). The second table puts the discovery p-value of the pair among those of
# every stacked comparison logged for CYP3A4 (the replicate logs left out, since they repeat comparisons already in the family) and applies Holm's adjustment (6.5) to the whole family; the replicate is a single
# pre-declared test and is not part of that adjustment.
from cypmode.challenge.replicate_verdict import read_stacked, verdict
from statsmodels.stats.multitest import multipletests

five_fold = read_stacked(LOGS9 / "aux_neural_CYP3A4.log")
ten_fold = read_stacked(LOGS9 / "aux_neural_f10_CYP3A4.log")
ten_result = read_gate_log("aux_neural_f10", "CYP3A4")
replicate_met = verdict(five_fold, ten_fold)
AP10 = re.compile(r"z0=(\\d\\.\\d+)\\s+stacked base\\+added vs base: AUC [+-][\\d.]+ \\(p=[\\d.]+, better \\d+/\\d+\\) AP ([+-][\\d.]+) \\(p=([\\d.]+)")
ap_ten = {float(z): (float(g), float(p)) for z, g, p in AP10.findall((LOGS9 / "aux_neural_f10_CYP3A4.log").read_text(errors="ignore"))}
factor_five, factor_ten = 1 / 15 + 1 / 4, 1 / 30 + 1 / 9
replicate_rows = [{"design": "5 outer folds x 3 repeats (discovery)", "folds J": 15, "variance factor": factor_five, "AUC gain at z0 1.0": five_fold[1.0][0], "p at z0 1.0": five_fold[1.0][1], "AUC gain at z0 1.96": five_fold[1.96][0], "p at z0 1.96": five_fold[1.96][1]},
                   {"design": "10 outer folds x 3 repeats (replicate)", "folds J": 30, "variance factor": factor_ten, "AUC gain at z0 1.0": ten_fold[1.0][0], "p at z0 1.0": ten_fold[1.0][1], "AUC gain at z0 1.96": ten_fold[1.96][0], "p at z0 1.96": ten_fold[1.96][1]}]
replicate_table = pd.DataFrame(replicate_rows)
show_table(replicate_table, floatfmt=".4f")
better_ten = ten_result["stacked"].set_index("z0")["better"]

STACK_LINE = re.compile(r"z0=(1\\.0|1\\.96)\\s+stacked base\\+added vs base: AUC ([+-]\\d\\.\\d+) \\(p=([\\d.]+)")
family_names, family_p = [], []
for path in sorted(LOGS9.glob("*CYP3A4.log")):
    if "_f10" in path.stem:
        continue  # a replicate repeats a comparison that is already in the family
    found = {z0: float(p) for z0, gain, p in STACK_LINE.findall(path.read_text(errors="ignore"))}
    if "1.0" in found and "1.96" in found:
        family_names.append(path.stem.removesuffix("_CYP3A4")), family_p.append(max(found["1.0"], found["1.96"]))
holm_p = multipletests(family_p, method="holm")[1]
pair_index = family_names.index("aux_neural")
n_family9 = len(family_p)
pair_raw_p, pair_holm_p = family_p[pair_index], float(holm_p[pair_index])
n_raw_below = int(sum(p < 0.05 for p in family_p))
print(f"replicate rule: {'MET' if replicate_met else 'NOT MET'}; the pair among {n_family9} logged stacked comparisons: raw p {pair_raw_p:.3f}, Holm-adjusted {pair_holm_p:.2f}; {n_raw_below} of the {n_family9} have a raw p below 0.05")
'''

NEURAL_BUILD = '''
# What the final build of the TDI stack (cypmode/challenge/final_tdi_build.py) says about the pair: the groups that joined each enzyme's stack are written into its manifest. A group listed in CONFIRMED_GROUPS (the pair is the only
# one) joins only if its replicate verdict is MET as well, and a missing replicate line is a fail.
import json

from cypmode.challenge.final_tdi_build import CONFIRMED_GROUPS

manifest9 = json.loads((ROOT / "data/tdi_track/outputs/final/manifest.json").read_text())
joined_3a4 = manifest9["isoforms"]["CYP3A4"]["surviving_groups"]
n_members_3a4 = manifest9["isoforms"]["CYP3A4"]["n_candidates"]
if "neural" in joined_3a4:
    neural_stack_sentence = f"The final build of the TDI stack records the pair among the groups that joined the CYP3A4 stack, which has {n_members_3a4} members."
else:
    neural_stack_sentence = f"The final build of the TDI stack that this page was made from does not list the pair among the groups that joined the CYP3A4 stack (it has {n_members_3a4} members); the build reads every verdict again at its last run."
print(neural_stack_sentence, "| groups that need a replicate:", ", ".join(CONFIRMED_GROUPS))
'''

ADDITIONS = '''
# The remaining additions of Part IX, read the way 9.9 read the neural pair: one log per addition and enzyme, whose lines give the paired change of the outer-fold AUC when the addition joins the default stack (corrected
# resampled t, J = 15 folds, 6.5) at the two confidence cut-offs, and a last line with the verdict of the pre-declared gate. The two selectors (an alternative to the logistic blend, not a new member) report the change of the
# AUC of the whole stack when the blend is replaced. A log that does not exist yet is shown as "not run yet": the table is built from whatever the queue has finished when the page is built.
SELECTOR_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+outer AUC: logistic blend [\\d.]+, [\\w-]+ blend [\\d.]+ \\(([+-][\\d.]+), p=([\\d.]+), better (\\d+)/(\\d+)\\)")
SCREEN_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+stacked with the screen vs without: AUC ([+-][\\d.]+) \\(p=([\\d.]+), better (\\d+)/(\\d+)\\)")
VERDICT_LINE = re.compile(r"^(CYP\\w+) GATE \\((?:stacked|selector|population screen stacked)[^)]*\\): (MET|NOT MET)", re.M)

ADDITION_LIST = [
    ("pretrained tables and representations", "TabICL on the descriptors and counts", "aux_tabicl", "added"),
    ("pretrained tables and representations", "TabPFN on the same inputs", "aux_tabpfn", "added"),
    ("pretrained tables and representations", "Monroe embeddings as a block", "monroe_embeddings", "added"),
    ("pretrained tables and representations", "TabICL on the Monroe embeddings", "aux_tabicl_monroe", "added"),
    ("pretrained tables and representations", "TabICL on the frozen CheMeleon embeddings", "aux_tabicl_chemeleon", "added"),
    ("pretrained tables and representations", "TabPFN on the frozen CheMeleon embeddings", "aux_tabpfn_chemeleon", "added"),
    ("pretrained tables and representations", "Atom-JEPA embeddings as a block", "aux_atomjepa", "added"),
    ("pretrained tables and representations", "Atom-JEPA attention readout on the atoms", "aux_atomjepa_head", "added"),
    ("pretrained tables and representations", "CheMeleon, fine-tuned (staged)", "aux_chemeleon", "added"),
    ("pretrained tables and representations", "OpenADMET pretraining weights, fine-tuned (staged)", "aux_chemprop_foundation", "added"),
    ("measurements from outside the challenge", "NCATS qHTS, continuous responses", "aux_ext_cont", "added"),
    ("measurements from outside the challenge", "Tox21 CYP screen", "aux_ext_tox", "added"),
    ("measurements from outside the challenge", "OpenADMET logD and protein-binding models", "aux_ext_adme", "added"),
    ("measurements from outside the challenge", "PXR induction readings", "aux_pxr", "added"),
    ("the organisers' extra readings", "single-concentration screen, as a predicted block", "aux_screen", "added"),
    ("the organisers' extra readings", "maximal effect (Emax), as a predicted block", "aux_emax", "added"),
    ("the organisers' extra readings", "screen inside the selection-aware CYP2D6 model", "population_screen", "screen"),
    ("the organisers' extra readings", "multitask network with the screen and Emax as extra targets", "aux_mlp_aux", "added"),
    ("the organisers' extra readings", "the 1,249 compounds without a direct curve", "aux_nodirect", "added"),
    ("the organisers' extra readings", "the label in two parts (gate and shift)", "aux_gate_shift", "added"),
    ("chemistry", "basic-centre block", "aux_basic_centre", "added"),
    ("the blend itself", "non-negative blend instead of the logistic blend", "nonneg_stack", "selector"),
    ("the blend itself", "softmax blend instead of the logistic blend", "softmax_selector", "selector"),
]

def read_addition(stem, iso, kind):
    path = LOGS9 / f"{stem}_{iso}.log"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    pattern = {"added": STACKED_LINE, "selector": SELECTOR_LINE, "screen": SCREEN_LINE}[kind]
    found = {float(m[0]): (float(m[1]), float(m[2])) for m in pattern.findall(text)}
    verdicts = {iso_found: verdict == "MET" for iso_found, verdict in VERDICT_LINE.findall(text)}
    if len(found) < 2:
        return None
    return {"gain1": found[1.0][0], "p1": found[1.0][1], "gain2": found[1.96][0], "p2": found[1.96][1], "gate": "MET" if verdicts.get(iso) else "NOT MET"}

addition_rows = []
for family, label, stem, kind in ADDITION_LIST:
    for iso in ("CYP3A4", "CYP2D6"):
        result = read_addition(stem, iso, kind)
        row = {"family": family, "addition": label, "enzyme": iso}
        if result is None:
            row.update({"AUC change at z0 1.0": np.nan, "p at z0 1.0": np.nan, "AUC change at z0 1.96": np.nan, "p at z0 1.96": np.nan, "gate": "not run yet"})
        else:
            row.update({"AUC change at z0 1.0": result["gain1"], "p at z0 1.0": result["p1"], "AUC change at z0 1.96": result["gain2"], "p at z0 1.96": result["p2"], "gate": result["gate"]})
        addition_rows.append(row)
addition_table = pd.DataFrame(addition_rows)
# the screen model and the no-direct experiment exist for one enzyme only
addition_table = addition_table[~((addition_table["addition"].str.contains("selection-aware")) & (addition_table["enzyme"] == "CYP3A4"))]
addition_table = addition_table[~((addition_table["addition"].str.contains("without a direct curve")) & (addition_table["enzyme"] == "CYP2D6"))]
show_table(addition_table.drop(columns="family"), floatfmt=".4f")
n_tests_9 = int((addition_table["gate"] != "not run yet").sum())
n_met_9 = int((addition_table["gate"] == "MET").sum())
n_pending_9 = int((addition_table["gate"] == "not run yet").sum())
best_gain_9 = addition_table.loc[addition_table["AUC change at z0 1.0"].idxmax()]
def line_of(label, iso):
    r = addition_table[(addition_table["addition"].str.startswith(label)) & (addition_table["enzyme"] == iso)]
    if r.empty or r.iloc[0]["gate"] == "not run yet":
        return "not run when this page was built"
    r = r.iloc[0]
    return f"{r['AUC change at z0 1.0']:+.4f} (p = {r['p at z0 1.0']:.2f}) and {r['AUC change at z0 1.96']:+.4f} (p = {r['p at z0 1.96']:.2f}), gate {r['gate']}"
'''

CHEMELEON_STATE = '''
# The state of the staged CheMeleon design at the moment the page is built (cypmode/challenge/chemeleon_stage.py and chemeleon_speed.py). Stage 1 is the 25 cross-validation tasks of repeat 0 of an enzyme; the cell counts from the
# nested plan and its cache how many are done, reads the decision of the speed test (job 202_chemeleon_speed) if it exists, and what the stage rule says about the stage-1 comparison if that exists.
from cypmode.challenge import chemeleon_speed
from cypmode.challenge.chemeleon_stage import MIN_FOLDS as chemeleon_min_folds
from cypmode.challenge.chemeleon_stage import MIN_GAIN as chemeleon_min_gain
from cypmode.challenge.chemeleon_stage import decide as chemeleon_decide
from cypmode.challenge.nested import NestedPlan, PredictionCache

chemeleon_progress = {}
for iso in ("CYP3A4", "CYP2D6"):
    nested_root = ROOT / f"data/tdi_track/outputs/nested/{iso}"
    nested_plan = NestedPlan.load(nested_root / "plan.npz")
    nested_cache = PredictionCache(nested_root / "cache", nested_plan)
    stage_one_tasks = [task for task in nested_plan.tasks() if task.repeat == 0]
    chemeleon_progress[iso] = (sum(nested_cache.has("chemeleon_mt_shift", task) for task in stage_one_tasks), len(stage_one_tasks), len(nested_plan.tasks()))
n_stage_one, n_full_design = chemeleon_progress["CYP3A4"][1], chemeleon_progress["CYP3A4"][2]
cpu_minutes_per_task = chemeleon_speed.processor_seconds_per_task() / 60
stage_one_cpu_hours = n_stage_one * cpu_minutes_per_task / 60
speed_decision_file = ROOT / "data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt"
speed_json = ROOT / "data/tdi_track/outputs/comparisons/chemeleon_speed_CYP3A4.json"
done_3a4 = chemeleon_progress["CYP3A4"][0]
if not speed_decision_file.exists():
    chemeleon_sentence = f"The speed test had not been run when this page was built, and stage 1 for CYP3A4 had {done_3a4} of its {n_stage_one} tasks."
else:
    decision = speed_decision_file.read_text().strip()
    measured = json.loads(speed_json.read_text()) if speed_json.exists() else {}
    gpu_text = f"; the GPU took {measured['gpu_seconds'] / 60:.0f} minutes for the task that the processor had needed {measured['cpu_seconds_per_task'] / 60:.0f} minutes for (rank correlation {measured.get('spearman', float('nan')):.2f}, AUC difference {abs(measured.get('auc_gpu', 0) - measured.get('auc_cpu', 0)):.3f})" if measured.get("gpu_seconds") else ""
    stage_text = ""
    stage_file = ROOT / "data/structure_track/outputs/cyp3a4/logs/stage1_chemeleon_CYP3A4.txt"
    if decision in ("mps", "cpu") and stage_file.exists() and "stacked base+added" in stage_file.read_text():
        stage_text = f" Stage 1 has finished and the rule says {'CONTINUE' if chemeleon_decide(stage_file) else 'STOP'}."
    elif decision in ("mps", "cpu"):
        stage_text = f" Stage 1 is under way, with {done_3a4} of {n_stage_one} tasks done when this page was built."
    outcome = "the fine-tuned CheMeleon was not run, for cost (the ledger says 'not run: cost', which is not a verdict)" if decision == "not run: cost" else f"stage 1 for CYP3A4 runs on the {'Apple GPU' if decision == 'mps' else 'processor'}"
    chemeleon_sentence = f"The speed test found that the {n_stage_one - done_3a4 if decision == 'not run: cost' else measured.get('tasks_left', n_stage_one - done_3a4)} missing tasks would need {measured.get('hours_left_on_cpu', float('nan')):.1f} hours on the processor{gpu_text}, so {outcome}.{stage_text}"
print(chemeleon_sentence)

# The two OpenADMET pretraining weights follow the same two stages for CYP3A4 (queue jobs 255 to 258); their stage-1 screen is a text file as well.
cyp3a4_root = ROOT / "data/tdi_track/outputs/nested/CYP3A4"
cyp3a4_plan = NestedPlan.load(cyp3a4_root / "plan.npz")
cyp3a4_cache = PredictionCache(cyp3a4_root / "cache", cyp3a4_plan)
cyp3a4_stage_one = [task for task in cyp3a4_plan.tasks() if task.repeat == 0]
foundation_done = {name: sum(cyp3a4_cache.has(name, task) for task in cyp3a4_stage_one) for name in ("chemeleon_minimol_mt_shift", "chemeleon_surrogate_mt_shift")}
foundation_file = ROOT / "data/structure_track/outputs/cyp3a4/logs/stage1_chemprop_foundation_CYP3A4.txt"
foundation_pattern = re.compile(r"z0=1\\.0\\s+stacked base\\+added vs base: AUC ([+-]\\d\\.\\d+) \\(p=([\\d.]+), better (\\d+)/(\\d+)\\)")
foundation_text = foundation_file.read_text() if foundation_file.exists() else ""
foundation_found = foundation_pattern.search(foundation_text)
if foundation_found:
    foundation_sentence = (f"For the two OpenADMET weights together, stage 1 has finished: added to the default stack they change the outer AUC of CYP3A4 by {float(foundation_found.group(1)):+.4f} at the first cut-off (corrected p {foundation_found.group(2)}, better in {foundation_found.group(3)} of {foundation_found.group(4)} folds), "
                           f"and the rule says {'CONTINUE' if chemeleon_decide(foundation_file) else 'STOP'}.")
elif "not run" in foundation_text:
    foundation_sentence = "The two OpenADMET weights were not run: " + foundation_text.strip().splitlines()[0] + "."
else:
    foundation_sentence = (f"Stage 1 of the two OpenADMET weights is under way: {foundation_done['chemeleon_minimol_mt_shift']} and {foundation_done['chemeleon_surrogate_mt_shift']} of its {len(cyp3a4_stage_one)} tasks per starting point were done when this page was built.")
print(foundation_sentence)

# The exploratory second look at the Atom-JEPA readout on the 10-outer-fold plan (queue jobs 252 and 253): written to a log that no gate parser reads, so it cannot change a model.
atomjepa_log = ROOT / "data/structure_track/outputs/cyp3a4/logs/exploratory_atomjepa_head_f10_CYP3A4.log"
second_look = re.findall(r"z0=(\\S+)\\s+stacked base\\+added vs base: AUC ([+-]\\d\\.\\d+) \\(p=([\\d.]+), better (\\d+)/(\\d+)\\)", atomjepa_log.read_text()) if atomjepa_log.exists() else []
if second_look:
    atomjepa_sentence = "; ".join(f"{float(d):+.4f} AUC at z0 = {z} (p {p}, better in {b} of {n} folds)" for z, d, p, b, n in second_look)
else:
    atomjepa_sentence = "the second look had not finished when this page was built"
print(atomjepa_sentence)
'''

EXTRA_FACTS = '''
# Facts that the text of this section quotes, read from the logs of the queue and not typed: the audit of the organisers' extra files (cypmode/challenge/official_extra.py, queue job 190_official_extra_audit), the extraction of the
# outside measurements (external_continuous.py) and of the basic-centre block (basic_centre.py), the negative control (permutation_control.py) and the exploration-exploitation curve of the softmax blend.
def grab(text, pattern):
    found = re.search(pattern, text)
    return int(found.group(1).replace(",", "")) if found else None

audit_text = (LOGS9 / "official_extra_audit.log").read_text(errors="ignore")
n_screened = grab(audit_text, r"same compounds\\s+\\[(\\d+) compounds per enzyme")
n_emax_rows = grab(audit_text, r"Emax file holds no blinded test compound\\s+\\[(\\d+) compounds")
n_direct_rows = grab(audit_text, r"direct-inhibition file holds no blinded test compound\\s+\\[(\\d+) compounds")
n_without_screen = grab(audit_text, r"training compounds without a screen reading\\s+\\[(\\d+) of")
n_training_total = grab(audit_text, r"training compounds without a screen reading\\s+\\[\\d+ of (\\d+)\\]")
d6_active = grab(audit_text, r"every compound with a direct curve is screen-active\\s+\\[(\\d+) of")
d6_curves = grab(audit_text, r"every compound with a direct curve is screen-active\\s+\\[\\d+ of (\\d+);")
d6_active_nocurve = grab(audit_text, r"screen-active without a curve (\\d+) of")
d6_nocurve = grab(audit_text, r"screen-active without a curve \\d+ of (\\d+)\\]")
n_checks_passed = grab(audit_text, r"(\\d+) passed, \\d+ failed")
n_checks_failed = grab(audit_text, r"\\d+ passed, (\\d+) failed")
extract_text = (LOGS9 / "ext_cont_extract.log").read_text(errors="ignore")
n_ncats = grab(extract_text, r"scored on (\\d+) external compounds")
n_tox = grab(extract_text, r"scored on (\\d+) Tox21 compounds")
n_tox_trained = grab(extract_text, r"tresp_cyp2d6: trained on (\\d+) Tox21")
basic_text = (LOGS9 / "basic_centre_extract.log").read_text(errors="ignore")
n_bc_total = grab(basic_text, r"(\\d+) compounds, \\d+ with a basic centre")
n_basic = grab(basic_text, r"compounds, (\\d+) with a basic centre")

perm = {}
for iso in ("CYP3A4", "CYP2D6"):
    found = re.search(r"z0=1\\.0\\s+real labels: AUC ([\\d.]+), MCC ([+-][\\d.]+) \\| permuted: AUC ([\\d.]+) \\(range ([\\d.]+) to ([\\d.]+)\\), MCC ([+-][\\d.]+)", (LOGS9 / f"permutation_control_{iso}.log").read_text(errors="ignore"))
    perm[iso] = dict(zip(("real_auc", "real_mcc", "perm_auc", "perm_lo", "perm_hi", "perm_mcc"), map(float, found.groups())))

curves = {iso: pd.read_csv(ROOT / f"data/tdi_track/outputs/comparisons/softmax_curve_{iso}.csv") for iso in ("CYP3A4", "CYP2D6")}
def curve_summary(iso, z0=1.0):
    auc = curves[iso][curves[iso]["z0"] == z0].set_index("selector")["outer_auc"]
    fixed = auc[[name for name in auc.index if name.startswith("softmax_T")]]
    return {"lowest": float(fixed.min()), "highest": float(fixed.max()), "committee": float(auc["softmax_Tinf"]), "logistic": float(auc["stack_logit"]), "chosen": float(auc["softmax"])}
curve_3a4, curve_2d6 = curve_summary("CYP3A4"), curve_summary("CYP2D6")
print(f"audit: {n_checks_passed} passed, {n_checks_failed} failed; screen {n_screened} compounds; NCATS {n_ncats}; Tox21 {n_tox}/{n_tox_trained}; basic centre {n_basic}/{n_bc_total}; softmax curve CYP2D6 committee {curve_2d6['committee']:.4f} against logistic {curve_2d6['logistic']:.4f}")
'''

LEDGER9 = '''
# The ledger of the stacked gates, with the correction for the number of attempts. Every experiment of the project that was put through the gate of 6.7 and logged a stacked comparison (or, for a selector, a comparison of the blends) at both
# confidence cut-offs is one row; the logs of the 10-fold replicates are left out, since they repeat comparisons already in the family. The p-value of a row is the larger of its two, because the gate needs both
# (cypmode/challenge/multiplicity.py, the project's own module, whose Holm function is the one of 6.5).
from cypmode.challenge.multiplicity import collect, holm

ledger_all = collect(LOGS9, patterns=("*_CYP*.log",))
def has_gate_line(name):
    text = (LOGS9 / name).read_text(errors="ignore")
    return bool(re.search(r"GATE \\((stacked|selector|population screen stacked)", text)) and "_f10" not in name
def gate_verdict(name, iso):
    found = re.findall(rf"^{iso} GATE \\((?:stacked|selector|population screen stacked)[^)]*\\): (MET|NOT MET)", (LOGS9 / name).read_text(errors="ignore"), re.M)
    return found[-1] if found else "no verdict"
ledger = ledger_all[ledger_all["log"].map(has_gate_line)].copy().reset_index(drop=True)
ledger["Holm-adjusted p"] = holm(ledger["p_both"].tolist())
ledger["gate"] = [gate_verdict(row["log"], row["isoform"]) for _, row in ledger.iterrows()]
ledger = ledger.rename(columns={"diff_z1.0": "AUC change at z0 1.0", "p_z1.0": "p at z0 1.0", "diff_z1.96": "AUC change at z0 1.96", "p_z1.96": "p at z0 1.96", "p_both": "p (larger of the two)"})
show_table(ledger.sort_values("p (larger of the two)").head(10)[["log", "isoform", "AUC change at z0 1.0", "p at z0 1.0", "AUC change at z0 1.96", "p at z0 1.96", "p (larger of the two)", "Holm-adjusted p", "gate"]], floatfmt=".4f")

n_ledger = len(ledger)
n_ledger_3a4, n_ledger_2d6 = int((ledger["isoform"] == "CYP3A4").sum()), int((ledger["isoform"] == "CYP2D6").sum())
ledger_met = ledger[ledger["gate"] == "MET"]["log"].tolist()
n_below_one, n_below_196, n_below_both = int((ledger["p at z0 1.0"] < 0.05).sum()), int((ledger["p at z0 1.96"] < 0.05).sum()), int((ledger["p (larger of the two)"] < 0.05).sum())
smallest_holm = float(ledger["Holm-adjusted p"].min())
median_one, median_196 = float(ledger["AUC change at z0 1.0"].median()), float(ledger["AUC change at z0 1.96"].median())
positive_one, positive_196 = int((ledger["AUC change at z0 1.0"] > 0).sum()), int((ledger["AUC change at z0 1.96"] > 0).sum())
largest_change = float(ledger["AUC change at z0 1.0"].max())
n_above_floor = int((ledger["AUC change at z0 1.0"] > floor_main).sum())
expected_false = 0.05 * n_ledger
ledger_met_text = ", ".join(f"`{name}`" for name in ledger_met) if ledger_met else "none"
print(f"{n_ledger} logged stacked comparisons ({n_ledger_3a4} CYP3A4, {n_ledger_2d6} CYP2D6); gates met: {ledger_met}; raw p below 0.05 at z0 1.0 / 1.96 / both: {n_below_one} / {n_below_196} / {n_below_both}; smallest Holm-adjusted p {smallest_holm:.3f}")
'''

WHY_NO_GAIN = r"""
# The diagnostics of why the ideas that did not join the final stack added nothing, read from the table that job 097 generates (data/tdi_track/outputs/final/why_no_gain.csv). For every group of candidates and
# each of the two confidence cut-offs it holds the paired change in outer AUC with its corrected p value, the redundancy of the group's predictions with the stack's, the weight the stack gave the group,
# the smallest effect the design could detect, and the reason the module assigns by its fixed rules; the equivalence column is the separate question whether the 90 percent interval of the change lies inside
# the margin of +-0.01 AUC that the module calls equivalence. The rows of a group that joined the final stack are left out.
why_gain = pd.read_csv(ROOT / "data/tdi_track/outputs/final/why_no_gain.csv")
why_gain = why_gain[~why_gain["in_final_stack"]].reset_index(drop=True)
why_gain["names no weight"] = why_gain["reason"].str.contains("almost no weight")
why_gain["names redundancy"] = why_gain["reason"].str.contains("largely redundant")
why_gain["names a lowered AUC"] = why_gain["reason"].str.contains("lowered the AUC")
why_gain["names a change below the detectable effect"] = why_gain["reason"].str.contains("smaller than the smallest effect")
why_summary_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    why_part = why_gain[why_gain["isoform"] == iso]
    why_summary_rows.append({"enzyme": iso, "group comparisons": len(why_part), "groups of candidates": why_part["group"].nunique(),
                             "equivalent to adding nothing": int((why_part["equivalence"] == "equivalent").sum()), "inconclusive": int((why_part["equivalence"] == "inconclusive").sum()),
                         "worse (interval below the margin)": int((why_part["equivalence"] == "worse").sum()),
                             "reason: almost no weight": int(why_part["names no weight"].sum()), "reason: redundancy": int(why_part["names redundancy"].sum()),
                             "reason: lowered the AUC": int(why_part["names a lowered AUC"].sum()), "median smallest detectable change": float(why_part["detectable_effect"].median())})
why_table = pd.DataFrame(why_summary_rows)
show_table(why_table, floatfmt=".4f")
n_why = len(why_gain)
n_why_3a4, n_why_2d6 = int((why_gain["isoform"] == "CYP3A4").sum()), int((why_gain["isoform"] == "CYP2D6").sum())
n_equivalent = int((why_gain["equivalence"] == "equivalent").sum())
n_inconclusive = int((why_gain["equivalence"] == "inconclusive").sum())
n_worse = int((why_gain["equivalence"] == "worse").sum())
worse_sentence = "" if n_worse == 0 else f", and in {n_worse} the whole interval lies below the margin, which the module calls worse"
n_flagged = int((why_gain["corrected_p"] < 0.05).sum())
n_flagged_inconclusive = int(((why_gain["corrected_p"] < 0.05) & (why_gain["equivalence"] == "inconclusive")).sum())
n_flagged_below_effect = int(((why_gain["corrected_p"] < 0.05) & (why_gain["delta_auc"].abs() < why_gain["detectable_effect"])).sum())
if n_flagged and n_flagged_below_effect == n_flagged and n_flagged_inconclusive == n_flagged:
    flagged_sentence = "For each of them the change is still below the smallest detectable effect, and the equivalence verdict is inconclusive rather than equivalent"
elif n_flagged:
    flagged_sentence = f"For {n_flagged_below_effect} of them the change is still below the smallest detectable effect, and for {n_flagged_inconclusive} the equivalence verdict is inconclusive rather than equivalent"
else:
    flagged_sentence = "None of them has a corrected p value below 0.05"
n_no_weight = int(why_gain["names no weight"].sum())
n_redundancy = int(why_gain["names redundancy"].sum())
n_lowered = int(why_gain["names a lowered AUC"].sum())
n_below_effect = int(why_gain["names a change below the detectable effect"].sum())
n_raw_below = int((why_gain["corrected_p"] < 0.05).sum())
n_shapiro = int((why_gain["shapiro_p"] < 0.05).sum())
median_redundancy = float(why_gain.loc[why_gain["names redundancy"], "redundancy_mean"].median())
median_floor_3a4 = float(why_gain.loc[why_gain["isoform"] == "CYP3A4", "design_floor"].median())
median_floor_2d6 = float(why_gain.loc[why_gain["isoform"] == "CYP2D6", "design_floor"].median())
nearest_table = why_gain.sort_values("delta_auc", ascending=False).head(4)[["group", "isoform", "z0", "delta_auc", "corrected_p", "detectable_effect", "equivalence"]]
nearest_table = nearest_table.rename(columns={"group": "group of candidates", "z0": "confidence cut-off", "delta_auc": "change in AUC", "corrected_p": "corrected p", "detectable_effect": "smallest detectable change"})
show_table(nearest_table, floatfmt=".4f")
assert n_equivalent + n_inconclusive + n_worse == n_why
assert set(why_gain["equivalence"]) <= {"equivalent", "inconclusive", "worse"}
assert (why_gain["detectable_effect"] > 0).all()
"""

ALL_IN_ONE = '''
# The all-in-one comparison (cypmode/challenge/all_in_one.py, queue job 254_all_in_one_comparison), read from its logs: the final stack of each enzyme against one stack of the default candidates and every candidate of every group
# whose cross-validation predictions are complete, whatever the verdict of the group's gate, on the experiments' plan with the corrected resampled t. The logs are named so that no gate parser reads them.
all_in_one = {}
for iso in ("CYP3A4", "CYP2D6"):
    log = ROOT / f"data/structure_track/outputs/cyp3a4/logs/exploratory_all_in_one_{iso}.log"
    text = log.read_text() if log.exists() else ""
    head = re.search(r"final stack of (\\d+) candidates.*?all-in-one stack of (\\d+) candidates from (\\d+) groups", text)
    cuts = re.findall(r"z0=(\\S+)\\s+outer AUC final ([\\d.]+)\\s+all-in-one ([\\d.]+)\\s+all-in-one vs final: AUC ([+-][\\d.]+) \\(p=([\\d.]+)\\)", text)
    verdict = re.search(r"ALL-IN-ONE .*: (MET|NOT MET)", text)
    all_in_one[iso] = {"final": int(head.group(1)), "everything": int(head.group(2)), "groups": int(head.group(3)), "cuts": [(float(z), float(a), float(b), float(d), float(q)) for z, a, b, d, q in cuts], "verdict": verdict.group(1)} if head and cuts and verdict else None
all_in_one_rows = [{"enzyme": iso, "z0": z, "final stack": a, "all-in-one": b, "change": d, "corrected p": q, "candidates (final, all-in-one)": f"{r['final']}, {r['everything']}"} for iso, r in all_in_one.items() if r for z, a, b, d, q in r["cuts"]]
if all_in_one_rows:
    all_in_one_table = pd.DataFrame(all_in_one_rows)
    show_table(all_in_one_table, floatfmt=".4f")
    all_in_one_verdicts = {iso: (r["verdict"] if r else "not run yet") for iso, r in all_in_one.items()}
else:
    all_in_one_verdicts = {"CYP3A4": "not run yet", "CYP2D6": "not run yet"}
cuts_3a4 = all_in_one["CYP3A4"]["cuts"] if all_in_one["CYP3A4"] else []
cuts_2d6 = all_in_one["CYP2D6"]["cuts"] if all_in_one["CYP2D6"] else []
if len(cuts_3a4) == 2 and all(c[3] > 0 for c in cuts_3a4) and sum(c[4] < 0.05 for c in cuts_3a4) == 1:
    reading_3a4 = "On CYP3A4 the larger stack is ahead at both cut-offs, by about a hundredth of an AUC, and the difference is significant at one cut-off and not at the other"
else:
    reading_3a4 = "The changes on CYP3A4 are in the table"
if len(cuts_2d6) == 2 and all(abs(c[3]) < 0.01 for c in cuts_2d6) and len({c[3] > 0 for c in cuts_2d6}) == 2:
    reading_2d6 = "with changes of a few thousandths in both directions"
else:
    reading_2d6 = "with the changes of the table"
n_all_3a4 = all_in_one["CYP3A4"]["everything"] if all_in_one["CYP3A4"] else 0
n_groups_3a4 = all_in_one["CYP3A4"]["groups"] if all_in_one["CYP3A4"] else 0
'''


CELLS = [
    md("""
## 9.9 The neural pair: a multitask network and plain Chemprop

> **Field:** Machine learning · Statistics

*Where this section sits:* 4.7 and 4.8 taught the two kinds of network that appear here; 6.7 fixed the rule an addition must meet (the gate) and the replicate that confirms it; 8.1 defined the stack of 23 members to which every addition is added. This section is the one case in Part IX, apart from the selection-aware model of 9.1 (which answers a different question), in which an addition cleared the gate, and the only one in which the gate had to be repeated.

**The two members.** The 23 members of the default stack are tree ensembles, nearest-neighbour models and support vector machines on the Tanimoto kernel (8.1); none is a neural network. Two networks were added to it, each in the way the data suggest. The **multitask network** reads the logarithm of the Morgan counts and the RDKit descriptors, each descriptor transformed so that its values follow a normal distribution (a quantile transformation fitted on the training rows). It passes them through a shared trunk of two layers of 1,024 and 512 units (4.7: batch normalisation, ReLU, dropout 0.35) and ends in nine heads: one regression head for each assay arm (the direct and the TDI pIC50 of the four enzymes, eight in all; the loss ignores the arms that a compound lacks) and one classification head for the label of the enzyme in question. It is trained for 60 epochs with AdamW and a one-cycle learning rate, three seeds are averaged, and it gives two scores, the probability of the label and the predicted shift $\\hat t - \\max(\\hat d, 4)$. **Plain Chemprop** is the directed message-passing network of 4.8 on the molecular graph, multitask over the same eight arms, trained for 30 epochs with a tenth of its training rows held out to stop it early, and scored by the predicted shift. "Plain" distinguishes it from the CheMeleon variant, which starts from pretrained weights and has its own staged design (9.10). Both networks learn from the arms of all four enzymes while predicting one: the information that 9.7 offered the tree models as predicted features, without effect, enters here as extra targets that shape one shared representation.

The first cell reads what each score achieved alone (the diagnostic lines of the experiment logs) and how closely each agrees in ranking with the best single tree model of the default stack, `lgbm_A+fps_deep`, on the outer folds of the main design (the rank correlation of 4.10, averaged over the folds). A member can add to a stack only if it does not repeat the members already there, and a rank correlation well below 1 is the first sign that it does not.
"""),
    code(NEURAL_MEMBERS),
    md("""
On CYP3A4 the multitask network's shift score ranks with an AUC of {{=member_table.iloc[1]['outer AUC alone at z0 1.0']:.3f}} at $z_0 = 1.0$ and {{=member_table.iloc[1]['outer AUC alone at z0 1.96']:.3f}} at $z_0 = 1.96$, its label score with {{=member_table.iloc[0]['outer AUC alone at z0 1.0']:.3f}} and {{=member_table.iloc[0]['outer AUC alone at z0 1.96']:.3f}}, and plain Chemprop with {{=member_table.iloc[2]['outer AUC alone at z0 1.0']:.3f}} and {{=member_table.iloc[2]['outer AUC alone at z0 1.96']:.3f}}. The scores agree with the tree model in rank at {{=member_table.iloc[1]['rank correlation with lgbm_A+fps_deep']:.2f}} (the network's shift), {{=member_table.iloc[0]['rank correlation with lgbm_A+fps_deep']:.2f}} (its label) and {{=member_table.iloc[2]['rank correlation with lgbm_A+fps_deep']:.2f}} (Chemprop), and the two networks with each other at {{=member_table.iloc[1]['rank correlation with chemprop_mt_shift']:.2f}}. Plain Chemprop is the member that differs most from the trees, and it is also the weakest alone. On CYP2D6 it is hardly better than chance (an AUC of {{=member_table.iloc[5]['outer AUC alone at z0 1.0']:.3f}} and {{=member_table.iloc[5]['outer AUC alone at z0 1.96']:.3f}}, a rank correlation of {{=member_table.iloc[5]['rank correlation with lgbm_A+fps_deep']:.2f}} with the trees), whereas the network's label score reaches {{=member_table.iloc[3]['outer AUC alone at z0 1.0']:.3f}} and {{=member_table.iloc[3]['outer AUC alone at z0 1.96']:.3f}}.

The second cell is the gate of the main design: each network added to the default stack, and the two together, with the corrected resampled t of 6.5.
"""),
    code(NEURAL_GATES),
    md("""
On CYP3A4 neither network clears the gate alone. The multitask network adds {{=mlp_3a4['AUC gain at z0 1.0']:+.4f}} AUC (p = {{=mlp_3a4['p at z0 1.0']:.3f}} at $z_0 = 1.0$ and {{=mlp_3a4['p at z0 1.96']:.3f}} at 1.96) and plain Chemprop {{=chemprop_3a4['AUC gain at z0 1.0']:+.4f}} (p = {{=chemprop_3a4['p at z0 1.0']:.3f}} and {{=chemprop_3a4['p at z0 1.96']:.3f}}); both are positive in {{=mlp_3a4['folds better at z0 1.0']}} folds. They are near-misses, a gain that the design almost sees. Together they add {{=pair_3a4['AUC gain at z0 1.0']:+.4f}} AUC (p = {{=pair_3a4['p at z0 1.0']:.3f}} and {{=pair_3a4['p at z0 1.96']:.3f}} at the two cut-offs), positive in {{=pair_3a4['folds better at z0 1.0']}} folds at the first, with an average precision that rises by {{=pair_3a4['AP gain at z0 1.0']:+.4f}}. The gate is {{=pair_3a4['gate']}} for the pair. The gain of the pair is a little less than the sum of the gains of the two ({{=mlp_3a4['AUC gain at z0 1.0']:+.4f}} and {{=chemprop_3a4['AUC gain at z0 1.0']:+.4f}} make {{=mlp_3a4['AUC gain at z0 1.0'] + chemprop_3a4['AUC gain at z0 1.0']:+.4f}}), as expected when the two overlap in what they know, so that nothing in the pair is more than its parts: two contributions of the same sign, each too small for the design to separate from zero, add up to one that it can. On CYP2D6 the pair adds {{=pair_2d6['AUC gain at z0 1.0']:+.4f}} (p = {{=pair_2d6['p at z0 1.0']:.2f}}); there is nothing to confirm.

**Why this result is not accepted as it stands.** The pair was not on the project's list of candidates before the results existed. It was formed after the two near-misses had been seen, because they pointed the same way. Choosing which candidates to combine after looking at how they did is a second round of selection that the p-value of the combination does not count, so that the p-values of the pair, {{=pair_3a4['p at z0 1.0']:.3f}} and {{=pair_3a4['p at z0 1.96']:.3f}}, are those of the best of several possible groupings and are optimistic by an amount that cannot be computed. The project's rule for such a case, written into the final build as `CONFIRMED_GROUPS`, is that a group found after looking joins the stack only if a second, pre-declared design confirms it. The rule was declared on 2026-10-16, after the fifteen-fold result and before any ten-fold result existed, in the docstring of `cypmode/challenge/replicate_verdict.py`: the pair is compared with the default stack on the 10-outer-fold plan of 6.7 (30 folds, each model fitted on nine tenths of the compounds, the variance factor of the corrected t falling from $1/15 + 1/4 = 0.317$ to $1/30 + 1/9 = 0.144$); the replicate is MET only if the ten-fold comparison has p < 0.05 and a positive gain at both cut-offs and the five-fold comparison has a positive gain at both. It was declared for CYP3A4 only, because the pair showed nothing on CYP2D6. Filling the ten-fold caches is the costly step: plain Chemprop needs about 136 seconds for each of 155 tasks, on a processor.
"""),
    code(NEURAL_REPLICATE),
    code(NEURAL_BUILD),
    md("""
The replicate rule is {{='MET' if replicate_met else 'NOT MET'}}. In the ten-fold design the pair adds {{=ten_fold[1.0][0]:+.4f}} AUC at $z_0 = 1.0$ (p = {{=ten_fold[1.0][1]:.3f}}, positive in {{=better_ten.loc[1.0]}} folds) and {{=ten_fold[1.96][0]:+.4f}} at $z_0 = 1.96$ (p = {{=ten_fold[1.96][1]:.3f}}, {{=better_ten.loc[1.96]}} folds), with the average precision rising by {{=ap_ten[1.0][0]:+.4f}} and {{=ap_ten[1.96][0]:+.4f}}. The size of the gain hardly changes between the two designs ({{=five_fold[1.0][0]:+.4f}} and {{=ten_fold[1.0][0]:+.4f}}) and the p-values are in the same range, which is what a real, small effect does and what a lucky partition would be unlikely to do. The last part of the cell sets the discovery in its family: of the {{=n_family9}} stacked comparisons logged for CYP3A4, {{=n_raw_below}} {{='has' if n_raw_below == 1 else 'have'}} a raw p below 0.05, and the pair's {{=pair_raw_p:.3f}} becomes {{=pair_holm_p:.2f}} after Holm's adjustment over all of them. Seen as the best of {{=n_family9}} tries, the pair would be nothing; seen as one hypothesis, stated before its test and passed, it is a result. The replicate is the second view, and it has a limit that should be stated plainly. The ten-fold plan cuts the *same* compounds into different folds. It protects against an unlucky or a lucky partition and it is a more powerful test (the variance factor halves); it does not protect against the compounds themselves, because an effect that exists in this sample by chance is in the ten-fold folds as well. Only compounds that the project has not seen can decide that, and the first of them are the 750 of the blinded set.

The pair therefore enters the CYP3A4 stack under the rule that was written for it, as an addition of a small and consistent gain (about one hundredth of an AUC) and not as an established improvement; the evidence for it is a pre-declared test that passed, with the caveats above. {{=neural_stack_sentence}} The effect on the MCC at the call rate that the project transfers is reported with the final results, in Part XIII. CYP2D6 does not take the pair.

*On what actually ran.* `cypmode/challenge/nested_runners_torch.py` (`fit_predict_multitask`, the groups `mlp_multitask` and `mlp_multitask_aux`; the second is the version with the official screen and Emax readings as extra targets, which did not pass and appears in the ledger of 9.11), `cypmode/challenge/nested_runners_chemprop.py` (`fit_predict`, the variant `chemprop_mt_shift`, which calls the Chemprop command line under the neural-network environment), `cypmode/challenge/cached_candidates.py` (the group `neural`), `cypmode/challenge/aux_experiments.py` (the experiments `mlp`, `chemprop` and `neural`, run through `addition_experiment.py`), `cypmode/challenge/replicate_verdict.py` (the rule and its verdict lines) and `cypmode/challenge/final_tdi_build.py` (`CONFIRMED_GROUPS`). The queue jobs are `198_mlp_addition` (the multitask network added to the stack), `206_chemprop_addition` (plain Chemprop added to the stack), `232_neural_f10_fill_CYP3A4` (the ten-fold caches), `233_neural_addition` (the fifteen-fold addition for both enzymes), `234_neural_addition_f10` (the ten-fold addition) and `235_neural_replicate_verdict` (the verdict). The logs are `aux_mlp_<enzyme>.log`, `aux_chemprop_<enzyme>.log`, `aux_neural_<enzyme>.log`, `aux_neural_f10_CYP3A4.log` and `replicate_verdict_<enzyme>.log`.
"""),
    md("""
## 9.10 The other additions, tested the same way

> **Field:** Machine learning · Cheminformatics · Biology

*Where this section sits:* 9.4 to 9.9 took one idea each and followed it through a section. The additions of this one were put through the same procedure (the nested design of 6.3, the stack of 23 members of 8.1, and the gate of 6.7: a positive gain with a corrected p below 0.05 at both confidence cut-offs), on both enzymes, and each of them would have needed a section of its own had it passed. None has, so far, and the ledger of 9.11 counts them. They fall into five families: models pretrained by others on very large collections, which serve here as sources of features; measurements from outside the challenge; the organisers' extra readings of the challenge's own compounds; a block of chemistry written for CYP2D6; and two changes to the blend itself, which add no member to the stack.

The table below gives every verdict at once. It is built from the logs of the experiments when the page is built: an addition whose queue job had not finished at that moment shows "not run yet", and the text says so. The columns are the change of the AUC of the whole stack when the addition joins it, at the two cut-offs, with the corrected resampled t of 6.5 (15 folds), and the verdict of the gate. In the sentences below a change is given in the same order: the value at $z_0 = 1.0$ with its p-value, then the value at $z_0 = 1.96$ with its p-value.
"""),
    code(ADDITIONS),
    md("""
{{=n_tests_9}} comparisons have been run, {{=n_met_9}} of them meet their gate, and {{=n_pending_9}} had not been run when this page was built. The largest change of the AUC at $z_0 = 1.0$ in the whole table is {{=best_gain_9['AUC change at z0 1.0']:+.4f}} ({{=best_gain_9['addition']}}, {{=best_gain_9['enzyme']}}, p = {{=best_gain_9['p at z0 1.0']:.2f}}); it is smaller than the least difference that the design recognises 8 times in 10 (6.7), so that it cannot be told from noise. The families follow, with what each addition is, why it was tried, and what was measured.
"""),
    md("""
**Pretrained tables and representations.** Section 4.9 explained tabular in-context learning, and 9.4 a frozen foundation model. Five more of the same kind were tried, each as a source of features or as an extra member.

*TabICL and TabPFN.* Both are transformers pretrained on millions of artificial tables, which read a table of labelled rows together with the rows to be predicted and output the missing labels in one pass (4.9). TabICL was given the 128 components of the truncated SVD of the logarithm of the Morgan counts together with the quantile-normalised descriptors; TabPFN (Hollmann and co-workers) the same inputs. Change of the AUC when added to the stack: TabICL, CYP3A4 {{=line_of('TabICL on the descriptors', 'CYP3A4')}}, CYP2D6 {{=line_of('TabICL on the descriptors', 'CYP2D6')}}; TabPFN, CYP3A4 {{=line_of('TabPFN', 'CYP3A4')}}, CYP2D6 {{=line_of('TabPFN', 'CYP2D6')}}.

*Monroe.* Monroe (Banaszewski, arXiv 2608.18982) is a **graph transformer** of the GRIT kind (an attention network over the atoms of a molecule, whose attention is told where each atom sits in the graph) pretrained on two things: the properties of about 81 million molecules computed with a semi-empirical quantum-chemical method (PM6; 7.8 and 10.4), and sparse PubChem bioassay labels. It reads a three-dimensional conformer of the molecule, generated with RDKit's ETKDG method, and returns 720 numbers. The project wrote its own wrapper, which reads the weights as plain tensors (so that no code stored in the file can run) and checks that every tensor of the encoder is found and none is left over. The 720 numbers are reduced to 64 principal components for the tree models, as for CheMeleon, and to 128 for TabICL. Change of the AUC when added to the stack: the block, CYP3A4 {{=line_of('Monroe embeddings', 'CYP3A4')}}, CYP2D6 {{=line_of('Monroe embeddings', 'CYP2D6')}}; TabICL on the Monroe embeddings, CYP3A4 {{=line_of('TabICL on the Monroe', 'CYP3A4')}}, CYP2D6 {{=line_of('TabICL on the Monroe', 'CYP2D6')}}.

*Atom-JEPA.* A **joint-embedding predictive architecture** (JEPA) is pretrained not to rebuild its input but to predict, from one part of an input, the learned representation of another part, so that it learns what the parts say about each other rather than the details of their pixels or atoms. Atom-JEPA (the authors' code and weights are at `github.com/khelverskovp/atom-jepa` and `huggingface.co/atom-jepa/atom-jepa`) is an **equivariant** graph network over three-dimensional conformers (if the molecule is rotated, the vectors inside the network rotate with it, so that it need not learn every orientation separately), with 39.0 million parameters, pretrained on about 19 million conformers. The authors fine-tune the whole network. On this machine one optimiser step cost 7.4 seconds per molecule on the processor and the graphics processor ran out of memory at 16 molecules per batch, which makes one fine-tuning of the 3,500 training molecules of a task about three days and the nested design needs about 80 of them for each enzyme. What is affordable is one forward pass per molecule, about half a second on the graphics processor, so the encoder is **frozen** (its weights never change) and every one of its eight blocks is kept (the mean over atoms of the features of each block), because frozen features are usually best in the middle blocks. The same pass stores the features of every atom for blocks 4, 6 and 8, and a small attention readout (a network that learns which atoms matter for the label) is trained on them in seconds per fit: the affordable form of fine-tuning the top of the network. The forward pass and both caches are made by `cypmode/challenge/atomjepa_embeddings.py` (queue job 228), which keys them by canonical SMILES and saves its chunks as they finish, so that an interrupted run resumes where it stopped; the evaluation reads the cache and never imports the network library. Change of the AUC when added to the stack: the embeddings, CYP3A4 {{=line_of('Atom-JEPA embeddings', 'CYP3A4')}}, CYP2D6 {{=line_of('Atom-JEPA embeddings', 'CYP2D6')}}; the readout, CYP3A4 {{=line_of('Atom-JEPA attention', 'CYP3A4')}}, CYP2D6 {{=line_of('Atom-JEPA attention', 'CYP2D6')}}.
"""),
    code(CHEMELEON_STATE),
    md("""
*CheMeleon, fine-tuned.* 9.4 used CheMeleon frozen. The other way to use a pretrained network is to start from its weights and train the whole of it on the project's labels (fine-tuning). That is plain Chemprop of 9.9 with the CheMeleon weights as the starting point, and it costs about {{=cpu_minutes_per_task:.0f}} minutes per task on the processor. A rule fixed on 2026-10-16, before any CheMeleon result existed, spends that cost in two stages (`cypmode/challenge/chemeleon_stage.py`). Stage 1 fits repeat 0 only, which is {{=n_stage_one}} cross-validation tasks for an enzyme ({{=stage_one_cpu_hours:.0f}} hours on the processor), and compares the stack with and without the network. Stage 2, the other {{=n_full_design - n_stage_one}} tasks of the design (two more repeats and the deployment fits), runs for an enzyme only if stage 1 shows a mean AUC gain of at least {{=chemeleon_min_gain}} at the first cut-off and a gain in at least {{=chemeleon_min_folds}} of the 5 folds. The stage-1 comparison is written to a text file and not to a gate log on purpose: it is a screen, not a pre-declared test, and it must not enter the ledger. The rule can only lower the number of tests that are run, so it cannot raise the share of false passes of the ordinary 15-fold gate that follows stage 2.

A second decision, also taken on 2026-10-16 before any CheMeleon measurement, was about cost and about what the network could add. CYP2D6 was dropped from the design: plain Chemprop is at chance there (an outer AUC of 0.54 and 0.53) and the frozen CheMeleon block reaches 0.63. For CYP3A4, the plain network added only +0.0048 AUC to the default stack (9.9), a gain below the least difference of about 0.01 that the design recognises, so that a fine-tuned network that adds a few thousandths on top of a stack that already holds the pair of 9.9 could not produce a decision-grade result; it is worth running only if it is cheap. A speed test decides (`cypmode/challenge/chemeleon_speed.py`, queue job `202_chemeleon_speed`): one task that is already in the cache from the processor is trained again on the Apple GPU, in a scratch directory, and the two results are compared; the GPU is accepted when the rank correlation of the scores is at least {{=chemeleon_speed.MIN_AGREEMENT}} and the AUC differs by at most {{=chemeleon_speed.MAX_AUC_DIFFERENCE}}; stage 1 then runs only if the missing tasks fit in {{=chemeleon_speed.STAGE_ONE_HOURS_LIMIT:.0f}} hours, on the GPU if it is accepted and otherwise on the processor; if they do not fit, the fine-tuned CheMeleon is not run and the ledger says \"not run: cost\", which is not a verdict. {{=chemeleon_sentence}} Two further variants start from other pretraining weights of the OpenADMET project (`chemeleon_minimol_mt_shift` and `chemeleon_surrogate_mt_shift` in `nested_runners_chemprop.py`; `cypmode/challenge/foundation_weights.py` wraps those weight files so that Chemprop accepts them); The first speed test of 2026-10-17 ended on a determinism error of the GPU backend before any GPU time was measured; the second, on 2026-10-22, went through a launcher (`scripts/chemprop_mps.sh` and `scripts/chemprop_warn_only.py`) that lets PyTorch run the one operation of the message passing that has no deterministic implementation on the GPU, with the seed fixed, so that a GPU task is repeatable to a tolerance and not byte for byte; the rule was unchanged. The two OpenADMET weights follow the same two stages for CYP3A4 (queue jobs `255_openadmet_weights_stage1_CYP3A4`, `256_openadmet_weights_stage1_screen`, `257_openadmet_weights_stage2_CYP3A4` and `258_openadmet_weights_addition`), added to the stack together as the group `chemprop_foundation`, so that their screen and their gate are one comparison. {{=foundation_sentence}}

*Atom-JEPA, a second look.* The readout on the atoms came closest to its gate on CYP3A4 (the numbers are in the table above and the reading of them is in 9.11). A near miss may be looked at again only in a way that cannot change a model, so a replicate of the readout on the 10-outer-fold plan (queue jobs `252_atomjepa_head_f10_fill_CYP3A4` and `253_atomjepa_head_f10_exploratory`) was declared before any ten-fold number of the readout existed and written to a log that no gate parser reads. It gives {{=atomjepa_sentence}}. The two plans use the same compounds, so this is the same estimate seen under another split and not independent evidence; the registered gate stays as it was.
"""),
    code(EXTRA_FACTS),
    md("""
**Measurements from outside the challenge.** The challenge compounds are a small part of what has been measured on cytochrome P450 enzymes, and three outside sources were turned into label-free blocks (2.8): columns computed from the structure alone, by models fitted to the outside measurements, never to the challenge labels. In every case a compound that is also a challenge compound (matched on the connectivity layer of the InChIKey, the part of the identifier that records the atoms and bonds) is removed before anything is fitted, so that the block scores the challenge compounds out of sample by construction and needs no nesting.

*The NCATS panel, continuous.* The binary active/inactive call of the same panel is already inside the stack (7.7). The panel keeps more: for every compound and enzyme the **maximal response** of its concentration-response curve (a signed percent, with −100 for complete inhibition and 0 for none) and, for the compounds with a fitted curve, the potency AC50. A binary call throws away how strongly a compound inhibited and how far from active an "inactive" one was. A regressor per enzyme predicts the response and another the potency from structure; the first is trained on {{=n_ncats:,}} outside compounds, and the block has ten columns. Change of the AUC: CYP3A4 {{=line_of('NCATS qHTS', 'CYP3A4')}}; CYP2D6 {{=line_of('NCATS qHTS', 'CYP2D6')}}.

*The Tox21 CYP screen.* Tox21 (the National Center for Advancing Translational Sciences and the US Environmental Protection Agency) screened thousands of environmental chemicals and drugs against five enzymes with the same kind of assay; the block is built from {{=n_tox:,}} of its compounds, and each regressor is trained on {{=n_tox_trained:,}} of them, those with a usable reading after the cleaning described next. The compounds are far from the challenge library (the median nearest-neighbour similarity of a challenge compound to Tox21 is 0.30, and 4 to 7 percent reach 0.5), so what it can add is a picture of weak inhibition, the range that the challenge's own pool lacks, and not a read-across from neighbours. The assay reads out light from a **luciferase**, the enzyme of fireflies: the substrate offered to the cytochrome is turned into a molecule that the luciferase converts to light, so a compound that inhibits the luciferase itself dims the light and looks like a cytochrome inhibitor. The **counter-screen** (the same compounds tested against the luciferase alone, PubChem assay 1224835) identifies those compounds, and their readings are removed before anything is fitted. Change of the AUC: CYP3A4 {{=line_of('Tox21 CYP screen', 'CYP3A4')}}; CYP2D6 {{=line_of('Tox21 CYP screen', 'CYP2D6')}}.

*Predicted logD and protein binding.* **logD** is the lipophilicity of a compound at the pH of blood, and **protein binding** is how much of a compound sticks to the proteins around it instead of staying free; lipophilic, strongly bound compounds meet cytochrome enzymes more often. The descriptors of Part III hold only Crippen logP, a sum of atom contributions that knows nothing of ionisation or binding. Two Chemprop networks of the OpenADMET project, started from its pretraining weights and fine-tuned on one endpoint each of its ExpansionRx challenge (a different library from this one), return a number per structure: logD (test R² 0.83 on a cluster split, as the project's repository reports) and the logarithm of mouse brain protein binding (R² 0.74). The two are run once over every challenge compound by `cypmode/challenge/openadmet_surrogates.py` (queue job 227) and written as a two-column cache keyed by canonical SMILES; Chemprop runs in its own environment and the evaluation reads only the cache. On 200 training compounds the predicted logD correlated with Crippen logP at r = 0.76, so it is not a copy of the logP the descriptors already hold. Change of the AUC: CYP3A4 {{=line_of('OpenADMET logD', 'CYP3A4')}}; CYP2D6 {{=line_of('OpenADMET logD', 'CYP2D6')}}.

*PXR induction.* **PXR**, the pregnane X receptor, is a protein in liver and gut cells that senses foreign chemicals; when a compound binds and activates it, the cell switches on the genes of detoxification enzymes, CYP3A4 among them. Raising the amount of an enzyme is **induction**, the opposite of inhibition. The reason to try PXR readings is that the receptor and the enzyme both bind large, lipophilic ligands, so that a compound's behaviour on one may say something about the other; it is a plausible reason and the experiment is what tests it. The OpenADMET PXR challenge measured the same compound library (4,652 compounds, of which 1,897 are CYP training compounds, and none a blinded CYP test compound), as a fitted induction (the potency pEC50 and the plateau of the curve) and as a test at one concentration, the kind of file explained below for the CYP compounds themselves. The readings of four blinded CYP test compounds that appear in the one-concentration file are removed by the guard `exclude_test_compounds` before any value is computed or read. Change of the AUC: CYP3A4 {{=line_of('PXR induction', 'CYP3A4')}}; CYP2D6 {{=line_of('PXR induction', 'CYP2D6')}}.
"""),
    md("""
**The organisers' extra readings.** The dataset behind the challenge holds more than the training file read in Part II. The audit of the project (`cypmode/challenge/official_extra.py`, queue job `190_official_extra_audit`, {{=n_checks_passed}} checks passed and {{=n_checks_failed}} failed) found three more tables of the same compounds and one table of another kind, and checked that none holds a blinded test compound, by name or by structure. The **single-concentration screen** tested each compound at one concentration, 49.5 µM, on all four enzymes and reports the log₂ fold change of the signal: it is the first stage of the experiment, the screen from which the compounds with a dose-response curve were promoted (2.6), and it holds {{=n_screened:,}} compounds. The **Emax** file reports for both arms of all four enzymes the **maximal effect**, the plateau that the fitted concentration-response curve reaches (how much of the enzyme's activity the compound can remove at most, whatever its potency), for {{=n_emax_rows:,}} compounds, with the TDI label, which agrees with the training file wherever both give one. The direct-inhibition file ({{=n_direct_rows:,}} compounds) repeats the direct pIC50 of the training file and adds none. Of the {{=n_training_total:,}} training compounds, {{=n_without_screen:,}} have no screen reading. The audit also shows what the screen is for CYP2D6: every compound with a direct curve is screen-active ({{=d6_active:,}} of {{=d6_curves:,}}), and almost no other is ({{=d6_active_nocurve}} of {{=d6_nocurve:,}}), so the screen is the variable on which the selection into the training set acts, observed and not inferred. The fourth table, the reaction phenotyping of Octant (a CYP3A4 and CYP2J2 tier), carries no signal for the TDI label (1,150 compounds, three positives, AUC 0.46), and one of its compounds is also in the blinded set; the guard removes it and its value is never read.

The blinded compounds have no screen or Emax reading, so a reading can never be an input of the final model. It can still teach the model what the structure says about it, in the construction of 9.7 (`cypmode/challenge/aux_blocks.py`): a first-stage regressor is fitted on the compounds that have the reading and predicts it for every compound, with five-fold cross-fitting, and the second-stage classifier receives the predictions next to the structure features. Inside a nested task the first stage may read only the training rows of the task and the compounds outside the plan's pool, never an evaluation row. Change of the AUC: the predicted screen as a block, CYP3A4 {{=line_of('single-concentration screen', 'CYP3A4')}}, CYP2D6 {{=line_of('single-concentration screen', 'CYP2D6')}}; the predicted Emax, CYP3A4 {{=line_of('maximal effect', 'CYP3A4')}}, CYP2D6 {{=line_of('maximal effect', 'CYP2D6')}}; the screen as the gate of the selection-aware CYP2D6 model of 9.1 (`cypmode/challenge/screen_selection.py`; the gate is learned from the real screen instead of from "has a curve"), CYP2D6 {{=line_of('screen inside', 'CYP2D6')}}; the multitask network of 9.9 with the screen and the Emax as twelve extra regression targets, CYP3A4 {{=line_of('multitask network with', 'CYP3A4')}}, CYP2D6 {{=line_of('multitask network with', 'CYP2D6')}}.

Two more uses of the rows that the pool leaves out. For CYP3A4, 1,249 compounds have a TDI-arm curve and no direct curve, and are labelled negative by convention; of them, 194 are certain negatives (the label implies a TDI-arm pIC50 above 4.301, so a compound at or below it is negative whatever its direct curve is) and the rest can be given a soft label by a first-stage model fitted on the pool rows of the task (structure and the TDI-arm value as inputs; its five-fold AUC was 0.77, against 0.60 for the TDI-arm value alone). Used so, the change of the AUC of CYP3A4 is {{=line_of('the 1,249', 'CYP3A4')}}. And the label in two parts, $P(t > 4.301) \\times P(t - d > \\log_{10} 2)$, where the first part can be learned from every compound with a TDI-arm curve: CYP3A4 {{=line_of('the label in two parts', 'CYP3A4')}}; CYP2D6 {{=line_of('the label in two parts', 'CYP2D6')}}. The first part cannot help CYP2D6, where 94 percent of the rows with both arms already have a TDI-arm pIC50 above 4.301.
"""),
    md("""
**A block of chemistry for CYP2D6.** CYP2D6 binds substrates and inhibitors that carry a **protonated** basic nitrogen (one that has taken up a hydrogen ion and carries a positive charge in the assay buffer) a few ångströms from a flat aromatic ring: the cation meets two acidic residues of the pocket, Asp301 and Glu216, while the ring sits over the heme. A fingerprint sees the substructure. It does not know that the nitrogen is protonated at the pH of the assay, how many sites could be, or how far the charge sits from the ring in a real conformation. The basic-centre block (`cypmode/challenge/basic_centre.py`, queue jobs `191_basic_centre_extract` and `199_basic_centre_experiment`) computes those quantities from the structure alone: the **protonation states** of the molecule between pH 6.4 and 8.4 with Dimorphite-DL (which enumerates the states in which the ionisable groups can be found), the class of the most protonated state (primary, secondary or tertiary ammonium, amidinium or guanidinium, quaternary), the shortest bond path from a basic centre to an aromatic atom, and, from ten conformers of the most protonated state, the distances from each basic nitrogen to the aromatic ring centres and the share of conformers with a ring centre 4.5 to 7.5 Å away, the window of the CYP2D6 pharmacophore. {{=n_basic:,}} of the {{=n_bc_total:,}} compounds have a basic centre. The block is the principled form of the idea behind the trialkylamine flag of 7.2, which looked only for one substructure. Change of the AUC: CYP2D6, where it matters most, {{=line_of('basic-centre', 'CYP2D6')}}; CYP3A4 {{=line_of('basic-centre', 'CYP3A4')}}.

**Two changes to the blend.** The blend that turns the scores of the members into one score (8.2) is a logistic regression on their transformed ranks, and nothing stops it from giving a member a negative weight: in the final CYP3A4 stack `knn_tanimoto_k5` has weight −0.093, a member that is subtracted because it is correlated with better ones, which is more often a fit to the noise of the inner folds than a property of the chemistry. The **non-negative blend** (`cypmode/challenge/nonneg_stack.py`) is the same regression with every weight constrained to be at least zero, an average of experts that can be switched off but never inverted, solved with the L-BFGS-B method. The **softmax blend** (`cypmode/challenge/softmax_selector.py`) fills the line between two extremes with one parameter. At one end the best single member takes all the weight (exploitation of what the inner folds say), at the other every member weighs the same (exploration: no member is trusted over another); between them the weight of member $k$ is $w_k = \\exp(s_k/T)/\\sum_j \\exp(s_j/T)$, a **softmax** of its inner score $s_k$ with a **temperature** $T$ chosen by an inner cross-validation, where $T \\to 0$ is the best single member and $T \\to \\infty$ the equal-weight committee. Each is tested against the logistic blend on the cached predictions of the default stack, with the same folds and the same yardstick. Change of the AUC of the whole stack when the logistic blend is replaced: the non-negative blend, CYP3A4 {{=line_of('non-negative blend', 'CYP3A4')}}, CYP2D6 {{=line_of('non-negative blend', 'CYP2D6')}}; the softmax blend, CYP3A4 {{=line_of('softmax blend', 'CYP3A4')}}, CYP2D6 {{=line_of('softmax blend', 'CYP2D6')}}. The logistic blend stays; the negative weight of the nearest-neighbour member is therefore not a defect that costs accuracy, and the exploration-exploitation curve of the softmax blend is written next to the verdict (`data/tdi_track/outputs/comparisons/softmax_curve_<enzyme>.csv`) and drawn in the figure `ml_exploration_exploitation`. At $z_0 = 1.0$, for CYP3A4 the outer AUC of the softmax blend runs from {{=curve_3a4['lowest']:.4f}} to {{=curve_3a4['highest']:.4f}} over the grid of fixed temperatures, below the {{=curve_3a4['logistic']:.4f}} of the logistic blend at every one of them. For CYP2D6 it rises with the temperature, to {{=curve_2d6['committee']:.4f}} for the equal-weight committee, above the {{=curve_2d6['logistic']:.4f}} of the logistic blend. The gate tested the softmax blend with its temperature chosen by the inner cross-validation ({{=curve_2d6['chosen']:.4f}} for CYP2D6), and not the equal-weight committee; the committee's advantage on CYP2D6 was seen after the gate, was not tested, is small against the spread of the fold AUCs of that enzyme (about 0.05), and is not adopted.

**The control.** One more experiment of the family is not an addition: the **negative control** (`cypmode/challenge/permutation_control.py`, queue job `200_permutation_control`). The labels of the pool are permuted at random and the whole nested procedure is repeated; the outer AUC must be 0.5 and the MCC 0, because a model that is shown labels unrelated to the structures cannot rank them. The control is MET on both enzymes: at $z_0 = 1.0$ the permuted labels give an AUC of {{=perm['CYP3A4']['perm_auc']:.3f}} for CYP3A4 (range {{=perm['CYP3A4']['perm_lo']:.3f}} to {{=perm['CYP3A4']['perm_hi']:.3f}} over three permutations) and an MCC of {{=perm['CYP3A4']['perm_mcc']:+.3f}}, against {{=perm['CYP3A4']['real_auc']:.3f}} and {{=perm['CYP3A4']['real_mcc']:+.3f}} with the real labels, and {{=perm['CYP2D6']['perm_auc']:.3f}} and {{=perm['CYP2D6']['perm_mcc']:+.3f}} for CYP2D6, against {{=perm['CYP2D6']['real_auc']:.3f}} and {{=perm['CYP2D6']['real_mcc']:+.3f}}. It is the evidence that the nested pipeline carries no leak that survives a permutation of the labels: a leak (6.8) would show as an AUC above 0.5 on labels that carry no information.
"""),
    md("""
## 9.11 The ledger of the gates, and what the correction for the number of attempts leaves

> **Field:** Statistics

*Where this section sits:* 6.5 showed how the number of comparisons inflates the share of false positives and introduced Holm's adjustment, and 6.7 introduced the pre-declared gate and counted the comparisons that the project had logged. Part IX has since added the experiments of 9.1 to 9.10. This section puts the logged comparisons together, applies the correction over the whole family, and asks what the one gate that was met means once the number of attempts is counted. It closes Part IX.

**The family.** Every experiment of the project that went through the gate of 6.7 and logged a stacked comparison (or, for a selector, a comparison of the blends) at both confidence cut-offs is one row of the family (`cypmode/challenge/multiplicity.py`); the logs of the ten-fold replicates are left out, because they repeat comparisons that are already in it. A row has two p-values, one for each cut-off, and the gate needs both to be below 0.05, so the p-value of the row is the larger of the two. This is an **intersection-union test**: the claim is that the addition helps at both cut-offs, and it is rejected unless both comparisons support it. Holm's adjustment of 6.5 is applied to the p-values of the rows. Experiments whose logs have another form (the population experiments of 9.1, the confidence-weighting and alert-context experiments of 9.8, the calibration of 8.5) have their own pre-declared gates and are not in this family. The cell builds it from the logs and shows the ten rows with the smallest p-values.
"""),
    code(LEDGER9),
    md("""
The family has {{=n_ledger}} rows, {{=n_ledger_3a4}} for CYP3A4 and {{=n_ledger_2d6}} for CYP2D6. {{=len(ledger_met)}} of them meets its gate: {{=ledger_met_text}}. The raw p-value is below 0.05 at $z_0 = 1.0$ for {{=n_below_one}} rows, at 1.96 for {{=n_below_196}}, and at both for {{=n_below_both}}. After Holm's adjustment over the {{=n_ledger}} rows the smallest adjusted p-value is {{=smallest_holm:.2f}}: no row survives the correction.

Several readings of these numbers are possible, and the ledger cannot separate them. If none of the {{=n_ledger}} additions had any effect, about {{=expected_false:.1f}} of them would pass a gate of level 0.05 by chance (at most: the two cut-offs are strongly correlated, so that a pass at both is little rarer than a pass at one); one passed, which is what chance alone gives. If a few of them have small real effects, the ledger would look the same, because the design recognises a difference 8 times in 10 only above {{=floor_main:.4f}} AUC (6.7): the number of changes at $z_0 = 1.0$ above that floor is {{=n_above_floor}}, and the largest is {{=largest_change:+.4f}}. The changes lean positive, and that deserves a remark without a conclusion: at $z_0 = 1.0$ the median change is {{=median_one:+.4f}} and {{=positive_one}} of {{=n_ledger}} are positive; at $z_0 = 1.96$ the median is {{=median_196:+.4f}} and {{=positive_196}} are positive. The additions are not independent tests, since each is added to the same stack and judged on the same folds, so that these counts are not a sign test. A shift of the whole distribution above zero at the cut-off whose scored labels are the most certain is what many additions each carrying a little real information would produce; it can also arise from the dependence among the tests. The ledger does not decide between the two.

**What the gate is for.** The rule of 6.7 was fixed at p < 0.05 at both cut-offs *per addition*, without a correction for the number of additions, and on purpose: a correction over {{=n_ledger}} tries would reject every addition, the real ones included, with a design whose resolution is about 0.01 AUC. The price of the choice is stated here and not hidden. With {{=n_ledger}} tries, about {{=expected_false:.1f}} passes are expected by chance alone, so that a pass is a reason to look closer and not a finding. That is why the one pass of the ledger, the neural pair, was not accepted on its own: it was formed after the results, it had to be confirmed by a replicate declared beforehand (9.9), and it joins the stack on those terms. The adjusted p-values are reported next to the uncorrected ones in the audit (the section on multiplicity, `cypmode/audit/sections_extra.py`, over its own batch of additions) and in the method report.

**What Part IX showed.** Three things. First, an addition that holds information the stack lacks can be seen only if it adds about 0.01 AUC, and of the additions of 9.4 to 9.10 only the neural pair of 9.9 did; a foundation model, outside measurements, the organisers' own extra readings, a new block of chemistry and new blends did not, whatever their origin. The simplest reading, and a reading only, is that 23 members that look at descriptors, fingerprints, neighbours and several learned representations already span what those additions know. Second, the changes that came nearest to the gate were not features at all but models of a different kind, the neural networks of 9.9, which is the only place in the part where the stack gained members that see the molecule in a way the others do not. Third, the largest change in what the project asks of the model is not an addition: it is the selection-aware question of 9.1, which changes what is scored and not how well a score ranks. Part X turns to the quantum-chemical features, whose experiments belong to the same ledger and were designed with the same gates.

"""),
    code(ALL_IN_ONE),
    md("""
**The all-in-one stack.** A gate asks that an addition helps on its own and inside the default stack, and the stack of 8.2 holds only what the gates let in. An idea that is weak alone and useful only together with others could be refused by every gate and still help a stack that holds all of them, so the question whether the gated stack is also the strongest is a fair one, and it was asked once, with the rule written before the result (`cypmode/challenge/all_in_one.py`, queue job `254_all_in_one_comparison`, test `tests/test_all_in_one.py`). The final stack of each enzyme is compared, on the same outer folds, the same population and the same two confidence cut-offs, with one stack of the default candidates and every candidate of every group whose cross-validation predictions are complete ({{=n_all_3a4}} candidates from {{=n_groups_3a4}} groups for CYP3A4), whatever the verdict of the group's gate: one logistic blend each, the paired corrected resampled t. The registered outcome is that the larger stack is better when its mean outer AUC exceeds the final stack's with corrected p < 0.05 at both cut-offs and by at least the design floor; it does not replace the final stack by itself, because it has no replicate (the groups have no predictions on the ten-outer-fold plan), so that a replacement would need that replicate.

The result is in the table. {{=reading_3a4}}; the registered outcome is {{=all_in_one_verdicts['CYP3A4']}}. On CYP2D6 it is {{=all_in_one_verdicts['CYP2D6']}}, {{=reading_2d6}}. A gain of this size is what a stack of ninety candidates would show if each of them added a very little, and it is also what chance and the correlation of the folds would show; with no replicate the design cannot tell the two apart. The final stack is therefore kept, on the rule that was declared, and the larger one is reported as the most notable result of the second looks: it says that the gates were not hiding a large gain in combinations, and that a small one cannot be excluded. A first run of the comparison counted a candidate as complete only if its deployment fit existed too, which left out every group that never joined a final stack (it used 34 candidates from 10 groups, not about ninety); that was a defect of the implementation and not a change of the rule, it was found by reading the log before any decision rested on it, and the log of the first run is kept (`exploratory_all_in_one_first_run_<enzyme>.log`) with its numbers, all of them NOT MET.
"""),
    md("""
*On what actually ran.* `cypmode/challenge/multiplicity.py` (`collect` reads the stacked lines of every log with a gate line, `holm` adjusts), `cypmode/challenge/all_in_one.py` (the second look at the stack), the logs of `data/structure_track/outputs/cyp3a4/logs`, and the audit section that reports the same batch (`cypmode/audit/sections_extra.py`). The notebook reads and counts; it runs no experiment.
"""),
    md("""
## 9.12 Why the ideas that did not join added nothing

> **Field:** Statistics

*Where this section sits:* 9.11 put the logged gates together and corrected for the number of attempts. Its verdict is about the gates: an idea either met its pre-declared rule or did not, and the ledger counts how many did. A verdict of NOT MET says what happened and not why. This section reads the diagnostics that the project generated for the question why: for every group that did not join, the size of its change with the corrected standard error, the redundancy of its predictions with the stack's, the weight the stack gave it, and the smallest effect the design could detect. The diagnostics are produced by code from the caches of the final build (`cypmode/challenge/why_no_gain.py`, queue job 097), so the table and the sentences around it describe the same run.

**Four measures and five descriptions.** For every candidate group and each confidence cut-off, `why_no_gain.py` measures four things with the machinery of the gates themselves: the paired change in the outer AUC when the group joins the default stack, with its corrected p value; the redundancy, the mean and the largest absolute rank correlation between the group's predictions and the stack's on the held-out rows; the weight the stack gave the group when it was allowed to use it, as the share of the absolute logistic coefficients; and the smallest effect the design could detect, the corrected standard error times the sum of the two t quantiles for a one-sided test at the 5 percent level with 80 percent **power**. Five further numbers are descriptive and not gates: a 90 percent **confidence interval** and its equivalence test against a margin of one hundredth of an AUC, Cohen's d_z of the paired differences, the Shapiro–Wilk p value of their normality, the design floor (the same smallest detectable effect with infinitely many repeats), and the partial correlation of the group's predictions with the label given the stack's score. A reason is then assigned by fixed rules in a fixed order, and every clause of the reason carries its number in the text; a group whose change is positive, larger than the detectable effect and still not significant is reported as no reason found. The next cell reads the table and counts.
"""),
    code(WHY_NO_GAIN),
    md("""
The table has {{=n_why}} comparisons of ideas that did not join, {{=n_why_3a4}} on CYP3A4 and {{=n_why_2d6}} on CYP2D6, one row per group and confidence cut-off. In {{=n_equivalent}} of them the 90 percent interval of the change lies inside the margin of one hundredth of an AUC, so the design shows that the idea added nothing that matters; {{=n_inconclusive}} are inconclusive, their interval wider than the margin, which is not the same statement{{=worse_sentence}}. In all {{=n_below_effect}} comparisons the change is smaller than the smallest effect that design could detect: the gate could not have been met by an effect of the size this design can see, whatever the idea's sign. The reason column names the closer causes, and a row can name more than one. Redundancy is the most frequent: {{=n_redundancy}} of the {{=n_why}} comparisons find the group's predictions largely redundant with the stack's, with a median mean absolute rank correlation of {{=median_redundancy:.2f}} among them, which says that the group mostly repeated what the stack already had. The stack gave {{=n_no_weight}} comparisons almost no weight, and {{=n_lowered}} of them lowered the AUC.

The largest changes are listed in the second table. They are those of the attention readout on the Atom-JEPA atoms (9.10) and of the multitask network (9.9), both on CYP3A4, at both confidence cut-offs; {{=n_raw_below}} comparisons in the whole table have a corrected p value below 0.05, and they are the first-cut-off rows of those two. {{=flagged_sentence}}, so the reading is that the design could not separate them from zero. That is also what the ledger of 9.11 says: after the correction over the family no row survives, and the one addition that joined (9.9) did so on a pre-declared replicate and not on this table.

**What the section adds.** The numbers answer the question that 9.11 could not. They do not say that the ideas that did not join are chemically wrong, and nothing in the table could say it: they say that their predictions were close to the stack's and their changes were smaller than the design's resolution, about a hundredth of an AUC at the first cut-off for CYP2D6 and a few thousandths for CYP3A4 (a median design floor of {{=median_floor_3a4:.4f}} for CYP3A4 and {{=median_floor_2d6:.4f}} for CYP2D6). A group whose predictions are redundant with the stack's cannot add much to it, and an effect below the floor cannot be separated from zero by more folds of the same data; only a different representation, or compounds the project has not seen, would settle it. That is the ceiling of 13.3, measured rather than asserted. The diagnostics are descriptive as well as diagnostic: the Shapiro–Wilk p value of the paired differences is below 0.05 in {{=n_shapiro}} of the {{=n_why}} comparisons, so the normal approximation of the corrected t is not exact here, which with so few folds can only flag a gross violation; the module reports it and does not act on it.

*On what actually ran.* The cell reads `data/tdi_track/outputs/final/why_no_gain.csv`, which job 097 writes by calling `cypmode/challenge/why_no_gain.py` on the caches of the final build; the module writes the same counts and the same reasons in the Markdown file `why_no_gain.md` beside it. The notebook reads the table and counts; it runs no experiment.
"""),
]
