# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX (first part): additions that were tested. The tools for reading a gate (9.0), the selection-aware CYP2D6 model (9.1), SAR pair models and matched pairs (9.2), the label-process committee (9.3)."""

from notebooks.notebook_builder import code, figure, md

GATE_TOOLS = '''
# Reading the verdicts. Every experiment of this part writes a log whose lines have the same form (cypmode/challenge/addition_experiment.py): for each confidence cut-off z0, the stacked comparison of the default stack plus the
# addition against the default stack, and for the additions that have a plain counterpart the comparison alone; and a last line, "GATE (...): MET" or "NOT MET". The functions below read them, so that each section can show the
# numbers of its experiment as a table and no number is typed into the text.
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOGS9 = ROOT / "data/structure_track/outputs/cyp3a4/logs"
ALONE_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+alone\\s+(\\S+)\\s+vs (\\S+)\\s*: AUC ([\\d.]+) -> ([\\d.]+) \\(([+-][\\d.]+), p=([\\d.]+), better (\\d+)/(\\d+)\\)")
STACKED_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+stacked base\\+added vs base: AUC ([+-][\\d.]+) \\(p=([\\d.]+), better (\\d+)/(\\d+)\\)")
GATE_LINE = re.compile(r"^(CYP\\w+) GATE \\((alone|stacked)[^)]*\\): (MET|NOT MET)", re.M)
DIAG_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+diagnostic (\\S+): outer AUC ([\\d.]+)")

def read_gate_log(stem, iso):
    """The stacked and the alone comparisons of one log and its verdicts; None when the log does not exist or holds no comparison."""
    path = LOGS9 / f"{stem}_{iso}.log"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    stacked = [{"z0": float(z0), "gain": float(g), "p": float(p), "better": f"{b} of {n}"} for z0, g, p, b, n in STACKED_LINE.findall(text)]
    alone = [{"z0": float(z0), "addition": a, "against": b, "AUC without": float(w), "AUC with": float(v), "gain": float(d), "p": float(p), "better": f"{k} of {n}"} for z0, a, b, w, v, d, p, k, n in ALONE_LINE.findall(text)]
    verdicts = {kind: verdict == "MET" for found_iso, kind, verdict in GATE_LINE.findall(text) if found_iso == iso}
    diagnostic = [{"z0": float(z0), "candidate": name, "outer AUC alone": float(v)} for z0, name, v in DIAG_LINE.findall(text)]
    if not stacked and not alone and not verdicts:
        return None
    return {"stacked": pd.DataFrame(stacked), "alone": pd.DataFrame(alone), "verdicts": verdicts, "diagnostic": pd.DataFrame(diagnostic)}

def stacked_summary(stem, iso):
    """One line for a ledger: the stacked gain and p at each cut-off and the verdict."""
    result = read_gate_log(stem, iso)
    if result is None or result["stacked"].empty:
        return {"gain at z0 1.0": np.nan, "p at z0 1.0": np.nan, "gain at z0 1.96": np.nan, "p at z0 1.96": np.nan, "gate": "not run" if result is None else "no stacked line"}
    table = result["stacked"].set_index("z0")
    row = {}
    for z0 in (1.0, 1.96):
        row[f"gain at z0 {z0}"] = float(table.loc[z0, "gain"]) if z0 in table.index else np.nan
        row[f"p at z0 {z0}"] = float(table.loc[z0, "p"]) if z0 in table.index else np.nan
    row["gate"] = "MET" if result["verdicts"].get("stacked") else "NOT MET"
    return row
'''

POPULATION_FACTS = '''
# The population problem of CYP2D6. The training file's TDI labels exist only for compounds that were *promoted to dose-response* on the enzyme (the screening hits). A part of the training file, found from the data by
# cypmode/challenge/library_blocks.py, is a diversity-library *panel* that was run on all four enzymes; inside it, "no CYP2D6 measurement" means "screened and not promoted", a compound that the challenge scores as inactive
# ("assigned negative", section 2.3). Outside the panel "no measurement" only means "never tested". The cell counts what that does to the positive rate: among the hits alone, among the panel population (hits and non-hits), and
# in the scored set, whose rate was recovered from the public boards (section 2.10).
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.library_blocks import panel_block_mask
from cypmode.challenge.nested import NestedPlan

train_file = pd.read_csv(ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
panel_mask = panel_block_mask(train_file)
d6_labelled = train_file["CYP2D6_is_TDI"].notna()
pop_plan = NestedPlan.load(ROOT / "data/tdi_track/outputs/nested/CYP2D6_pop_panel/plan.npz")
pop_labels = pop_plan.labels(pop_plan.pool)
d6_prevalence = float(estimate_from_csv(ROOT / "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP2D6_2026-10-04.csv")["prevalence"])
facts_pop = [
    {"quantity": "compounds of the training file in the panel block", "value": int(panel_mask.sum())},
    {"quantity": "CYP2D6-labelled compounds (the hits)", "value": int(d6_labelled.sum())},
    {"quantity": "of which in the panel block", "value": int((panel_mask & d6_labelled).sum())},
    {"quantity": "positive rate among the hits", "value": float(train_file["CYP2D6_is_TDI"].mean())},
    {"quantity": "compounds of the population pool (hits and the screened non-hits of the panel)", "value": int(len(pop_plan.pool))},
    {"quantity": "positive rate of the population pool", "value": float(pop_labels.mean())},
    {"quantity": "share of the pool that are hits", "value": float(d6_labelled.loc[train_file.index[panel_mask & d6_labelled.to_numpy()]].sum() / len(pop_plan.pool))},
    {"quantity": "positive rate of the scored set (recovered from the boards)", "value": d6_prevalence},
]
show_table(pd.DataFrame(facts_pop), floatfmt=".3f")
pop_values = {row["quantity"]: row["value"] for row in facts_pop}
'''

HECKMAN_TOY = '''
# The selection problem, in a toy where the truth is known (Heckman, 1979). A quantity y = 1.0 x + e depends on a feature x. It is observed only for compounds that were *selected*, and selection depends on the same feature and
# on a hidden factor u that is correlated with the noise e of y: s = 1.0 x + u, observed when s > 0. A regression of y on x in the selected compounds is biased, because selection keeps the compounds whose hidden factor happened to
# be high, and so those whose e is high. The two-step correction adds the *inverse Mills ratio* of the selection model, lambda(z) = phi(z) / Phi(z) (phi the density and Phi the cumulative distribution function of the standard normal), where
# z is the fitted index of a probit regression of selection on x, as an extra regressor; its coefficient absorbs the part of e that selection has distorted.
import statsmodels.api as sm
from scipy.stats import norm

def heckman_toy(seed, n=20000, rho=0.7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    u, e0 = rng.normal(size=n), rng.normal(size=n)
    e = rho * u + np.sqrt(1 - rho**2) * e0                                   # the noise of y, correlated with the hidden selection factor u
    selected = (1.0 * x + u) > 0
    y = 1.0 * x + e
    naive = sm.OLS(y[selected], sm.add_constant(x[selected])).fit().params[1]
    z = sm.Probit(selected.astype(int), sm.add_constant(x)).fit(disp=0).fittedvalues
    mills = norm.pdf(z) / norm.cdf(z)
    corrected = sm.OLS(y[selected], np.column_stack([np.ones(selected.sum()), x[selected], mills[selected]])).fit().params[1]
    everyone = sm.OLS(y, sm.add_constant(x)).fit().params[1]
    return naive, corrected, everyone
heckman_runs = np.array([heckman_toy(seed) for seed in range(20)])
heckman_table = pd.DataFrame({"estimate of the coefficient of x (true value 1.0)": ["regression on the selected compounds", "the same with the Mills ratio added", "regression on everyone (not available in practice)"],
                              "mean over 20 data sets": heckman_runs.mean(axis=0), "sd": heckman_runs.std(axis=0, ddof=1)})
show_table(heckman_table, floatfmt=".3f")
heckman_naive, heckman_corrected = float(heckman_runs[:, 0].mean()), float(heckman_runs[:, 1].mean())
'''

ADJUSTED_COUNT = '''
# Estimating the share of hits in an unlabelled set. The gate model (a LightGBM that predicts "was promoted on this enzyme" from structure) gives every compound a probability. Suppose the panel is a mixture of hits and non-hits,
# and a test set is a different mixture of the same two kinds. The mean gate probability of a set is the mixture of the mean probability of hits and the mean probability of non-hits, with the share of hits as weight, so
# share(test) = (mean P(test) - mean P(non-hits)) / (mean P(hits) - mean P(non-hits)) ("probabilistic adjusted count"). Classify-and-count, the share of the set above a threshold, is biased by the false-positive and false-negative rates
# of the gate. A toy with a known answer: a gate whose output for hits is a Beta(4, 3) variable and for non-hits a Beta(2, 4), applied to sets of 750 compounds with true hit shares between 0.1 and 0.9.
rng_ac = np.random.default_rng(4)
p_hits_ref, p_non_ref = rng_ac.beta(4, 3, 20000), rng_ac.beta(2, 4, 20000)
adjusted_rows = []
for true_share in (0.1, 0.3, 0.5, 0.7, 0.9):
    estimates, counts = [], []
    for _ in range(200):
        is_hit = rng_ac.random(750) < true_share
        p_set = np.where(is_hit, rng_ac.beta(4, 3, 750), rng_ac.beta(2, 4, 750))
        estimates.append((p_set.mean() - p_non_ref.mean()) / (p_hits_ref.mean() - p_non_ref.mean()))
        counts.append((p_set >= 0.5).mean())
    adjusted_rows.append({"true share of hits": true_share, "adjusted count, mean": float(np.mean(estimates)), "adjusted count, sd": float(np.std(estimates)), "classify and count at 0.5, mean": float(np.mean(counts))})
adjusted_table = pd.DataFrame(adjusted_rows)
show_table(adjusted_table, floatfmt=".3f")
adjusted_error = float(np.abs(adjusted_table["adjusted count, mean"] - adjusted_table["true share of hits"]).max())
count_error = float(np.abs(adjusted_table["classify and count at 0.5, mean"] - adjusted_table["true share of hits"]).max())
'''

SELECTION_RESULTS = '''
# What the selection-aware stacks did, from the logs of the experiments (population_experiment.py for v1, selection_aware_v2.py for v2, test_nonhit_share.py for the share, hedged_blend.py for the blend). The population
# problem is evaluated on the panel pool with the same nested design, the same yardstick (the non-hits are unambiguous and keep their full weight) and the same corrected t.
SETLINE = re.compile(r"(S[0-3]) ([^\\n]*?)\\s+outer AUC ([\\d.]+)\\s+outer MCC at transferred rate ([+-][\\d.]+)(?: \\| vs S0 AUC ([+-][\\d.]+) \\(p=([\\d.]+), better (\\d+)/(\\d+)\\))?")
population_text = (LOGS9 / "population_panel_CYP2D6_r3.log").read_text(errors="ignore")
blocks = re.split(r"evaluation: similarity x prevalence x confident rows \\(z0 = ", population_text)[1:]
set_rows = []
for block in blocks:
    z0 = float(block.split(")")[0])
    for name, label, auc, mcc, gain, p, better, n in SETLINE.findall(block):
        set_rows.append({"z0": z0, "set": f"{name} {label.strip()}", "outer AUC": float(auc), "MCC at the transferred rate": float(mcc), "gain over S0": float(gain) if gain else np.nan, "p (corrected t)": float(p) if p else np.nan, "folds better": f"{better} of {n}" if better else ""})
set_table = pd.DataFrame(set_rows)
show_table(set_table, floatfmt=".4f")
v2_text = (LOGS9 / "selection_aware_v2_CYP2D6.log").read_text(errors="ignore")
v2_gain = [(float(z0), float(g), float(p)) for z0, g, p in re.findall(r"z0 = ([\\d.]+)\\n(?:.*\\n)*?\\s+V2 vs V1: AUC ([+-][\\d.]+) \\(p=([\\d.]+)", v2_text)]
share_text = (LOGS9 / "test_nonhit_share_CYP2D6.log").read_text(errors="ignore")
share_hit = float(re.search(r"hit share of the test set ([\\d.]+) \\(95% bootstrap ([\\d.]+)-([\\d.]+)\\)", share_text).group(1))
share_interval = tuple(float(v) for v in re.search(r"\\(95% bootstrap ([\\d.]+)-([\\d.]+)\\)", share_text).groups())
panel_share = float(re.search(r"panel ([\\d.]+)\\n", share_text).group(1))
equivalent_weight = float(re.search(r"equivalent non-hit weight on the job-012 axis: ([\\d.]+) \\(95%: ([\\d.]+)-([\\d.]+)\\)", share_text).group(1))
equivalent_interval = tuple(float(v) for v in re.search(r"equivalent non-hit weight on the job-012 axis: [\\d.]+ \\(95%: ([\\d.]+)-([\\d.]+)\\)", share_text).groups())
s0_row, s3_row = set_table[(set_table["z0"] == 1.0) & (set_table["set"].str.startswith("S0"))].iloc[0], set_table[(set_table["z0"] == 1.0) & (set_table["set"].str.startswith("S3"))].iloc[0]
'''

HEDGED = '''
# The regime problem. The selection-aware stack wins only when the scored set holds enough never-promoted compounds, and the share is only estimated. hedged_blend.py fits the stacks under one assumed regime and scores them under
# every true regime of a grid (the non-hit weight w, 1 = the training share). The table of the log for the fit weight that the build uses (1.0) is read, at z0 = 1.0, for the three models: the hits-only stack, the selection-aware stack
# and the rank blend of the two.
hedged_text = (LOGS9 / "hedged_blend_CYP2D6.log").read_text(errors="ignore")
final_block = hedged_text.split("stacks fitted under non-hit weight 1.0")[1]
first_grid = final_block.split("z0 = 1.0")[1].split("z0 = 1.96")[0]
weights_axis = [float(w) for w in re.findall(r"w=([\\d.]+)", first_grid.splitlines()[0])]
regime_rows = []
for line in first_grid.splitlines():
    match = re.match(r"\\s+(hits_only|selection_aware|blend)\\s+(.*)", line)
    if match:
        values = [float(v) for v in re.findall(r"([\\d.]+)\\|", match.group(2))]
        regime_rows.append({"model": match.group(1), **{f"w={w:g}": v for w, v in zip(weights_axis, values)}})
regime_table = pd.DataFrame(regime_rows)
show_table(regime_table, floatfmt=".3f")
fig, axis = plt.subplots(figsize=(7.2, 3.8))
for model, colour, label in (("hits_only", theme["blue"], "hits-only stack"), ("selection_aware", theme["orange"], "selection-aware stack"), ("blend", theme["green"], "rank blend of the two")):
    row = regime_table[regime_table["model"] == model].iloc[0]
    axis.plot(weights_axis, [row[f"w={w:g}"] for w in weights_axis], "o-", color=colour, lw=2.2, ms=5, label=label)
axis.axvspan(0.4, 0.68, color=theme["muted"], alpha=0.18, lw=0)
axis.set(xlabel="true weight of the never-promoted compounds in the scored set (1 = the training share)", ylabel="mean outer AUC (z0 = 1.0)", title="CYP2D6: which stack wins depends on the scored set")
axis.legend(frameon=False, fontsize=8, loc="upper left")
tidy(axis, theme)
fig.tight_layout()
hits_row, sel_row = regime_table[regime_table["model"] == "hits_only"].iloc[0], regime_table[regime_table["model"] == "selection_aware"].iloc[0]
crossover = next((w for w in weights_axis if sel_row[f"w={w:g}"] > hits_row[f"w={w:g}"]), None)
before_crossover = weights_axis[weights_axis.index(crossover) - 1] if crossover is not None and weights_axis.index(crossover) > 0 else 0.0
hedged_gates = dict(re.findall(r"CYP2D6 GATE \\((hedged blend adopted by the minimax rule|selection-aware replaces hits-only)[^)]*fit weight 1\\.0\\): (MET|NOT MET)", hedged_text))
'''



PAIR_DEMO = '''
# The pair model on one outer fold of CYP3A4 (cypmode/challenge/pair_transfer.py). A nearest-neighbour model says that a compound resembles its neighbours' labels; it cannot say which *differences* from a neighbour matter. The pair model
# forms, for each compound q and each of its K = 5 most similar labelled compounds n, a row [descriptors of q, descriptors of q minus those of n, Tanimoto(q, n), label of n, rank of n] and learns the label of q from it; a new compound's
# five pair predictions are averaged with the similarity as weight. The cell builds the pair rows of one validation compound, fits the model on the outer-training compounds of the first fold, and compares it with the plain LightGBM on
# the descriptors and with the nearest-neighbour average, on the 467 validation compounds.
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from cypmode.challenge.pair_transfer import PAIR_K, fit_predict_pairs, neighbour_table, pair_matrix

bits_pair = store8["morgan_bits"].astype(np.float32)
descriptors_pair = store8["rdkit_desc"].astype(np.float32)
refit_pair = plan8.task(*plan8.outer_folds()[0], OUTER_REFIT)
rows_fit, rows_eval = refit_pair.train, refit_pair.evaluate
overlap = bits_pair[rows_eval] @ bits_pair[rows_fit].T
similarity_et = overlap / (bits_pair[rows_eval].sum(1)[:, None] + bits_pair[rows_fit].sum(1)[None, :] - overlap)
overlap_tt = bits_pair[rows_fit] @ bits_pair[rows_fit].T
similarity_tt = overlap_tt / (bits_pair[rows_fit].sum(1)[:, None] + bits_pair[rows_fit].sum(1)[None, :] - overlap_tt)
y_fit, y_eval = plan8.labels(rows_fit), plan8.labels(rows_eval)
x_fit, x_eval = descriptors_pair[rows_fit], descriptors_pair[rows_eval]
top_e, sims_e = neighbour_table(similarity_et, np.arange(len(rows_eval)), np.arange(len(rows_fit)), PAIR_K, False)
example_rows = pair_matrix(x_eval[:1], x_fit, y_fit, top_e[:1], sims_e[:1])
n_descriptors_pair = x_fit.shape[1]
pair_prediction = fit_predict_pairs(x_fit, y_fit, x_eval, similarity_tt, similarity_et, PAIR_K, {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 31, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 10}, 4, 0)
plain_prediction = LGBMClassifier(verbose=-1, n_estimators=300, learning_rate=0.04, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, min_child_samples=10, random_state=0).fit(x_fit, y_fit).predict_proba(x_eval)[:, 1]
neighbour_vote = (np.maximum(sims_e, 1e-6) * y_fit[top_e]).sum(1) / np.maximum(sims_e, 1e-6).sum(1)
typical = int(np.argsort(sims_e.mean(axis=1))[len(rows_eval) // 2])                      # the validation compound whose mean neighbour similarity is the median one
example_rows = pair_matrix(x_eval[typical:typical + 1], x_fit, y_fit, top_e[typical:typical + 1], sims_e[typical:typical + 1])
pair_view = pd.DataFrame({"neighbour": np.arange(1, PAIR_K + 1), "Tanimoto similarity": sims_e[typical], "label of the neighbour": y_fit[top_e[typical]]})
show_table(pair_view, floatfmt=".2f")
pair_compare = pd.DataFrame([{"model": "plain LightGBM on the descriptors", "AUC on the validation compounds": roc_auc_score(y_eval, plain_prediction)}, {"model": "similarity-weighted vote of the 5 neighbours", "AUC on the validation compounds": roc_auc_score(y_eval, neighbour_vote)},
                             {"model": "pair model (5 pair rows per compound)", "AUC on the validation compounds": roc_auc_score(y_eval, pair_prediction)}])
show_table(pair_compare, floatfmt=".3f")
pair_columns, pair_rows_fit = int(example_rows.shape[1]), int(len(y_fit) * PAIR_K)
example_values = {"truth": int(y_eval[typical]), "pair model": float(pair_prediction[typical]), "plain model": float(plain_prediction[typical]), "vote": float(neighbour_vote[typical]), "similarity": float(sims_e[typical].mean())}
'''

MMP_DEMO = '''
# Matched molecular pairs (cypmode/challenge/mmp_transfer.py). Two compounds form a matched pair when cutting a single bond of each (RDKit's rdMMPA) leaves the same large *constant* part and different small *variable* parts of 1 to 10
# heavy atoms; the transformation is written "variable of n >> variable of q". For a query q, the model looks at the training compounds n that form a pair with it and asks how that same transformation moved the label across all other
# training pairs: delta_T = sum over the pairs a -> b with transformation T of (y_b - y_a) / (count_T + 3), the shrinkage by 3 pulling a transformation seen few times towards "no change". The pairs are label-free and cached; the cell reads
# them for the CYP3A4 pool, counts them, lists the best-supported transformations with the label change they carry, and shows one worked example.
from cypmode.challenge.mmp_transfer import SHRINK, get_pairs, mmp_label_features, transformation_report

all_smiles = [str(v) for v in store8["smiles"]]
pairs = get_pairs(all_smiles)
pool_rows, pool_labels = plan8.pool, plan8.labels(plan8.pool)
in_pool = np.zeros(len(all_smiles), dtype=bool)
in_pool[pool_rows] = True
own_features = mmp_label_features(pairs, len(all_smiles), pool_rows, pool_labels, pool_rows, True)
test_features = mmp_label_features(pairs, len(all_smiles), pool_rows, pool_labels, plan8.external, False)
report = transformation_report(pairs, pool_rows, pool_labels, top=8, min_support=2)
report_table = pd.DataFrame(report, columns=["transformation (variable of n >> variable of q)", "training pairs", "shrunk mean change of the label"])
show_table(report_table, floatfmt=".3f")
a_idx, b_idx, t_idx, names_mmp = pairs
best_name = report[0][0]
best_id = names_mmp.index(best_name)
example_pair = next((int(a), int(b)) for a, b, t in zip(a_idx, b_idx, t_idx) if t == best_id and in_pool[a] and in_pool[b])
label_of = dict(zip(pool_rows, pool_labels))
within_pool = in_pool[a_idx] & in_pool[b_idx]
support = np.bincount(t_idx[within_pool], minlength=len(names_mmp))
mmp_facts = {"directed pairs among all the compounds": len(a_idx), "directed pairs inside the CYP3A4 pool": int(within_pool.sum()), "distinct transformations among all the compounds": len(names_mmp),
             "pool compounds with a matched partner in the pool": float(np.mean(own_features[:, 0] > 0)), "blinded compounds with a matched partner in the pool": float(np.mean(test_features[:, 0] > 0)),
             "transformations seen in 2 or more pool pairs": int((support >= 2).sum()), "transformations seen in 3 or more pool pairs": int((support >= 3).sum()), "pairs of the most frequent transformation": int(support.max())}
show_table(pd.DataFrame({"quantity": list(mmp_facts), "value": list(mmp_facts.values())}), floatfmt=".3f")
print(f"worked example, transformation {best_name}: {all_smiles[example_pair[0]]} (label {label_of[example_pair[0]]}) -> {all_smiles[example_pair[1]]} (label {label_of[example_pair[1]]})")
'''

PAIR_RESULTS = '''
# The gate results of the two ideas, for both enzymes: the pair model with the descriptors in the delta block (pair_desc), with the xTB physics block in the delta (pair_physics), and the matched-pair model (mmp), each
# added to the default stack. The "alone" AUC is the diagnostic of the new candidate on its own.
sar_rows = []
for name, stem in (("pair model, descriptor deltas", "pair_transfer"), ("pair model, physics deltas", "pair_transfer_physics"), ("matched-pair transformations", "mmp_transfer")):
    for iso in ("CYP3A4", "CYP2D6"):
        row = stacked_summary(stem, iso)
        result = read_gate_log(stem, iso)
        diag = result["diagnostic"] if result is not None and "diagnostic" in result and len(result["diagnostic"]) else None
        best_alone = float(diag[diag["z0"] == 1.0]["outer AUC alone"].max()) if diag is not None and (diag["z0"] == 1.0).any() else np.nan
        sar_rows.append({"idea": name, "enzyme": iso, "best candidate alone (AUC, z0 1.0)": best_alone, **row})
sar_table = pd.DataFrame(sar_rows)
show_table(sar_table, floatfmt=".4f")
sar_largest = float(sar_table[["gain at z0 1.0", "gain at z0 1.96"]].abs().max().max())
sar_smallest_p = float(sar_table[["p at z0 1.0", "p at z0 1.96"]].min().min())
n_sar_met = int((sar_table["gate"] == "MET").sum())
'''



COMMITTEE_MATH = '''
# The label as a threshold of a noisy number, in numbers. The label is positive when the shift s = t - max(d, 4) exceeds tau = log10 2 (section 2.3). A model that predicts s with a Gaussian error, s ~ Normal(m, sigma^2), knows
# more than the yes/no label: the probability of a positive is P(s > tau) = Phi((m - tau) / sigma), and the *expected improvement* of s over the threshold, the expected size of the exceedance E[max(s - tau, 0)], is
# sigma (phi(z) + z Phi(z)) with z = (m - tau) / sigma (phi and Phi the density and the cumulative distribution function of the standard normal). A compound can have the same P and a different EI: a sure bet barely above the
# threshold and a long shot far above it. The cell evaluates the formulas for three compounds and checks them by simulation.
from scipy.stats import norm

from cypmode.challenge.tdi_targets import SHIFT_THRESHOLD

tau = float(SHIFT_THRESHOLD)
toy_compounds = pd.DataFrame({"compound": ["sure, barely above", "coin flip", "long shot, far above if it is"], "predicted shift m": [0.45, 0.30, 0.10], "sigma": [0.10, 0.20, 0.70]})
z_toy = (toy_compounds["predicted shift m"] - tau) / toy_compounds["sigma"]
toy_compounds["P(label)"] = norm.cdf(z_toy)
toy_compounds["expected improvement"] = toy_compounds["sigma"] * (norm.pdf(z_toy) + z_toy * norm.cdf(z_toy))
rng_c = np.random.default_rng(0)
simulated = rng_c.normal(toy_compounds["predicted shift m"].to_numpy()[:, None], toy_compounds["sigma"].to_numpy()[:, None], size=(3, 400000))
toy_compounds["P by simulation"] = (simulated > tau).mean(axis=1)
toy_compounds["EI by simulation"] = np.maximum(simulated - tau, 0).mean(axis=1)
show_table(toy_compounds, floatfmt=".4f")
formula_check = float(max(np.abs(toy_compounds["P(label)"] - toy_compounds["P by simulation"]).max(), np.abs(toy_compounds["expected improvement"] - toy_compounds["EI by simulation"]).max()))
'''

COMMITTEE_RESULTS = '''
# The committee in the nested design, from the logs. The candidates are committee_m (the committee's mean shift), committee_p (P(label)) and committee_ei; the first is there to separate what the uncertainty adds from what the
# regression alone gives. Gate 1: the stack with the three candidates beats the default stack with corrected p < 0.05 at both cut-offs. Gate 2: committee_p must be calibrated as it comes out, with no post-hoc map
# (ECE at most 0.05, slope between 0.8 and 1.2, the bar of 8.5).
CAL_LINE = re.compile(r"z0=(\\d\\.\\d+)\\s+ECE ([\\d.]+)\\s+Brier ([\\d.]+)\\s+slope ([\\d.]+)\\s+mean predicted ([\\d.]+) vs observed ([\\d.]+)\\s+-> calibration gate (MET|NOT MET)")
committee_rows, committee_cal = [], []
for iso in ("CYP3A4", "CYP2D6"):
    summary = stacked_summary("committee_label_model", iso)
    result = read_gate_log("committee_label_model", iso)
    diag = result["diagnostic"] if result is not None else pd.DataFrame()
    alone_auc = {z: float(diag[diag["z0"] == z]["outer AUC alone"].max()) for z in (1.0, 1.96)} if len(diag) else {1.0: np.nan, 1.96: np.nan}
    committee_rows.append({"enzyme": iso, "best member alone, AUC at z0 1.0": alone_auc[1.0], **summary})
    text = (LOGS9 / f"committee_label_model_{iso}.log").read_text(errors="ignore") if (LOGS9 / f"committee_label_model_{iso}.log").exists() else ""
    for z0, ece, brier, slope, predicted, observed, verdict in CAL_LINE.findall(text):
        committee_cal.append({"enzyme": iso, "z0": float(z0), "ECE": float(ece), "slope": float(slope), "mean predicted": float(predicted), "mean observed": float(observed), "calibration gate": verdict})
committee_table, committee_cal_table = pd.DataFrame(committee_rows), pd.DataFrame(committee_cal)
show_table(committee_table, floatfmt=".4f")
show_table(committee_cal_table, floatfmt=".3f")
committee_3a4 = committee_table[committee_table["enzyme"] == "CYP3A4"].iloc[0]
'''


CELLS = [
    md("""
# Part IX. Additions that were tested

Part VIII described the default stack. This part describes what was proposed to be added to it, and what happened to each proposal. Every section has the same parts: the idea and where it came from, the exact construction with every formula built up, the gate written down before the result existed (6.7), the result alone and inside the stack with its corrected p-values, and the reason the idea did or did not help. The ideas are of two kinds. Some change *what the model is asked to predict* (the population of the scored set, the hidden structure of the label, the noise of the measurements); some change *what the model is given* (the structure of the neighbours, a pretrained representation, the other enzymes' measurements). None of them is a variation of a hyperparameter: each is a hypothesis about the data with a mechanism behind it.

The sections are in the order in which the ideas arose. Section 9.1 is the longest, because its idea has the longest chain of reasoning and because it is the one the numbers alone cannot decide. Sections 9.2 to 9.8 each take one idea from a family; 9.9 the neural networks of the last queue; 9.10 the other late additions; 9.11 the ledger of all the gates, read from the logs.
"""),
    md("""
## 9.1 The selection-aware CYP2D6 model: when the training labels come from a filtered population

> **Field:** Statistics · Machine learning

*Where this section sits:* the stack of Part VIII is trained and scored on the compounds that have a TDI label. This section asks whether those compounds are the population the challenge scores, finds that for CYP2D6 they are not, builds a model for the population that is, and then meets the difficulty that decides the section: how much of the scored population is of the other kind cannot be measured, only estimated. The estimate, the model and the decision it leaves open are all described here.

**The observation.** The screen that produced the training file tested compounds at one concentration first, and *promoted* those that looked active on an enzyme to a full dose-response measurement. The TDI label (section 2.3) exists only for a compound that was promoted on that enzyme; a compound that was screened and not promoted has no TDI measurement. The scored set was built differently: it is a full dose-response panel of analogues of hits found on *other* enzymes, and the challenge scores a compound that was measured as inactive as "assigned negative". So the CYP2D6 compounds of the scored set are mostly of a kind that the training labels do not contain: *non-hits*. For CYP3A4 the scored set is made of analogues of CYP3A4 hits, the labelled compounds are the right population, and CYP3A4 serves as a control on which no gain is expected. The first cell counts what the difference does to the positive rate.
"""),
    code(GATE_TOOLS),
    code(POPULATION_FACTS),
    md("""
The panel block of the training file holds {{=pop_values['compounds of the training file in the panel block']:,}} compounds, and {{=pop_values['of which in the panel block']:,}} of the {{=pop_values['CYP2D6-labelled compounds (the hits)']:,}} CYP2D6-labelled compounds are in it. Among the hits {{=pop_values['positive rate among the hits']:.1%}} are positive. If the screened non-hits of the panel are added as negatives, the population pool of {{=pop_values['compounds of the population pool (hits and the screened non-hits of the panel)']:,}} compounds has a positive rate of {{=pop_values['positive rate of the population pool']:.1%}} and the hits are {{=pop_values['share of the pool that are hits']:.1%}} of it; the positive rate of the scored set, recovered from the public boards (section 2.10), is {{=pop_values['positive rate of the scored set (recovered from the boards)']:.1%}}. The scored set looks like the population pool and not like the hits. A model trained on the hits and applied to the pool has to rank hits and non-hits together, which it has never been asked to do.

**The model.** The probability that a compound of the scored set is positive factors into two probabilities,

$$P(y = 1) \\;=\\; P(\\text{hit}) \\cdot P(y = 1 \\mid \\text{hit}),$$

where $y = 1$ means that the compound is a time-dependent inhibitor and "hit" that it was promoted on CYP2D6; a non-hit is negative by the rule of 2.3, so $P(y=1 \\mid \\text{non-hit}) = 0$. The second factor is the model of Part VIII, trained on the hits. The first, the *gate*, is a LightGBM trained on *all* the panel compounds on the question "was this compound promoted on CYP2D6?", from structure alone. Their product ranks the whole population. This is `lgbm_gate`, `lgbm_hits_only` and `lgbm_two_part` in `cypmode/challenge/nested_runners.py`, and it is the first selection-aware model (v1). A second version (v2, `cypmode/challenge/selection_aware_v2.py`) gives the gate more to see: the screen promoted compounds on four enzymes, promiscuity is shared among them, and whether a compound was promoted on CYP1A2, CYP2C9 and CYP3A4 says much about whether it was promoted on CYP2D6. A first-stage LightGBM per enzyme predicts "promoted on enzyme $k$", its out-of-fold log-odds (cross-fitted, so that no compound is scored by a model that saw its own pattern) feed a logistic second stage for CYP2D6, and only the structure of the evaluated compound enters its gate; the hit pattern of an evaluated compound is never read.

**The classical remedy for a filtered sample.** Selection into a sample is a well-studied problem in econometrics. Heckman (1979) showed that when a quantity is observed only for selected units, and the selection depends on a hidden factor that is correlated with the quantity's noise, the regression on the selected units is biased, and that adding a correction term, the *inverse Mills ratio* of the selection model, removes the bias. The toy below makes it concrete: a quantity $y = x + e$ is observed only when the selection index $s = x + u > 0$, with the hidden factor $u$ correlated with the noise $e$ (correlation 0.7). The inverse Mills ratio is $\\lambda(z) = \\varphi(z)/\\Phi(z)$, with $\\varphi$ and $\\Phi$ the density and the cumulative distribution function of the standard normal, evaluated at the fitted index $z$ of a probit regression (a regression whose link is $\\Phi$) of selection on $x$.
"""),
    code(HECKMAN_TOY),
    md("""
The true coefficient of $x$ is 1. The regression on the selected units gives {{=heckman_naive:.3f}}, biased downwards because selection kept the units whose hidden factor, and so whose noise, happened to be high for low $x$ and not for high $x$; with the Mills ratio added it gives {{=heckman_corrected:.3f}}, as the regression on everyone would. The project's `sel2_heckman` candidate puts the probit score $z = \\Phi^{-1}(\\hat p_{\\text{gate}})$ and the ratio $\\lambda(z)$ of the gate into the feature set of the hits-model, the classical correction applied to a classifier.

**The regime problem.** The selection-aware model beats the hits-only one only if the scored set holds enough non-hits, and nobody has the labels of the scored set. Two estimates exist. The first uses the public boards: the official baseline is a model trained on hits, its accuracy, precision and recall on the scored set are public, and how it behaves on a population depends on the mix; re-weighting the out-of-fold predictions of the training compounds so that the baseline's behaviour matches its published numbers gives a band of plausible mixes (`cypmode/challenge/population_share.py`, a chi-square region of two weights). The second estimates the share of hits in the scored set from the structures alone, by the *probabilistic adjusted count* (Forman's adjusted count, applied to the mean of a classifier's probabilities): the mean gate probability of a set is a mixture of the mean over hits and the mean over non-hits, with the share of hits as the weight, so that
$$\\text{share of hits in the set} \\;=\\; \\frac{\\overline{P}(\\text{set}) - \\overline{P}(\\text{non-hits})}{\\overline{P}(\\text{hits}) - \\overline{P}(\\text{non-hits})},$$
where $\\overline{P}$ is the mean gate probability over the set named, the two reference means being those of the panel compounds' out-of-fold probabilities. Counting the compounds above a threshold instead ("classify and count") is biased by the gate's error rates. A toy with a known answer shows both.
"""),
    code(ADJUSTED_COUNT),
    md("""
In the toy the adjusted count recovers the true share to within {{=adjusted_error:.3f}} for every share from 0.1 to 0.9, and classify-and-count is off by as much as {{=count_error:.2f}}, over-counting when hits are rare and under-counting when they are common. The assumption that makes the estimator valid is that the structures of hits (and of non-hits) have the same distribution in the test set and in the panel, and analogue expansion violates it, so the real estimate has a stated bias and no more. The cell below reads the results of the experiments from their logs: the population experiment (the hits-only set S0 and the selection-aware sets S1 to S3, in the nested design, with the yardstick of 6.4), the v2 comparison and the label-shift estimate.
"""),
    code(SELECTION_RESULTS),
    md("""
Evaluated on the population, the hits-only stack scores an AUC of {{=s0_row['outer AUC']:.3f}} at $z_0 = 1.0$, hardly above chance, because it has never seen a non-hit. Adding the gate lifts it to {{=set_table[(set_table['z0'] == 1.0) & (set_table['set'].str.startswith('S1'))]['outer AUC'].iloc[0]:.3f}} (corrected p = {{=set_table[(set_table['z0'] == 1.0) & (set_table['set'].str.startswith('S1'))]['p (corrected t)'].iloc[0]:.3f}}), the two-part product to {{=set_table[(set_table['z0'] == 1.0) & (set_table['set'].str.startswith('S2'))]['outer AUC'].iloc[0]:.3f}} and the end-to-end population classifier to {{=s3_row['outer AUC']:.3f}}, with the MCC at the transferred rate going from {{=s0_row['MCC at the transferred rate']:.3f}} to {{=s3_row['MCC at the transferred rate']:.3f}}; at the stricter cut-off the gains are larger and every p-value is below 0.01. On the population the selection-aware stack is a different model class from the hits-only one. The v2 gate adds nothing to v1: the log reports {{=', '.join(f'{g:+.4f} (p = {p:.2f}) at z0 = {z}' for z, g, p in v2_gain)}} for V2 against V1, and the gate of v2 is NOT MET.

The label-shift estimate says where the scored set is on the axis of non-hit weight, 1 being the training share. The hit share of the 750 test structures is estimated at {{=share_hit:.3f}} (95 percent interval {{=share_interval[0]:.3f}} to {{=share_interval[1]:.3f}}) against {{=panel_share:.3f}} in the panel, which is equivalent to a non-hit weight of {{=equivalent_weight:.2f}} (interval {{=equivalent_interval[0]:.2f}} to {{=equivalent_interval[1]:.2f}}). The scored set therefore holds fewer non-hits than the training population, about half as many in relative terms, and many more than a hits-only world. What each stack earns across the whole axis, when it is fitted under the assumption that the scored set has the training share and then scored under every true weight, is in the next cell.
"""),
    code(HEDGED),
    md("""
The picture of the figure is the heart of the section. The hits-only stack is best when the scored set has no non-hits (AUC {{=hits_row['w=0']:.3f}} at $w = 0$, against {{=sel_row['w=0']:.3f}} for the selection-aware one), and it loses steadily as non-hits are added ({{=hits_row['w=1']:.3f}} at $w = 1$). The selection-aware stack is the reverse, and the two cross between the weights {{=before_crossover:g}} and {{=crossover:g}}. Inside the plausible band (the shaded range 0.4 to 0.68, where the label-shift estimate of {{=equivalent_weight:.2f}} sits) the selection-aware stack scores between {{=sel_row['w=0.4']:.3f}} and {{=sel_row['w=0.68']:.3f}} and the hits-only stack between {{=hits_row['w=0.68']:.3f}} and {{=hits_row['w=0.4']:.3f}}. The rank blend, the average of the two stacks' empirical-distribution ranks, sits between them. The decision at deployment has to be made without knowing the weight, and `cypmode/challenge/hedged_blend.py` states two rules before the numbers: the blend is adopted if over the plausible band its worst-case AUC is at least the better worst case of the two pure models, and the selection-aware stack replaces the hits-only one only if its mean AUC is higher at every band weight at both cut-offs *and* the corrected paired p is below 0.05 at every one of them. The paired differences in the band are large (+0.10 to +0.17 AUC) and their p-values are 0.05 to 0.16, so that neither rule is met: the blend gate is {{='MET' if hedged_gates.get('hedged blend adopted by the minimax rule') == 'MET' else 'NOT MET'}} and the replacement gate is {{='MET' if hedged_gates.get('selection-aware replaces hits-only') == 'MET' else 'NOT MET'}}.

What follows from a gate that is not met is exactly what the project's rule says: the default (the hits-only stack) stays the model. The evidence is not symmetric, though, and the project does not hide it. Inside the band that the two independent estimates support, the selection-aware stack is far better, by 0.1 AUC or more, and outside it, at the edge where the scored set would have no non-hits at all, it is worse by 0.14. The gate is not met because the paired folds are few and the differences vary between them, not because the sign is in doubt. The build therefore writes both stacks as candidate files for CYP2D6, A (hits-only) and B (selection-aware), with CYP3A4 identical in the two (8.6), and the choice between them is a decision under uncertainty about a quantity that no experiment can measure; section 13.4 reports which is submitted.

*On what actually ran.* `cypmode/challenge/population_experiment.py` (v1; the sets S0 to S3), `selection_aware_v2.py` (the cross-fitted promotion-pattern gate and the Heckman candidate), `population_share.py` (the band from the public baseline), `test_nonhit_share.py` (the adjusted count and classify-and-count at three thresholds, with a nearest-neighbour hit fraction as a sanity check), `hedged_blend.py` (the grid of regimes, the blend and the two rules), `library_blocks.py` (the panel block, found from the data and not from hand-typed identifiers), and, for the same population with physics columns, `population_physics.py` (9.11 and Part X). The logs read above are `population_panel_CYP2D6_r3.log`, `selection_aware_v2_CYP2D6.log`, `test_nonhit_share_CYP2D6.log` and `hedged_blend_CYP2D6.log`.
"""),
    figure("ml_hit_gate_vs_screen"),
    figure("ml_selection_sensitivity"),
    md("""
## 9.2 Making SAR explicit: pair models and compounds that differ by one change

> **Field:** Cheminformatics · Machine learning

*Where this section sits:* 7.3 gave the model the labels of a compound's neighbours as four columns. A nearest neighbour is only a vote; it cannot tell which differences between a compound and its neighbour matter. That knowledge is what medicinal chemists call the **structure-activity relationship** (**SAR**): a methyl group next to an amine may switch the activity off, a chlorine on a distant ring may do nothing. This section tries two ways of learning it from the data, and finds, for a reason that the data make plain, that neither can be learned from this much of it.

**A pair model.** Instead of describing a compound and its neighbours separately, describe each *pair*. For a compound $q$ and each of its $K = 5$ most similar labelled training compounds $n$, the row is the descriptors of $q$, the *differences* of the descriptors of $q$ and $n$, the Tanimoto similarity of the two, the label of $n$, and the rank of $n$ among the neighbours. One LightGBM learns the label of $q$ from such rows, so that it can learn, for instance, that a neighbour that is positive but has a lower hydrogen-abstraction energy at the best site makes $q$ less likely to be positive (with the physics block of Part X as the descriptors, the differences are physical). For a new compound the five pair predictions are averaged with the similarity as weight. The training pairs use only training rows and a training compound is not its own neighbour (the rule of 7.3). The next cell builds the pair rows of one validation compound, fits the model on the first outer fold of CYP3A4, and compares it with the plain LightGBM on the same descriptors and with the neighbours' vote.
"""),
    code(PAIR_DEMO),
    md("""
The pair rows of the typical validation compound of this fold (the one whose mean neighbour similarity is the median) have {{=pair_columns}} columns each, {{=pair_rows_fit:,}} rows for the {{=len(y_fit):,}} training compounds. Its five neighbours are at similarities around {{=example_values['similarity']:.2f}}, far from identical, and a typical compound of this pool has no close relative in it: that is the first reason the pair model cannot teach much. The neighbours' vote scores an AUC of {{=pair_compare.iloc[1]['AUC on the validation compounds']:.3f}}, the plain LightGBM {{=pair_compare.iloc[0]['AUC on the validation compounds']:.3f}} and the pair model {{=pair_compare.iloc[2]['AUC on the validation compounds']:.3f}}. The pair model learned as much as the model that never saw a pair and not more.

**Matched molecular pairs.** The same idea can be made exact. Two compounds form a **matched molecular pair** when cutting one bond of each leaves the *same* large constant part and *different* small variable parts, one to ten heavy atoms (RDKit's `rdMMPA` does the fragmentation). The change of one variable part into another is a *transformation*, written "variable of $n$ $\\gg$ variable of $q$". The classical use (Kenny and Sadowski; Hussain and Rea, 2010) tabulates, over all pairs that share a transformation $T$, how the activity moved, and predicts that the same change in a new compound moves it likewise; the Free-Wilson analysis is the same idea for sums of substituent contributions. The project's version, `cypmode/challenge/mmp_transfer.py`, sums over the training pairs $a \\to b$ of the transformation the label change, shrunk toward zero when the transformation has been seen few times,
$$\\delta_T \\;=\\; \\frac{\\sum_{a \\to b \\,\\in\\, T} (y_b - y_a)}{N_T + 3},$$
with $y_a$ and $y_b$ the labels (0 or 1) of the two compounds of a pair and $N_T$ the number of training pairs of the transformation $T$, and gives LightGBM five features per compound (the number of matched partners, their mean label, the mean of $y_n + \\delta_T$ clipped to $[0, 1]$, the mean $\\delta_T$, and the number of partners whose transformation has at least three supporting pairs). The table and the partners' labels come from the training rows only, and a training compound's own pairs are removed from the table before its features are computed. The cell reads the cached pairs for the CYP3A4 pool and looks at what the table holds.
"""),
    code(MMP_DEMO),
    md("""
The whole collection yields {{=mmp_facts['directed pairs among all the compounds']:,}} directed pairs and {{=mmp_facts['distinct transformations among all the compounds']:,}} distinct transformations; inside the CYP3A4 pool there are {{=mmp_facts['directed pairs inside the CYP3A4 pool']:,}} pairs, and {{=mmp_facts['pool compounds with a matched partner in the pool']:.0%}} of the pool compounds have a matched partner in the pool, {{=mmp_facts['blinded compounds with a matched partner in the pool']:.0%}} of the blinded compounds. That is already a thin coverage, and the shape of the table is thinner: of the transformations, {{=mmp_facts['transformations seen in 2 or more pool pairs']}} are seen in two pool pairs or more, {{=mmp_facts['transformations seen in 3 or more pool pairs']}} in three or more, and the most frequent is seen in {{=mmp_facts['pairs of the most frequent transformation']}}. The table above is the best the data support, and a shrunk change of {{=report[0][2]:+.2f}} from {{=report[0][1]}} pairs is a statement that nothing has been learned. A matched-pair table works in a data set where one series was explored by many analogues. This is a diversity collection with analogue expansion on top, so that each transformation occurs once or twice, and one or two examples are not a trend.

*The gates.* Both ideas, and the pair model with the xTB physics block as the descriptors, were added to the default stack and judged by the gate of 6.7 for each enzyme.
"""),
    code(PAIR_RESULTS),
    md("""
No gate is met ({{=n_sar_met}} of the {{=len(sar_table)}} lines), the largest stacked change of the table, in absolute value, is {{=sar_largest:.4f}} AUC and the smallest p-value is {{=sar_smallest_p:.3f}}. The pair model reaches an AUC of {{=sar_table.iloc[0]['best candidate alone (AUC, z0 1.0)']:.3f}} alone on CYP3A4, close to the single models of 8.1 and well below the stack; with the physics differences {{=sar_table.iloc[2]['best candidate alone (AUC, z0 1.0)']:.3f}}, a better single model that adds {{=sar_table.iloc[2]['gain at z0 1.0']:+.4f}} to the stack. The reasons are the ones the section has shown. A neighbour-based model can only be as good as the neighbours, and the neighbours of a typical compound here are far; a transformation-based one needs transformations seen many times, and none is. Both would be strong on a series, a set of analogues of one scaffold with many measured members, and the blinded set is made of analogue expansions, which is why the idea was worth the test; but the expansions are of top hits taken from a diversity collection, so that each series in the training file is a handful of compounds and not a hundred. The idea reappears in Part XI in a different form, as the agreement of a pose with the poses of its analogues.

*On what actually ran.* `cypmode/challenge/pair_transfer.py` (`neighbour_table`, `pair_matrix`, `fit_predict_pairs`, the gate through `addition_experiment.py`), `cypmode/challenge/mmp_transfer.py` (`build_pairs`, `get_pairs` with its cache, `mmp_label_features`, `transformation_report`), and the logs `pair_transfer_<enzyme>.log`, `pair_transfer_physics_<enzyme>.log` and `mmp_transfer_<enzyme>.log`. The nested design, the weights and the corrected t are those of Part VI.
"""),
    md("""
## 9.3 The label as a threshold of two noisy numbers: a committee that outputs a probability

> **Field:** Statistics · Machine learning

*Where this section sits:* 9.2 asked for more from the neighbours. This section asks for more from the *label*. The label is a yes/no, but it is made from two measured numbers, each with a measurement error, by a threshold (section 2.3). A classifier fitted to the label sees only the thresholded shadow of the process that made it; 8.1 already used the numbers in two regression members. This section builds a model of the whole process and asks it for a probability and for an expected improvement, the quantity used elsewhere in the project to decide where an extra measurement or sample is worth most.

**The model.** The shift $s = t - \\max(d, 4)$ is positive-labelled when $s > \\tau = \\log_{10} 2$. A **committee** of gradient-boosted regressors, differing in the feature set, in a random 70 percent of the training rows (bagging) and in the seed, each fits *both* arms, $t$ and $d$, on arms clipped from below at the assay floor 4 (clipping leaves every label unchanged, because only $\\max(d, 4)$ and whether $t$ clears 4 matter; 9.6 tests the clipping on its own). Each training row is weighted by the inverse of its own measurement variance, $\\mathrm{sd}(s)^2 = \\mathrm{sd}_t^2 + [d \\ge 4]\\,\\mathrm{sd}_d^2$, so that a measurement that is known to be noisy counts less (a weighted regression, section 4.6). A member's predicted shift is $\\hat t - \\max(\\hat d, 4)$. The committee mean $m$ is the point estimate, and the spread $\\mathrm{sd}_e$ of the members' predictions is the *epistemic* uncertainty (the uncertainty of the model, as against the noise of the measurement). The out-of-bag error of the members, each scored on the rows it never saw, gives a fair scale $\\sigma_{\\text{base}}$ for the noise, and the predictive distribution of the shift is taken to be a normal one, $s \\sim \\mathcal N\\big(m,\\ \\sigma_{\\text{base}}^2 + \\mathrm{sd}_e^2\\big)$. With $\\sigma = \\sqrt{\\sigma_{\\text{base}}^2 + \\mathrm{sd}_e^2}$ and $z = (m - \\tau)/\\sigma$, the two outputs, the probability $P$ that the label is positive and the expected improvement $\\mathrm{EI}$, are
$$P(\\text{label}) \\;=\\; \\Phi(z), \\qquad \\mathrm{EI} \\;=\\; \\sigma\\,\\big(\\varphi(z) + z\\,\\Phi(z)\\big),$$
where $\\Phi$ and $\\varphi$ are the cumulative distribution function and the density of the standard normal. The expected improvement is the expected size of the exceedance of the threshold, $E[\\max(s-\\tau, 0)]$; it comes from Bayesian optimisation (Jones, Schonlau and Welch, 1998), where it chooses the next experiment. Three candidates are cached: `committee_m` (the point estimate alone, to isolate what the uncertainty adds), `committee_p` and `committee_ei`. Only the training rows' arms and standard deviations and the evaluated compounds' structures are read; the evaluated rows' measurements never are. The cell shows the two formulas on three compounds, checked by simulation.
"""),
    code(COMMITTEE_MATH),
    md("""
The formulas agree with the simulation to {{=formula_check:.4f}}. The three compounds make the point of having both outputs. The long shot has the lowest probability of the three ({{=toy_compounds.iloc[2]['P(label)']:.2f}}) and the highest expected improvement ({{=toy_compounds.iloc[2]['expected improvement']:.2f}}); the sure compound barely above the threshold has the highest probability ({{=toy_compounds.iloc[0]['P(label)']:.2f}}) and a smaller expected improvement ({{=toy_compounds.iloc[0]['expected improvement']:.2f}}). The probability says how likely the yes is, the expected improvement how much the compound is worth if its uncertainty is explored. Now the committee on the real compounds, in the nested design.
"""),
    code(COMMITTEE_RESULTS),
    md("""
The committee, alone, ranks about as well as the single members of 8.1: the best of its three outputs scores an AUC of {{=committee_3a4['best member alone, AUC at z0 1.0']:.3f}} on CYP3A4 at $z_0 = 1.0$. Inside the stack its three candidates add {{=committee_3a4['gain at z0 1.0']:+.4f}} AUC at $z_0 = 1.0$ (corrected p = {{=committee_3a4['p at z0 1.0']:.2f}}) and {{=committee_3a4['gain at z0 1.96']:+.4f}} at $z_0 = 1.96$ (p = {{=committee_3a4['p at z0 1.96']:.2f}}); the stacked gate is {{=committee_3a4['gate']}}. The second gate, calibration, is the more instructive failure. The probability that the committee outputs is *not* a probability: its mean is {{=committee_cal_table.iloc[0]['mean predicted']:.3f}} where the observed positive share is {{=committee_cal_table.iloc[0]['mean observed']:.3f}}, the ECE is {{=committee_cal_table.iloc[0]['ECE']:.3f}} and the slope {{=committee_cal_table.iloc[0]['slope']:.2f}} at $z_0 = 1.0$ ({{=committee_cal_table.iloc[0]['calibration gate']}} against the bar of 0.05 and 0.8 to 1.2). The normal model of the error is too simple: the arms are floored, the errors are not symmetric, and the committee's members share their errors, so that the spread of the members understates how far the truth can be. The expected improvement inherits the fault. The idea of a model of the whole process is sound and the design, with a Gaussian predictive distribution, did not deliver the calibrated probability that was its reason. What does carry over is the weighting by the inverse of the measurement variance, which 9.8 tests separately.

*On what actually ran.* `cypmode/challenge/committee_label_model.py` (`measurement_variance`, `committee_predict` with its three outputs `m`, `p`, `ei`, and the calibration gate), run through `addition_experiment.py` for the stacked gate; the log `committee_label_model_<enzyme>.log`; the family `arm_committee` of `nested_runners.py`. The calibration bar is the one of `calibration_experiment.py` (8.5).
"""),
    figure("data_label_gaussian"),
]
