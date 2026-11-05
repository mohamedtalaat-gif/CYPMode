# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part II (second half): how the compounds were measured, how the test set differs from the training set, the other data sources, matching structures across sources, the leaderboard as data."""

from notebooks.notebook_builder import code, figure, md

PANEL_BLOCK = '''
# The training file is not one library. Compounds were measured on several enzymes, and which enzymes a compound was measured on depends on where its number lies. library_blocks finds the
# stretch of compound numbers where a large share of the compounds carry a measurement on an enzyme other than CYP3A4: a rolling share over 300 compounds in the order of the numbers,
# and the longest run in which it stays at or above 0.3.
from cypmode.challenge.library_blocks import MIN_SHARE, WINDOW, compound_numbers, panel_block_mask

ids = compound_numbers(train["Molecule_Name"])
panel = panel_block_mask(train)
region = np.where(panel, "panel", np.where(ids < ids[panel].min(), "low", "top"))
other = train[["CYP1A2_pIC50_TDI_condition", "CYP2C9_pIC50_TDI_condition", "CYP2D6_pIC50_TDI_condition"]].notna().any(axis=1).to_numpy()
d3_measured, d6_measured = train["CYP3A4_pIC50_TDI_condition"].notna().to_numpy(), train["CYP2D6_pIC50_TDI_condition"].notna().to_numpy()
d3_label, d6_label = train["CYP3A4_is_TDI"].fillna(False).astype(bool).to_numpy(), train["CYP2D6_is_TDI"].fillna(False).astype(bool).to_numpy()
block_rows = []
for name, title in (("low", "low block"), ("panel", "all-isoform panel block"), ("top", "top block")):
    sel = region == name
    block_rows.append({"block": title, "compounds": int(sel.sum()), "numbers from": int(ids[sel].min()), "numbers to": int(ids[sel].max()), "measured on another enzyme": float(other[sel].mean()),
                       "measured on CYP3A4": float(d3_measured[sel].mean()), "measured on CYP2D6": float(d6_measured[sel].mean()),
                       "CYP3A4 positives": int(d3_label[sel].sum()), "CYP2D6 positives": int(d6_label[sel].sum())})
block_table = pd.DataFrame(block_rows)
show_table(block_table, floatfmt=".3f")

order = np.argsort(ids, kind="stable")
rolling = pd.Series(other[order].astype(float)).rolling(WINDOW, center=True, min_periods=WINDOW // 2).mean().to_numpy()
fig, axis = plt.subplots(figsize=(10.5, 3.4))
axis.plot(np.arange(len(rolling)), rolling, color=theme["blue"], lw=1.8)
axis.axhline(MIN_SHARE, color=theme["muted"], ls=":", lw=1)
in_panel_sorted = panel[order]
axis.fill_between(np.arange(len(rolling)), 0, 1, where=in_panel_sorted, color=theme["orange"], alpha=0.18, transform=axis.get_xaxis_transform(), label="the all-isoform panel block")
axis.set(xlabel="training compounds in the order of their numbers", ylabel=f"share measured on another enzyme\\n(rolling over {WINDOW} compounds)", ylim=(0, 1))
axis.legend(frameon=False, loc="upper left")
tidy(axis, theme)
fig.tight_layout()
n_panel, share_inside, share_outside = int(panel.sum()), float(other[panel].mean()), float(other[~panel].mean())
'''

SELECTION = '''
# The factorisation behind a "hit": inside the panel block, a compound is a CYP2D6 positive only if it was measured on CYP2D6 (a hit) AND its shift passes the threshold. So
# P(positive) = P(hit) x P(positive | hit), computed here from the counts.
in_panel = panel
n_in, n_hit = int(in_panel.sum()), int((d6_measured & in_panel).sum())
n_positive = int((d6_label & in_panel).sum())
p_hit, p_pos_given_hit, p_pos = n_hit / n_in, n_positive / n_hit, n_positive / n_in
print(f"CYP2D6 inside the panel block: {n_in} compounds, {n_hit} measured (hits), {n_positive} positive")
print(f"P(hit) = {p_hit:.4f}; P(positive | hit) = {p_pos_given_hit:.4f}; the product = {p_hit * p_pos_given_hit:.4f}; P(positive) counted directly = {p_pos:.4f}")
outside_measured = int((d6_measured & ~in_panel).sum())
print(f"CYP2D6 measurements outside the panel block: {outside_measured} of {int(d6_measured.sum())}")
'''

SIMILARITY = '''
# How close is each compound to the compounds the model can learn from? For every compound, the highest Tanimoto similarity (on Morgan fingerprints of radius 2 and 2048 bits, section 3.1) to any compound
# of a reference set: the test compounds against all training compounds; every training compound against all OTHER training compounds; and, as the project's earlier check did, every training compound
# against the training part of a random 5-fold split.
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp_train = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train["SMILES"]]
fp_test = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in test["SMILES"]]
K_NEIGHBOURS = 5
nn_test = np.array([max(DataStructs.BulkTanimotoSimilarity(f, fp_train)) for f in fp_test])
nn_leave_one_out, near_train, near_test = np.zeros(len(fp_train)), np.zeros(len(fp_train)), np.zeros(len(fp_train))
for i, f in enumerate(fp_train):
    to_train = np.array(DataStructs.BulkTanimotoSimilarity(f, fp_train))
    to_train[i] = -1.0                                                  # a compound is not its own neighbour
    to_test = np.array(DataStructs.BulkTanimotoSimilarity(f, fp_test))
    nn_leave_one_out[i] = to_train.max()
    near_train[i], near_test[i] = np.sort(to_train)[-K_NEIGHBOURS:].mean(), np.sort(to_test)[-K_NEIGHBOURS:].mean()
nn_cv = np.zeros(len(fp_train))
for fit_part, held_out in KFold(5, shuffle=True, random_state=42).split(fp_train):
    reference = [fp_train[j] for j in fit_part]
    for j in held_out:
        nn_cv[j] = max(DataStructs.BulkTanimotoSimilarity(fp_train[j], reference))

def summarise(values):
    return {"compounds": len(values), "median nearest-neighbour similarity": float(np.median(values)), "share at 0.5 or more": float((values >= 0.5).mean()), "share at 0.7 or more": float((values >= 0.7).mean())}
groups = {"the 750 test compounds, against the training set": nn_test, "training compounds in a random 5-fold split, against the training part": nn_cv,
          "training compounds, against all other training compounds": nn_leave_one_out, "  of which the low block": nn_leave_one_out[region == "low"],
          "  of which the all-isoform panel block": nn_leave_one_out[region == "panel"], "  of which the top block": nn_leave_one_out[region == "top"]}
similarity_table = pd.DataFrame([{"compounds compared": name, **summarise(v)} for name, v in groups.items()])
show_table(similarity_table, floatfmt=".3f")

fig, axis = plt.subplots(figsize=(10.5, 3.6))
bins = np.linspace(0.1, 1.0, 37)
for values, color, label in ((nn_test, theme["red"], "test compounds"), (nn_cv, theme["blue"], "random cross-validation folds"), (nn_leave_one_out[region == "panel"], theme["orange"], "panel block")):
    axis.hist(values, bins=bins, histtype="step", lw=2.0, density=True, color=color, label=label)
axis.axvline(0.5, color=theme["muted"], ls=":", lw=1)
axis.set(xlabel="highest Tanimoto similarity to any compound of the reference set", ylabel="density")
axis.legend(frameon=False)
tidy(axis, theme)
fig.tight_layout()
test_summary, cv_summary = summarise(nn_test), summarise(nn_cv)
'''

REGIONS = '''
# Where do the test compounds sit in the numbering? The same three ranges as the training blocks (below the panel, inside it, above it), counted for both files.
test_ids = compound_numbers(test["Molecule_Name"])
test_region = np.where(test_ids < ids[panel].min(), "low", np.where(test_ids > ids[panel].max(), "top", "panel"))
region_rows = []
for name, title in (("low", "low block"), ("panel", "all-isoform panel block"), ("top", "top block")):
    sel_test, sel_train = test_region == name, region == name
    region_rows.append({"range of compound numbers": title, "training compounds": int(sel_train.sum()), "test compounds": int(sel_test.sum()), "test share": float(sel_test.mean()),
                        "lowest test number": int(test_ids[sel_test].min()) if sel_test.any() else "-", "highest test number": int(test_ids[sel_test].max()) if sel_test.any() else "-"})
region_table = pd.DataFrame(region_rows)
show_table(region_table, floatfmt=".3f")
top_train_high, top_test_low = int(ids[region == "top"].max()), int(test_ids[test_region == "top"].min())
top_d6_share, top_d3_share = float(d6_measured[region == "top"].mean()), float(d3_measured[region == "top"].mean())
n_test_top = int((test_region == "top").sum())
print(f"training compounds of the top block reach number {top_train_high}; the lowest test number of the top range is {top_test_low}")
'''

ADVERSARIAL = '''
# Can a model tell a test compound from a training compound by its structure? A gradient-boosted classifier is trained to separate the two files (5-fold cross-validation, so every compound is scored
# by a model that did not see it); its area under the ROC curve says how different they are, and Bayes' rule turns its probabilities into the density ratio p_test(x) / p_train(x).
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

generator_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
features_all = np.array([generator_1024.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in pd.concat([train["SMILES"], test["SMILES"]])], dtype=np.float32)
source = np.r_[np.zeros(len(train)), np.ones(len(test))]                     # 0 = training compound, 1 = test compound
p_test = np.zeros(len(source))
for fit_part, held_out in StratifiedKFold(5, shuffle=True, random_state=0).split(features_all, source):
    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, n_jobs=2, verbose=-1, random_state=0)
    p_test[held_out] = model.fit(features_all[fit_part], source[fit_part]).predict_proba(features_all[held_out])[:, 1]
auc_shift = roc_auc_score(source, p_test)
p_on_train = p_test[: len(train)]
ratio = p_on_train / (1 - p_on_train) * (len(train) / len(test))                 # Bayes: p_test(x) / p_train(x) = [P(test | x) / P(train | x)] x (n_train / n_test)
ess_ratio = ratio.sum() ** 2 / (ratio**2).sum()                                 # effective sample size of the weights (defined in the text)

# The project's own proxy for the same ratio (nested_runners.testlike_weights): similarity to the nearest test structures over similarity to the nearest other training compounds, tempered.
def proxy_weights(alpha):
    w = np.clip((near_test / np.maximum(near_train, 1e-3)) ** alpha, 0.2, 5.0)
    return w / w.mean()
weight_rows = []
for name, w in (("classifier ratio, untempered", ratio / ratio.mean()), ("project proxy, alpha = 1", proxy_weights(1.0)), ("project proxy, alpha = 0.5", proxy_weights(0.5))):
    row = {"weights": name, "effective sample size": float(w.sum() ** 2 / (w**2).sum()), "largest weight": float(w.max())}
    for block in ("low", "panel", "top"):
        row[f"share of the weight on the {block} block"] = float(w[region == block].sum() / w.sum())
    weight_rows.append(row)
weight_table = pd.DataFrame(weight_rows)
print(f"AUC of telling a test compound from a training compound by structure: {auc_shift:.3f} (0.5 would mean no difference)")
show_table(weight_table, floatfmt=".2f")
share_rows = {block: float((region == block).mean()) for block in ("low", "panel", "top")}
'''

SOURCES = '''
# The other sources the project read, with their sizes. Counts are read from the files each source's loader parses, and the overlap with the challenge compounds is counted by the
# connectivity key (the first block of the InChIKey, a fast key that ignores stereochemistry and isotopes, section 2.9).
import glob
import json

from cypmode.challenge import bindingdb_auxiliary as bindingdb
from cypmode.challenge import chembl_auxiliary as chembl
from cypmode.challenge import pubchem_auxiliary as pubchem
from cypmode.challenge.external_activity import connectivity_key

challenge_keys = {k for k in (connectivity_key(s) for s in pd.concat([train["SMILES"], test["SMILES"]])) if k}

def describe(name, what, smiles, records=None):
    keys = [connectivity_key(s) for s in smiles]
    shared = sum(1 for k in set(keys) if k in challenge_keys)
    return {"source": name, "what is measured": what, "records": records if records is not None else len(smiles), "distinct compounds": len(set(smiles)), "connectivity keys shared with the challenge": shared}

source_rows = []
tdc_smiles = []
for isoform in ("cyp1a2", "cyp2c9", "cyp2c19", "cyp2d6", "cyp3a4"):
    table = pd.read_csv(ROOT / f"data/tdi_track/inputs/tdc_cache/{isoform}_veith.tab", sep="\\t")
    tdc_smiles += table[next(c for c in table.columns if c.lower() in ("drug", "smiles"))].tolist()
source_rows.append(describe("TDC / NCATS qHTS panel (five enzymes)", "yes/no inhibitor call per enzyme", pd.Series(tdc_smiles).drop_duplicates().tolist(), records=len(tdc_smiles)))
for isoform in ("CYP3A4", "CYP2D6"):
    chembl_records = json.loads((ROOT / chembl.CHEMBL_RAW_DIR / f"{isoform}_{chembl.CHEMBL_ISOFORM_TARGETS[isoform]}.json").read_text())
    frame = chembl.load_chembl_isoform(ROOT / chembl.CHEMBL_RAW_DIR / f"{isoform}_{chembl.CHEMBL_ISOFORM_TARGETS[isoform]}.json")
    source_rows.append(describe(f"ChEMBL, {isoform}", "pChEMBL value (-log10 of IC50, Ki, ...) from the literature", frame["SMILES"].tolist(), records=len(chembl_records)))
for isoform, uniprot in (("CYP3A4", "P08684"), ("CYP2D6", "P10635")):
    frame = bindingdb.load_bindingdb_isoform(ROOT / f"data/tdi_track/inputs/bindingdb_cache/{isoform}_{uniprot}.json")
    source_rows.append(describe(f"BindingDB, {isoform}", "Ki / IC50 / EC50 in micromolar, from the literature", frame["SMILES"].tolist()))
aid = pubchem.load_aid1851()
source_rows.append(describe("PubChem AID 1851 (NCATS qHTS, five enzymes)", "active / inactive and a potency per enzyme", aid["SMILES"].tolist()))
for isoform, path in pubchem.SINGLE_ISOFORM_AID_PATHS.items():
    frame = pubchem.load_single_isoform_aid(ROOT / path, isoform)
    source_rows.append(describe(f"PubChem single-enzyme screen, {isoform}", "active / inactive", frame["SMILES"].tolist()))
octant = pd.read_parquet(ROOT / "data/tdi_track/inputs/octant_cache/inhibition.parquet")
octant_validation = json.loads((ROOT / "data/tdi_track/inputs/octant_cache/motif_potency_validation.json").read_text())
source_rows.append({"source": "Octant diversity-library screen (CYP3A4)", "what is measured": "pIC50 after a 30-minute pre-incubation", "records": len(octant), "distinct compounds": len(octant),
                    "connectivity keys shared with the challenge": "-"})
source_table = pd.DataFrame(source_rows)
show_table(source_table)
print(f"Octant: {octant_validation['n_qc_passed_with_pic50']} of {len(octant)} records have a pIC50 that passed the quality control of the dose-response fit")
tdc_keys = {connectivity_key(s) for s in set(tdc_smiles)} - {None}
aid_keys = {connectivity_key(s) for s in aid["SMILES"]} - {None}
share_tdc_in_aid = len(tdc_keys & aid_keys) / len(tdc_keys)
print(f"{len(tdc_keys & aid_keys)} of the {len(tdc_keys)} connectivity keys of the TDC panel are also keys of PubChem assay 1851 ({share_tdc_in_aid:.0%})")
'''

STANDARDISE = '''
# What the standardised parent does to a salt. Three real compounds of the TDC panel that are in the challenge files only under another form: the same medicine as a salt, with a counter-ion
# or a solvent molecule in the same string. (The cell finds them by comparing the keys of section 2.9; only those that the canonical-SMILES comparison misses are shown.)
from cypmode.challenge.structure_standardization import standardized_parent_smiles

def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None

tdc_unique = pd.Series(tdc_smiles).drop_duplicates().reset_index(drop=True)
challenge_all = pd.concat([train["SMILES"], test["SMILES"]], ignore_index=True)
canonical_challenge = set(challenge_all.map(canonical).dropna())
parent_challenge = set(challenge_all.map(standardized_parent_smiles).dropna())
tdc_canonical = tdc_unique.map(canonical)
tdc_parent = tdc_unique.map(standardized_parent_smiles)
tdc_key = tdc_unique.map(connectivity_key)
rule_rows = [
    {"matching rule": "the strings are identical", "TDC strings matched to a challenge compound": int(tdc_unique.isin(set(challenge_all)).sum())},
    {"matching rule": "the canonical SMILES are identical", "TDC strings matched to a challenge compound": int(tdc_canonical.isin(canonical_challenge).sum())},
    {"matching rule": "the connectivity keys of the InChIKey are identical", "TDC strings matched to a challenge compound": int(tdc_key.isin(challenge_keys).sum())},
    {"matching rule": "the standardised parent structures are identical (salts and solvents removed, charges neutralised)", "TDC strings matched to a challenge compound": int(tdc_parent.isin(parent_challenge).sum())}]
show_table(pd.DataFrame(rule_rows))
only_by_parent = tdc_unique[(~tdc_canonical.isin(canonical_challenge)) & tdc_parent.isin(parent_challenge)]
n_only_by_parent, n_salts = len(only_by_parent), int(tdc_unique.str.contains(r"\\.").sum())
for original in only_by_parent.head(3):
    print(f"as written: {original}\\n   parent:  {standardized_parent_smiles(original)}")
print(f"{n_salts} of the {len(tdc_unique)} distinct TDC strings contain more than one component")
'''

LEADERBOARD = '''
# The leaderboard as data. A prediction file is scored against the same hidden labels whoever submitted it; each row of the public table gives accuracy A, precision P and recall R of one file,
# and those three numbers fix the positive rate p of the scored set: p = (1 - A) / (1 - 2R + R/P). First a check on a synthetic set with a known positive rate, then the real tables.
from cypmode.challenge.leaderboard_prior import estimate_from_csv, implied_prevalence

rng = np.random.default_rng(7)
check_rows = []
for true_rate in (0.06, 0.3):
    labels = rng.random(100_000) < true_rate
    for strength in (0.8, 2.0):                                          # how far apart the scores of positives and negatives lie, in units of their spread
        scores = labels * strength + rng.normal(size=labels.size)
        called = scores > np.quantile(scores, 1 - 1.3 * true_rate)
        tp, fp, fn = int((called & labels).sum()), int((called & ~labels).sum()), int((~called & labels).sum())
        accuracy, precision, recall = (called == labels).mean(), tp / (tp + fp), tp / (tp + fn)
        check_rows.append({"positive rate of the set (counted)": float(labels.mean()), "A": accuracy, "P": precision, "R": recall, "recovered p": float(implied_prevalence(accuracy, precision, recall))})
show_table(pd.DataFrame(check_rows), floatfmt=".4f")

board_rows = []
for enzyme in ("CYP3A4", "CYP2D6"):
    for kind, stamp in (("interim", "2026-10-04"), ("live", "2026-10-07")):
        estimate = estimate_from_csv(ROOT / f"data/tdi_track/inputs/leaderboard_snapshots/{kind}_{enzyme}_{stamp}.csv")
        board_rows.append({"enzyme": enzyme, "table": f"{kind} ({stamp})", "rows used": estimate["n_rows"], "positive rate p": estimate["prevalence"], "spread of p across rows": estimate["spread"]})
board_table = pd.DataFrame(board_rows)
show_table(board_table, floatfmt=".5f")
prevalence_board = {(r["enzyme"], r["table"].split()[0]): r["positive rate p"] for r in board_rows}
'''

PREVALENCE_COMPARE = '''
# The positive rate of the training compounds that carry a label with both arms, against the rate of the scored test set recovered from the tables.
rate_view_3a4 = float(training_view(train, "CYP3A4").query("population == 'both_arms'")["label"].astype(bool).mean())
rate_both_arms = float(training_view(train, "CYP2D6").query("population == 'both_arms'")["label"].astype(bool).mean())
low_high = {e: (min(v for (en, _), v in prevalence_board.items() if en == e), max(v for (en, _), v in prevalence_board.items() if en == e)) for e in ("CYP3A4", "CYP2D6")}
compare = pd.DataFrame([
    {"enzyme": "CYP3A4", "training, both arms": rate_view_3a4, "scored set, interim table": prevalence_board[("CYP3A4", "interim")], "scored set, live table": prevalence_board[("CYP3A4", "live")]},
    {"enzyme": "CYP2D6", "training, both arms": rate_both_arms, "scored set, interim table": prevalence_board[("CYP2D6", "interim")], "scored set, live table": prevalence_board[("CYP2D6", "live")]}])
show_table(compare, floatfmt=".3f")
'''

POPULATION_SHARE = '''
# The same three numbers can test a hypothesis about what the scored set contains (population_share): re-weight the baseline's out-of-fold predictions on the training compounds by w, the weight of
# the never-measured compounds, and c, the weight of the positives among the measured ones, and keep every (w, c) at which the weighted accuracy, precision and recall are within a chi-square
# 95 % region of the baseline's three published numbers.
band = pd.read_csv(ROOT / "data/tdi_track/outputs/comparisons/population_share_band.csv")
plausible = band[band["plausible"]]
band_summary = {"cells": len(band), "plausible": len(plausible), "w low": float(plausible["w"].min()), "w high": float(plausible["w"].max()), "grid w low": float(band["w"].min()), "grid w high": float(band["w"].max()),
                "c low": float(plausible["c"].min()), "c high": float(plausible["c"].max())}
by_w = band.groupby("w")["plausible"].any()
print(f"{band_summary['plausible']} of the {band_summary['cells']} (w, c) pairs of the grid are within the 95 % region; w runs from {band_summary['grid w low']} to {band_summary['grid w high']} on the grid and the plausible pairs "
      f"cover w from {band_summary['w low']} to {band_summary['w high']}: {int(by_w.sum())} of the {len(by_w)} values of w have at least one plausible c")
'''

LABEL_SHIFT = '''
# A second estimate of the positive rate of the scored set that reads no board (cypmode/challenge/label_shift.py, job 285): the log of that job holds, for each enzyme, the estimate for the scored compounds with its
# 95 % interval, the rate that the final build uses (read there for the comparison only), the error of the recovery test of known rates as a share of the rate, and the verdict of the rule written before the first run.
from cypmode.audit.method_report import label_shift_summary

shift = label_shift_summary()
shift_table = pd.DataFrame([{"enzyme": iso, "estimate": float(v["estimate"]), "interval low": float(v["lo"]), "interval high": float(v["hi"]), "rate of the final build": float(v["board"]),
                             "recovery error, share of the rate": float(v["ratio"]), "verdict of the rule": v["verdict"]} for iso, v in shift.items()])
inside = {iso: "inside" if float(v["lo"]) <= float(v["board"]) <= float(v["hi"]) else "outside" for iso, v in shift.items()}
show_table(shift_table, floatfmt=".3f")
'''

CELLS = [
    md("""
## 2.6 Which compounds were measured on which enzyme: blocks and selection

> **Field:** Data science · Statistics

Section 2.2 showed that the compounds are not measured evenly across the four enzymes. The pattern has a structure that matters for everything that follows. Compound names end in a number (the letters `OCNT-` and seven digits), and when the compounds are put in the order of that number, the share of
compounds that carry a measurement on an enzyme *other than CYP3A4* is not constant. The module `cypmode/challenge/library_blocks.py` finds the stretch where that share is high without being told where it is: it computes the share in a rolling window
of compounds in the order of the numbers and takes the longest run in which the share stays above a minimum. The cell runs it on the training file and draws the share along the numbering.
"""),
    code(PANEL_BLOCK),
    md("""
The window is {{=WINDOW}} compounds wide and the minimum share is {{=MIN_SHARE}}. The result is a block of {{=n_panel}} compounds, here called the **all-isoform panel block**: {{=share_inside:.0%}} of
its compounds carry a measurement on another enzyme, against {{=share_outside:.0%}} of the compounds outside it. The numbers below the block and the numbers above it are the other two blocks (the table gives the share of their compounds that carry a measurement on another enzyme).

**What a missing measurement means.** The explanation that fits the pattern, and that the project adopted, is a two-stage process. A collection of compounds is first screened cheaply on all enzymes at once, as a
**panel**; the compounds that show activity, the **hits**, are promoted to the expensive full dose-response measurement that yields a pIC50, and only for those is the measurement stored. If so, a compound of the panel block with no CYP2D6 measurement is
not a compound that was forgotten but one that was screened and *not* promoted, that is, a compound with no sign of inhibiting CYP2D6, which is what a label of "not a time-dependent inhibitor" would say. Outside the block the same empty cell means only "never tested". This is an inference from the
pattern in the files; the organisers' documentation was not found to state it, and Part IX says what depends on it. The inference is useful because of what it implies for counting. Inside the panel block, a compound is a positive only if it was a hit *and* its shift passes the threshold, so
that

$$P(\\text{positive}) \\;=\\; P(\\text{hit}) \\times P(\\text{positive} \\mid \\text{hit}),$$

where $P(\\cdot \\mid \\cdot)$ is a conditional probability, the probability of the left statement among the cases for which the right one holds. The factorisation is an identity of counting and the cell checks it on the CYP2D6 numbers.
"""),
    code(SELECTION),
    md("""
Of the {{=n_in}} compounds of the panel block, {{=n_hit}} ({{=p_hit:.0%}}) are CYP2D6 hits, and {{=n_positive}} of the hits are positive, which is {{=p_pos_given_hit:.1%}} of them but only {{=p_pos:.1%}} of all the compounds of the block. Which of the two rates is the right one for a test
compound depends on what the test set contains: only compounds that were hits (the first rate), or hits and non-hits in the proportion of the panel (the second). A model trained on the hits alone has never been shown a non-hit, and it is
the second situation that exposes it. This is called **selection bias**: the compounds from which a model learns were chosen by a process (here, being promoted after a screen) that is itself related to the label, so that they are not a random sample of the compounds that the model will meet. The
bias was described for statistical models of this kind by Heckman (1979), whose correction Part IX derives and tests as one of the project's ideas (the selection-aware model of CYP2D6). What the data files can and cannot say about the proportion in the scored set is the subject of section 2.10.

## 2.7 How the test set differs from the training set

> **Field:** Statistics · Cheminformatics

A model is trained on one set of compounds and judged on another; the judgement transfers only if the two resemble each other. The question is how far the 750 test compounds are from the training compounds, and there are two ways in which they can differ.
The compounds themselves can be different, that is, $P(x)$, the distribution of the structures $x$, can differ while the rule linking structure to label, $P(y \\mid x)$, stays the same: this is called **covariate shift**. Or the share of positives can differ, $P(y)$,
which is called **label shift**, or a difference in **prevalence**, the proportion of positives in a set. Both are measured here, the first in this section and the second in section 2.10.

**Similarity.** The simplest way to say how near a compound is to a set is the highest **Tanimoto similarity** between its fingerprint and the fingerprints of the compounds of that set: the share of the fingerprint bits switched on in either molecule that are switched on in both (section 3.2 builds the
formula, and section 3.1 the fingerprint). It runs from 0 (nothing shared) to 1 (identical). Values near 0.5 and above mean that the set contains a close relative of the compound, which for compounds of one medicinal-chemistry series is common: a series is made by changing one part of a molecule at a time. The cell computes it for the
test compounds against the training set and, for comparison, for the training compounds against the other training compounds, in the three blocks, and as the project's earlier check did, for the held-out fifth of a random five-fold split of the training set. That last comparison is what a cross-validated score is
implicitly tested against.
"""),
    code(SIMILARITY),
    md("""
A test compound has a close relative in the training set far more often than a held-out compound of a random split does: {{=test_summary['share at 0.5 or more']:.0%}} of the test compounds have a training neighbour at similarity 0.5 or more, against {{=cv_summary['share at 0.5 or more']:.0%}}
of the compounds of a random split, and the median is {{=test_summary['median nearest-neighbour similarity']:.2f}} against {{=cv_summary['median nearest-neighbour similarity']:.2f}}. The panel block, which the module describes as a diversity-library panel, is the most isolated of the three blocks. The consequence for a cross-validated score is
that it is measured on compounds that are, on average, farther from the training data than the test compounds are, and so is likely to understate how well a model that exploits close relatives will do on the test set, unless the split is made to resemble the test (section 6.4 and Part VI).

**Where the test compounds are.** The numbering gives another view. Counting the test compounds in the three ranges of compound numbers of the training blocks gives:
"""),
    code(REGIONS),
    md("""
{{=n_test_top}} of the {{=len(test)}} test compounds have a number above the panel block: they belong to the same top range as the {{=int((region == 'top').sum())}} training compounds of the top block, whose numbers stop at {{=top_train_high}}, and the lowest test number there is {{=top_test_low}}. Only a handful of the test compounds fall in the
low block or in the panel block. The training compounds of the top block were measured on CYP3A4 for {{=top_d3_share:.0%}} of them and on CYP2D6 for {{=top_d6_share:.0%}}: for CYP2D6 the training file holds no compound from the range where almost all the test compounds lie. The test set therefore does not resemble the diversity panel that makes most of the training file: it resembles the last, analog-rich range, which is the reason the project built a hold-out from the top block (section 6.8) and the reason for the weighting below.

**Importance weighting**. If a model is judged by its average loss $L(x)$ over the test distribution, that average is an integral over structures,

$$\\mathbb{E}_{\\text{test}}[L] \\;=\\; \\int p_{\\text{test}}(x)\\,L(x)\\,dx \\;=\\; \\int p_{\\text{train}}(x)\\,\\underbrace{\\frac{p_{\\text{test}}(x)}{p_{\\text{train}}(x)}}_{w(x)}\\,L(x)\\,dx \\;=\\; \\mathbb{E}_{\\text{train}}[\\,w(x)\\,L(x)\\,],$$

where $p_{\\text{test}}(x)$ and $p_{\\text{train}}(x)$ are the densities of the structures in the two sets, $\\mathbb{E}_{\\text{train}}$ is the average over the training compounds, and the second step only multiplies and divides by $p_{\\text{train}}(x)$ (it needs $p_{\\text{train}}(x) > 0$
wherever $p_{\\text{test}}(x) > 0$). The ratio $w(x)$ is the **density ratio**, and weighting every training compound by it makes the training average equal the test average: a training compound that is typical of the test set counts for more, one that the test set never visits counts for less. Neither density can be written
down for molecules, but their ratio can be estimated by a classifier. Let a model be trained to say, for a structure $x$, whether it comes from the test file (label 1) or the training file (label 0), and call its probability of "test" $c(x) = P(\\text{test} \\mid x)$. By Bayes' rule, with $n_{\\text{test}}$ and $n_{\\text{train}}$ the sizes of the two files,

$$\\frac{c(x)}{1 - c(x)} \\;=\\; \\frac{p_{\\text{test}}(x)\\,n_{\\text{test}}}{p_{\\text{train}}(x)\\,n_{\\text{train}}} \\quad\\Longrightarrow\\quad w(x) = \\frac{c(x)}{1 - c(x)}\\cdot\\frac{n_{\\text{train}}}{n_{\\text{test}}}.$$

The same classifier answers the first question as well: its area under the ROC curve (the probability that it scores a random test compound above a random training compound, 0.5 for no difference, defined in section 4.10) says how different the two sets are; this is sometimes called adversarial validation. Weights have a price. If a few of them are very large, a weighted average rests on a few compounds. The **effective sample size** of a set of weights,
$n_{\\mathrm{eff}} = (\\sum_i w_i)^2 / \\sum_i w_i^2$, equals the number of compounds when all the weights are equal and shrinks as they concentrate. The project therefore did not use the raw ratio. It used a tempered proxy computed from similarities alone (`testlike_weights` in `cypmode/challenge/nested_runners.py`): for a training compound $i$ let $s_i^{\\text{test}}$ be its mean
similarity to its {{=K_NEIGHBOURS}} most similar test compounds and $s_i^{\\text{train}}$ its mean similarity to its {{=K_NEIGHBOURS}} most similar other training compounds; the weight is $\\mathrm{clip}\\bigl((s_i^{\\text{test}}/s_i^{\\text{train}})^{\\alpha},\\,0.2,\\,5\\bigr)$, rescaled to a mean of 1, with the temper $\\alpha$ between 0 (no weighting)
and 1. A compound that sits among the test compounds and alone in the training set is up-weighted, and a member of a dense series that the test set hardly touches is down-weighted. Only structures enter; no label of a test compound exists. The cell computes all three weightings.
"""),
    code(ADVERSARIAL),
    md("""
A classifier separates the two files with an AUC of {{=auc_shift:.2f}}: the test compounds are different from the training compounds in a way that the structures alone reveal, which is a strong covariate shift. The raw density ratio is so concentrated that its effective sample size is {{=weight_table.iloc[0]['effective sample size']:.0f}} of {{=len(train)}}
compounds, and the tempered proxies keep {{=weight_table.iloc[1]['effective sample size']:.0f}} and {{=weight_table.iloc[2]['effective sample size']:.0f}}. Whether weighting the training compounds by such a proxy improves a model is a question that the project asked of the data and answered with a pre-declared gate (section 9.5), and
that is where its result is reported.

## 2.8 Other data sources: what exists, what each one measures, and how it differs

> **Field:** Cheminformatics · Bioinformatics

The challenge's training set holds {{=len(train)}} compounds. Other public data exist for the same enzymes, and the project looked at six of them. Four questions were asked of each. What does it measure (a yes/no call, a potency, a kinetic quantity)? How many compounds does it
hold? How different is it from the challenge compounds? Could it leak the answer, that is, hold the very compounds whose labels are being predicted (section 2.9)? The table gives the first three.
"""),
    code(SOURCES),
    md("""
- **The TDC panel** (Therapeutics Data Commons, from the cytochrome screen of Veith and co-workers, 2009, run by the National Center for Advancing Translational Sciences (NCATS) as a **quantitative high-throughput screen**, qHTS, in which every compound is tested at several concentrations) gives a yes/no inhibitor call for five enzymes on more than twelve thousand compounds each. It is the largest set, and like the PubChem screens it contains many compounds that do *not* inhibit, which the challenge's own training set, built from compounds promoted after a screen (section 2.6), does not. The project used it twice: as the data of the motif screen of Part V (`cypmode/data/panel.py` joins the labels to the motif matches), and as the training data of a **label-free** feature block (`cypmode/challenge/external_activity.py`: a model fitted on this panel only,
  with every compound that is also a challenge compound removed from its training data, scores every challenge compound with five probabilities of being an inhibitor of the five enzymes). Because no challenge label is involved, such a block cannot leak and needs no nested handling.
- **ChEMBL** (the literature database of the European Bioinformatics Institute) was read through targeted queries for CYP3A4 (target CHEMBL340) and CYP2D6 (CHEMBL289) (`cypmode/challenge/chembl_auxiliary.py`). It holds measured potencies, a pChEMBL value, which is $-\\log_{10}$ of the IC50, Ki or a similar quantity in mol/L, from the medicinal-chemistry literature, the same chemical
  space as the challenge's own series. The same compound appears in many papers, so the many records of one compound are combined by their median.
- **BindingDB** (`cypmode/challenge/bindingdb_auxiliary.py`) is another literature-curated source, of affinities in micromolar for the same two enzymes (queried by their UniProt identifiers P08684 and P10635), combined by the median in the same way.
- **PubChem** (Kim and co-workers, 2025; the loader is `cypmode/challenge/pubchem_auxiliary.py`) holds the NCATS screens: assay 1851, a five-enzyme panel with an active/inactive call and a potency, and the two single-enzyme screens, assay 884 for CYP3A4 and assay 891 for CYP2D6. The TDC panel and assay 1851 are two forms of one experiment, the NCATS screen of Veith and co-workers: {{=share_tdc_in_aid:.0%}} of the connectivity keys of the TDC compounds are also keys of assay 1851, so they are not independent evidence. A real pitfall of the file is recorded in the loader: a compound
  can have several rows for one enzyme (replicate measurements), and a naive merge of five enzymes multiplied the rows, inflating {{=len(aid):,}} distinct compounds into 86,721 rows (the figure recorded in the loader) before it was caught and fixed.
- **The Octant screen** is a diversity-library set of CYP3A4 dose-responses with a 30-minute pre-incubation released by OpenADMET, so its potency mixes reversible and time-dependent inhibition; {{=octant_validation['n_qc_passed_with_pic50']}} of its {{=len(octant)}} records pass the quality control of the curve fit.
  The project used it as an independent test of the heme-binding screen of Part V (`cypmode/data/octant_benchmark.py`), where its result is reported.
- **The public boards** of the challenge's leaderboard, saved as tables (`data/tdi_track/inputs/leaderboard_snapshots/`), carry no compound and no label, only the scores of submitted files, and are used for the one purpose of section 2.10.

None of these sources measures time-dependent inhibition as a pre-incubation shift, which is what the challenge's label is made of (section 1.8). They measure how strongly a compound inhibits (or whether it does), and are therefore, at best, correlates of the label. Their use as inputs is the subject of section 7.7, and the reading of the numbers there must keep in mind the shift between the compounds of
the sources and the challenge compounds, which the table above gives for each source as the number of compounds they share. The rest of the table is left to the experiments.

"""),
    figure("data_screen_vs_pic50"),
    figure("data_external_coverage"),
    md("""
## 2.9 Standardising structures: when two strings are the same compound

> **Field:** Cheminformatics

To remove from an external source the compounds that are also in the challenge's files, one has to decide when two strings stand for the same compound. A comparison of the strings as written finds almost nothing, because one compound has many strings (section 2.1) and a medicine is sold, and entered in
databases, as a salt: the active molecule together with a counter-ion such as chloride or a solvent such as water, written as extra components in the same string and separated by a dot. The same active molecule is then written with and without them. Four comparison rules, from the crudest to the most careful, applied to the TDC panel against all the
challenge compounds, give:
"""),
    code(STANDARDISE),
    md("""
The count rises from {{=rule_rows[0]['TDC strings matched to a challenge compound']}} (identical strings) to {{=rule_rows[1]['TDC strings matched to a challenge compound']}} (canonical SMILES) to {{=rule_rows[3]['TDC strings matched to a challenge compound']}} (standardised parent), and the connectivity key finds {{=rule_rows[2]['TDC strings matched to a challenge compound']}} because it is a different and coarser statement. The standardised parent is the
structure that remains after removing salts and solvents and neutralising charges where that is possible; the project uses the standardisation pipeline of ChEMBL itself (`chembl_structure_pipeline`, the tool with which ChEMBL curates its own structures) in `cypmode/challenge/structure_standardization.py`. A compound that cannot be read gives `None` and not an error, so that a handful of malformed rows cannot stop a pass over a
whole dataset. The {{=n_only_by_parent}} compounds found only by the parent rule are what a string comparison would have let through. The connectivity key, the first block of the InChIKey (a hash of the atoms and bonds that ignores stereochemistry, charges and isotopes), is a looser statement: it declares two stereoisomers to be one compound, which is the safe direction for removing leaks and the unsafe one for
counting distinct compounds. The project therefore uses the key where the aim is to leave compounds out (the training of `external_activity`) and the standardised parent where it checks that an external set does not contain a challenge compound (the three auxiliary sources and the similarity check of Part VI).

## 2.10 The public leaderboard as data: the positive rate of the scored set

> **Field:** Statistics

The challenge publishes, for every submitted file, the **accuracy** $A$ (the fraction of compounds called correctly), the **precision** $P$ (of the compounds called positive, the fraction that are positive) and the **recall** $R$ (of the positive compounds, the fraction that were called positive). They are quantities
of the confusion matrix of the file, whose four counts are the true positives $TP$ (called positive, are positive), the false positives $FP$ (called positive, are negative), the false negatives $FN$ (called negative, are positive) and the true negatives $TN$. Let $n$ be the number of scored compounds and $p$ the fraction of them that are positive.
Then, by the definitions, the positives are $pn = TP + FN$, recall is $R = TP/(pn)$, so $TP = Rpn$ and $FN = (1 - R)pn$. The compounds called positive are $TP + FP = TP/P = Rpn/P$, so $FP = (R/P - R)\\,pn$. The errors are $FN + FP$, and accuracy is one minus the error rate:

$$1 - A \\;=\\; \\frac{FN + FP}{n} \\;=\\; p\\,\\Bigl[(1 - R) + \\bigl(\\tfrac{R}{P} - R\\bigr)\\Bigr] \\quad\\Longrightarrow\\quad p \\;=\\; \\frac{1 - A}{1 - 2R + R/P}.$$

Three published numbers therefore give the positive rate of the set on which the file was scored, without any label being seen. Every row of a table is scored against the same labels, so the value of $p$ must be the same for all rows; the spread across rows is a check of the whole derivation. The module `cypmode/challenge/leaderboard_prior.py` implements it (`implied_prevalence`, `estimate_from_board`)
and takes the median over rows as the estimate. The cell checks the formula on synthetic sets of known positive rate and then applies it to the saved tables.
"""),
    code(LEADERBOARD),
    md("""
On synthetic data the formula returns the counted positive rate of the set to rounding, and on the real tables the estimate is the same for every row to within {{=board_table['spread of p across rows'].max():.4f}}, so that the three numbers do describe one set of labels. The two kinds of table are different: the interim table was frozen at the deadline of 24 September 2026 and is scored on the full test set, while the live table is scored on half of it (`data/tdi_track/inputs/leaderboard_snapshots/README.md`), so the estimates differ a little, and the project works with the range between them. Compared with the training compounds:
"""),
    code(PREVALENCE_COMPARE),
    md("""
The scored test set has between {{=low_high['CYP3A4'][0]:.0%}} and {{=low_high['CYP3A4'][1]:.0%}} positives on CYP3A4 (the training rate is {{=rate_view_3a4:.0%}}) and between {{=low_high['CYP2D6'][0]:.0%}} and {{=low_high['CYP2D6'][1]:.0%}} on CYP2D6, against {{=rate_both_arms:.0%}} among the CYP2D6 training compounds that have both arms. That is a large label shift on CYP2D6, and a
model whose threshold was chosen on the training rate would call far too many compounds positive; it is the reason the final model chooses its yes/no threshold by a call *rate* that is transferred to the scored population and not by a probability cut (section 8.4). It is also close to the {{=p_pos:.1%}} that CYP2D6 has among *all* compounds of the panel block, never-measured ones counted as negative (section 2.6), which is
the observation that suggested the selection-aware model of Part IX. It does not prove that the scored set is built like the panel, because the same positive rate can arise from other mixtures of hits and non-hits. The last cell asks whether the baseline's own public numbers can say what the mixture is (`cypmode/challenge/population_share.py`): the baseline's predictions on the training compounds are re-weighted for a range of mixtures, and the mixtures whose weighted accuracy, precision and recall agree with the baseline's published ones, within the
sampling uncertainty, are kept.
"""),
    code(POPULATION_SHARE),
    md("""
The data cannot say. Whatever weight $w$ is put on the never-measured compounds, from none to {{=band_summary['grid w high']}} times the training share, some choice of the second parameter fits the three published numbers, so the region of plausible mixtures spans the whole grid and does not identify the share of non-hits in the scored set (the audit of the project lists this as a failed check on purpose, `audit/AUDIT_REPORT.md`). That is a
limit of the information in the tables and not of the method, and it is why the project judges the selection-aware model over the *whole* band of shares (section 9.1) instead of choosing one, and decides between the two models for CYP2D6 by a rule fixed in advance (the final decision is in section 13.4).
"""),
    md("""
## 2.11 A check of the positive rate that reads no board

> **Field:** Statistics · Machine learning

Section 2.10 recovers the positive rate of the scored set from numbers that the boards publish. That rate sets the weights of the evaluation and the number of positive calls, so a reader may ask whether the training file and the structures of the blinded file could give the same rate without any board. They can, but only roughly, and this section shows how roughly.

**The idea.** Take a classifier that gives every compound a score $s$, higher for a positive compound than for a negative one on average. Let $m_1$ be the average score of positive compounds and $m_0$ the average score of negative compounds, both measured on compounds whose labels are known and that the classifier was not trained on. If the blinded file holds a share $\\pi$ of positives, its average score must be $\\pi m_1 + (1 - \\pi) m_0$, because the average of a mixture is the mixture of the averages. The average score $\\bar s_{\\text{blinded}}$ over the blinded file needs no label, so the equation can be solved for $\\pi$:

$$\\pi \\;=\\; \\frac{\\bar s_{\\text{blinded}} - m_0}{m_1 - m_0}.$$

This is label-shift estimation with a black-box classifier (Lipton, Wang and Smola, 2018), here in its soft form, the probabilistic adjusted classify and count (Bella and colleagues, 2010). It assumes that within each class the compounds look the same in the training file and in the blinded file, and that only the share of the classes differs. That assumption is partly false here, because the blinded compounds are analogues of training compounds, so the held-out compounds are weighted to the similarity profile of the blinded file (a function of structures only), and the folds keep chemical series whole, so that a held-out compound has no close relative in the part that the model was trained on, as is true of many blinded compounds.

Two steps are specific to this challenge. The held-out compounds and the blinded compounds must be scored on one scale, so that $m_0$ and $m_1$ mean the same on both sides; each of the fifteen fold models therefore scores both. And only confidently labelled compounds are scored (section 2.5), so the rate among all compounds is moved to the scored compounds with the share of confident compounds among the positives and among the negatives, measured on the training file. For CYP2D6 the training population is the whole panel, with the compounds that were never promoted counted as negatives (section 2.6), because the scored set holds compounds of both kinds.

**The rule, fixed before the first run.** First, a recovery test: the held-out compounds of a fold are split by series, one half fixes $m_0$ and $m_1$ and the other half is resampled to a known positive rate and scored without its labels; the estimate counts only if the recovered rates are off by at most a quarter of the rate on average. Second, the estimate is informative only if its 95% interval is at most half as wide as the rate of the final build. Third, the verdict is AGREES if that rate lies inside the interval, DISAGREES if it lies outside, and NOT INFORMATIVE if either earlier condition fails. No verdict changes a submitted file.
"""),
    code(LABEL_SHIFT),
    md("""
For CYP3A4 the estimate for the scored compounds is {{=shift['CYP3A4']['estimate']}} with the interval {{=shift['CYP3A4']['lo']}} to {{=shift['CYP3A4']['hi']}}, and the final build uses {{=shift['CYP3A4']['board']}}, which lies {{=inside['CYP3A4']}} the interval; the rule gives **{{=shift['CYP3A4']['verdict']}}** ({{=shift['CYP3A4']['reason']}}). For CYP2D6 the estimate is {{=shift['CYP2D6']['estimate']}} with the interval {{=shift['CYP2D6']['lo']}} to {{=shift['CYP2D6']['hi']}} against {{=shift['CYP2D6']['board']}}, which lies {{=inside['CYP2D6']}} the interval; the rule gives **{{=shift['CYP2D6']['verdict']}}** ({{=shift['CYP2D6']['reason']}}).

The intervals are wide for a reason that the formula shows: the denominator $m_1 - m_0$ is small, because even a good classifier gives scores that overlap between the classes, and every error in the two averages is divided by it. The training file and the structures alone therefore fix the positive rate only to within about the width of these intervals, which is why the boards remain the source of the rate that the final build uses, and why section 2.10 was written. The check is kept because it could have disagreed: a rate outside an informative interval would have been reported as a conflict.
"""),
]
