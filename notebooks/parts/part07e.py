# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VII (fifth part): learned representations, ChemBERTa and the shared-trunk network (7.9); structural evidence from Boltz-2 (7.10); the first assembled models (7.11)."""

from notebooks.notebook_builder import code, md

CHEMBERTA_FACTS = '''
# The ChemBERTa embedding of every labelled compound, as cached by cypmode/challenge/chemberta_features.py (the model seyonec/ChemBERTa-zinc-base-v1 reads the SMILES token by token and the project keeps the mean of the
# 768-number vectors of the tokens). The notebook does not load the transformer: the neural-network library cannot share a process with LightGBM (the cell above lists why), so the cache is read. Three facts: its size; how many
# directions carry most of its variance (principal components, section 4.9); and how well a linear classifier on the embedding alone separates the labels, in the five folds of 7.1, against the LightGBM on the descriptors.
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

embedding_cache = pd.read_csv(ROOT / "data/tdi_track/outputs/caches/chemberta_embedding_cache.csv")
embedding_columns = [c for c in embedding_cache.columns if c != "SMILES"]
embedded = tutorial_frame.merge(embedding_cache, on="SMILES", how="left")
usable = ~embedded[embedding_columns].isna().any(axis=1).to_numpy()
embedding_matrix = embedded[embedding_columns].to_numpy(float)
explained = np.cumsum(PCA().fit(embedding_matrix[usable]).explained_variance_ratio_)
components_90, components_99 = int(np.searchsorted(explained, 0.90) + 1), int(np.searchsorted(explained, 0.99) + 1)
stored_fixed_auc = pd.read_csv(COMPARISONS / "baseline_fixed_seed.csv").groupby("isoform")["roc_auc"].mean()
probe_rows = []
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    rows = usable & embedded[column].notna().to_numpy()
    y, X = embedded.loc[rows, column].astype(int).to_numpy(), embedding_matrix[rows]
    probability = np.zeros(len(y))
    for fit_rows, held in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        probability[held] = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)).fit(X[fit_rows], y[fit_rows]).predict_proba(X[held])[:, 1]
    probe_rows.append({"enzyme": iso, "AUC of a linear classifier on the embedding alone": roc_auc_score(y, probability), "AUC of LightGBM on the descriptors (7.1)": float(stored_fixed_auc[iso])})
probe_table = pd.DataFrame(probe_rows)
show_table(probe_table, floatfmt=".3f")
n_embedded, n_dims, n_unusable = len(embedding_cache), len(embedding_columns), int((~usable).sum())
print(f"{n_embedded:,} embeddings of {n_dims} numbers; {n_unusable} labelled compounds without one; {components_90} components hold 90 % of the variance, {components_99} hold 99 %")
'''

CHEMBERTA_TESTS = '''
# The two experiments with the embedding, from their stored per-fold scores. First the concatenation to M3 for LightGBM (the repeated-partition design of 6.6, chemberta_significance_raw.csv), re-analysed with the paired
# corrected t. Then the fusion network (cypmode/challenge/multitask.py): one small branch for the M3 columns and one for the embedding, joined before the shared trunk, against the same trunk on M3 alone; the two runs
# use the same compounds and folds, so the five fold scores are paired (J = 5).
from cypmode.challenge.model_comparison import corrected_resampled_t
from cypmode.challenge.stacking_significance import paired_comparison_summary

chemberta_rows = []
raw = pd.read_csv(COMPARISONS / "chemberta_significance_raw.csv")
for iso, part in raw.groupby("isoform"):
    summary = paired_comparison_summary(part)
    row = summary[(summary["group1"] == "M3") & (summary["group2"] == "M3+ChemBERTa")].iloc[0]
    chemberta_rows.append({"experiment": "LightGBM, embedding concatenated to M3 (5 x 5)", "enzyme": iso, "MCC without": float(part[part["variant"] == "M3"]["mcc"].mean()), "MCC with": float(part[part["variant"] == "M3+ChemBERTa"]["mcc"].mean()),
                           "change": row["meandiff"], "p (corrected t)": row["p-adj"]})
trunk_alone, trunk_fusion = pd.read_csv(COMPARISONS / "shared_trunk_m1.csv"), pd.read_csv(COMPARISONS / "shared_trunk_m1_chemberta_fusion.csv")
for iso in TDI_ISOFORMS:
    a, b = (t[t["isoform"] == iso].sort_values("fold")["mcc"].to_numpy() for t in (trunk_alone, trunk_fusion))
    result = corrected_resampled_t(b - a, 4, 1)
    chemberta_rows.append({"experiment": "shared-trunk network, fusion branch for the embedding (5 folds)", "enzyme": iso, "MCC without": float(a.mean()), "MCC with": float(b.mean()), "change": result.mean_difference, "p (corrected t)": result.p_value})
chemberta_table = pd.DataFrame(chemberta_rows)
show_table(chemberta_table, floatfmt=".3f")
concat_3a4 = chemberta_table[(chemberta_table["experiment"].str.startswith("LightGBM")) & (chemberta_table["enzyme"] == "CYP3A4")].iloc[0]
fusion_rows = chemberta_table[chemberta_table["experiment"].str.startswith("shared")].set_index("enzyme")
'''

SHARED_TRUNK = '''
# The shared-trunk network against the LightGBM it was meant to challenge, from the stored per-fold tables. The network has one trunk and one head per enzyme and a loss that counts only the labelled enzymes of a compound
# (a masked binary cross-entropy; section 4.7); the LightGBM has one independent model per enzyme. "tuned" is the threshold chosen inside the training part (7.5). Two feature sets: M3 with the amine flag of the day, and M3
# without it. The folds of the network are those of the union of the compounds, the folds of the LightGBM those of each enzyme, so the two are not paired: the table gives the mean and the spread over five folds and no p-value.
def fold_summary(path, columns, label, feature_set, isoform=None):
    table = pd.read_csv(COMPARISONS / path)
    if isoform is not None:
        table["isoform"] = isoform                                           # the per-enzyme files do not carry the column
    return [{"model": label, "features": feature_set, "enzyme": iso, **{f"{c} mean": float(part[c].mean()) for c in columns}, **{f"{c} sd": float(part[c].std()) for c in columns}} for iso, part in table.groupby("isoform")]
tuning_columns = ["mcc_naive_0.5", "mcc_tuned"]
architecture_rows = (fold_summary("threshold_tuning.csv", tuning_columns, "independent LightGBM", "M3 with the amine flag")
                     + fold_summary("shared_trunk_m1_threshold_tuning.csv", tuning_columns, "shared-trunk network", "M3 with the amine flag")
                     + fold_summary("../cyp3a4/threshold_tuning_cyp3a4_no_amine.csv", tuning_columns, "independent LightGBM", "M3 without the flag", "CYP3A4")
                     + fold_summary("../cyp2d6/threshold_tuning_cyp2d6_no_amine.csv", tuning_columns, "independent LightGBM", "M3 without the flag", "CYP2D6")
                     + fold_summary("shared_trunk_threshold_tuning_no_amine.csv", tuning_columns, "shared-trunk network", "M3 without the flag"))
architecture_table = pd.DataFrame(architecture_rows)
show_table(architecture_table[["model", "features", "enzyme", "mcc_naive_0.5 mean", "mcc_naive_0.5 sd", "mcc_tuned mean", "mcc_tuned sd"]], floatfmt=".3f")
def architecture_value(model, features, iso, column="mcc_tuned mean"):
    return float(architecture_table[(architecture_table["model"] == model) & (architecture_table["features"] == features) & (architecture_table["enzyme"] == iso)][column].iloc[0])
'''


TIER3_SELECTION = '''
# Which 50 compounds were worth a structure prediction. Each Boltz-2 run (Part XI) takes 15 to 30 minutes, so the 3,584 labelled CYP3A4 compounds could not all be done. The project ranks every labelled compound by its out-of-fold
# uncertainty, |p - 0.5| of a LightGBM trained on the other folds (cypmode/challenge/tier3_selection.py, the same out-of-fold discipline as the bridge of 7.4), and takes the 50 most uncertain: the compounds on which a
# structure has the most to add. The cell redoes the ranking, compares it with the list that was submitted to the structure pipeline, and checks the list against the blinded test file.
from cypmode.challenge.tier3_selection import select_fixed_budget_oof

redone = select_fixed_budget_oof(described, "CYP3A4_is_TDI", budget=50)
stored_selection = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp3a4/tier3_selected_compounds.csv")
structural = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp3a4/tier3_structural_results.csv")
test_names, test_smiles = set(test_file["Molecule_Name"]), set(test_file["SMILES"])
selection_rows = [
    {"quantity": "compounds selected", "value": len(stored_selection)},
    {"quantity": "of which the redone ranking also selects", "value": int(len(set(stored_selection["Molecule_Name"]) & set(redone["Molecule_Name"])))},
    {"quantity": "share of the labelled CYP3A4 compounds", "value": len(stored_selection) / int(tutorial_frame["CYP3A4_is_TDI"].notna().sum())},
    {"quantity": "largest |p - 0.5| among the selected (out of fold)", "value": float(redone["oof_uncertainty"].max())},
    {"quantity": "positives among the selected", "value": int(redone.merge(stored_selection[["Molecule_Name"]], on="Molecule_Name")["CYP3A4_is_TDI"].sum())},
    {"quantity": "selected compounds that are also in the blinded test file (by name)", "value": int(stored_selection["Molecule_Name"].isin(test_names).sum())},
    {"quantity": "selected compounds that are also in the blinded test file (by structure)", "value": int(stored_selection["SMILES"].isin(test_smiles).sum())},
    {"quantity": "structure predictions that finished without an error", "value": int(structural["error"].isna().sum())},
    {"quantity": "of which the nitrogen of a Type II motif is within bonding distance of the iron", "value": int(structural["coordinated"].fillna(False).astype(bool).sum())},
    {"quantity": "of which a motif nitrogen exists at all", "value": int(structural["closest_motif_nitrogen_distance_angstrom"].notna().sum())},
]
show_table(pd.DataFrame(selection_rows), floatfmt=".3f")
selection_values = {row["quantity"]: row["value"] for row in selection_rows}
'''

TIER3_TEST = '''
# The test the selection allows. The structural columns exist only for the 50 compounds, so a model that is given them can only differ from the plain one on those, and the comparison is scored on the validation compounds
# of each fold that are among the 50 (about ten per fold; cypmode/challenge/tier3_features.py, repeated 5 x 5 as in 6.6). The stored per-fold scores are re-analysed with the paired corrected t, and the smallest change that so
# small a design could have found is computed with the formula of 6.7.
from cypmode.challenge.model_comparison import corrected_resampled_t

tier3_raw = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp3a4/tier3_significance_raw.csv")
paired = tier3_raw.pivot_table(index=["repeat", "fold"], columns="variant", values="mcc").dropna()
tier3_difference = (paired["M3+Tier3"] - paired["M3"]).to_numpy()
tier3_result = corrected_resampled_t(tier3_difference, 4, 1)
tier3_sd = float(np.std(tier3_difference, ddof=1))
tier3_floor = float((stats.t.ppf(0.975, len(tier3_difference) - 1) + stats.t.ppf(0.8, len(tier3_difference) - 1)) * tier3_sd * np.sqrt(1 / len(tier3_difference) + 1 / 4))
n_tier3_per_fold = tier3_raw.drop_duplicates(["repeat", "fold"])["n_val_tier3"]
tier3_table = pd.DataFrame([{"quantity": "folds with scorable compounds", "value": len(tier3_difference)}, {"quantity": "validation compounds of the 50 per fold, median", "value": float(n_tier3_per_fold.median())},
                            {"quantity": "mean MCC, M3 alone", "value": float(paired["M3"].mean())}, {"quantity": "mean MCC, M3 with the structural columns", "value": float(paired["M3+Tier3"].mean())},
                            {"quantity": "sd of the per-fold MCC (M3)", "value": float(paired["M3"].std())}, {"quantity": "mean change", "value": tier3_result.mean_difference},
                            {"quantity": "p (corrected t)", "value": tier3_result.p_value}, {"quantity": "smallest change found 8 times in 10", "value": tier3_floor}])
show_table(tier3_table, floatfmt=".3f")
tier3_values = {row["quantity"]: row["value"] for row in tier3_table.to_dict("records")}
# The two weak blocks together. The structural columns of this section and the read-across columns of 7.3 were each null on their own; a combination of two weak signals can be worth more than either, so the pair was
# tested as well (cypmode/challenge/tier3_knn_significance.py): the four variants M3, M3 + the structural columns, M3 + the read-across columns and M3 + both, on the Tier 3-covered validation rows of a five-fold
# cross-validation repeated five times, compared pairwise with Tukey's HSD. The cell reads the comparison the module wrote for each enzyme.
tier3_knn_tukey = {iso: pd.read_csv(ROOT / f"data/tdi_track/outputs/{iso.lower()}/tier3_knn_significance_tukey.csv") for iso in ("CYP3A4", "CYP2D6")}
n_tier3_knn_pairs = sum(len(tukey_frame) for tukey_frame in tier3_knn_tukey.values())
n_tier3_knn_rejected = sum(int(tukey_frame["reject"].sum()) for tukey_frame in tier3_knn_tukey.values())
smallest_tier3_knn_p = min(float(tukey_frame["p-adj"].min()) for tukey_frame in tier3_knn_tukey.values())
'''



ARCH_PAIRED = '''
# Shared-trunk network against independent LightGBM for CYP2D6, on equal terms. cypmode/challenge/architecture_significance.py scores both models on exactly the same held-out CYP2D6 compounds in every fold of a 5 x 5
# design (the network is trained on all the CYP3A4 labels plus the CYP2D6 labels of the training part, the LightGBM on the CYP2D6 labels alone), and cyp2d6_tuning_significance.py tries three changes to the network (class weights, a
# smaller network with more dropout, an average over seeds) in the same design. The stored per-fold scores are re-analysed with the paired corrected t of 6.5.
from cypmode.challenge.stacking_significance import paired_comparison_summary

architecture_raw = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp2d6/architecture_significance_raw.csv")
architecture_pair = paired_comparison_summary(architecture_raw).iloc[0]
architecture_means = architecture_raw.groupby("variant")["mcc"].agg(["mean", "std"])
architecture_diff = (architecture_raw[architecture_raw["variant"] == "Shared-trunk"].sort_values(["repeat", "fold"])["mcc"].to_numpy() - architecture_raw[architecture_raw["variant"] == "Independent-head"].sort_values(["repeat", "fold"])["mcc"].to_numpy())
architecture_floor = float((stats.t.ppf(0.975, 24) + stats.t.ppf(0.8, 24)) * np.std(architecture_diff, ddof=1) * np.sqrt(1 / 25 + 1 / 4))
tuning_raw = pd.read_csv(ROOT / "data/tdi_track/outputs/cyp2d6/tuning_significance_raw.csv")
tuning_means = tuning_raw.groupby("variant")["mcc"].agg(["mean", "std"])
tuning_summary = paired_comparison_summary(tuning_raw)
arch_table = pd.concat([architecture_means.assign(experiment="architecture (5 x 5)"), tuning_means.assign(experiment="levers of the network (5 x 5)")]).reset_index().rename(columns={"mean": "mean MCC", "std": "sd over the 25 folds"})
show_table(arch_table[["experiment", "variant", "mean MCC", "sd over the 25 folds"]], floatfmt=".3f")
show_table(pd.concat([pd.DataFrame([architecture_pair]).assign(experiment="architecture"), tuning_summary[tuning_summary["group1"] == "Baseline"].assign(experiment="levers")])[["experiment", "group1", "group2", "meandiff", "p-adj", "lower", "upper"]], floatfmt=".3f")
lever_best_p = float(tuning_summary["p-adj"].min())
lever_largest = float(tuning_summary[tuning_summary["group1"] == "Baseline"]["meandiff"].abs().max())
'''


MIXED_COST = '''
# What the mixed population costs. The 2,334 CYP3A4 compounds with both arms measured are the population the test set resembles (section 2.4). The same LightGBM on the descriptors is trained twice, in five folds of those compounds:
# on the training compounds of the fold alone, and on them together with the 1,249 compounds that have no direct reading (all labelled negative by the convention of 2.3). Both are scored on the same validation compounds of the fold,
# which are always compounds with both arms.
from cypmode.challenge.tdi_targets import BOTH_ARMS, NO_DIRECT, training_view

population = training_view(train_file, "CYP3A4")["population"]
population_by_name = dict(zip(train_file["Molecule_Name"], population))
frame_3a4 = described[described["CYP3A4_is_TDI"].notna()].reset_index(drop=True)
kinds = frame_3a4["Molecule_Name"].map(population_by_name).to_numpy()
rows_both, rows_no_direct = np.where(kinds == BOTH_ARMS)[0], np.where(kinds == NO_DIRECT)[0]
X_3a4, y_3a4 = np.stack(frame_3a4["descriptors"]), frame_3a4["CYP3A4_is_TDI"].astype(int).to_numpy()
mixed_rows = []
for fold, (fit_pos, held_pos) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(rows_both, y_3a4[rows_both])):
    fit_both, held = rows_both[fit_pos], rows_both[held_pos]
    for name, fit_rows in (("trained on the corrected population", fit_both), ("trained on the mixed population", np.concatenate([fit_both, rows_no_direct]))):
        probability = LGBMClassifier(verbose=-1, random_state=42).fit(X_3a4[fit_rows], y_3a4[fit_rows]).predict_proba(X_3a4[held])[:, 1]
        mixed_rows.append({"version": name, "fold": fold, "AUC": roc_auc_score(y_3a4[held], probability)})
mixed_scores = pd.DataFrame(mixed_rows)
mixed_table = mixed_scores.groupby("version", sort=False)["AUC"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "mean AUC over the 5 folds", "std": "sd over the folds"})
show_table(mixed_table, floatfmt=".3f")
mixed_gain = float(mixed_table.iloc[0]["mean AUC over the 5 folds"] - mixed_table.iloc[1]["mean AUC over the 5 folds"])
mixed_paired = corrected_resampled_t(mixed_scores[mixed_scores["version"] == "trained on the corrected population"].sort_values("fold")["AUC"].to_numpy() - mixed_scores[mixed_scores["version"] == "trained on the mixed population"].sort_values("fold")["AUC"].to_numpy(), 4, 1)
'''


VII_LEDGER = '''
# Part VII in one table: each idea, the number that decided it (read from the cells of this part), and where it stands in the final model. Nothing in the last column is a verdict of this table; it names the section of Part VIII to X where
# the idea is judged again, inside the stack, by its gate.
ledger_vii = pd.DataFrame([
    {"section": "7.1", "idea": "the challenge's LightGBM on RDKit descriptors", "evidence": f"CV MCC {baseline_value('fixed-seed five-fold, run here', 'CYP3A4', 'MCC mean'):.3f} and {baseline_value('fixed-seed five-fold, run here', 'CYP2D6', 'MCC mean'):.3f}; the board has {board_table.loc['CYP3A4', 'board MCC']:.3f} and {board_table.loc['CYP2D6', 'board MCC']:.3f}", "fate": "the reference of every later number"},
    {"section": "7.2", "idea": "motif flag, amine flag, Morgan bits", "evidence": f"amine step on CYP2D6: {stored_amine_step['mean MCC change']:+.3f} stored, {today_amine_step['mean MCC change']:+.3f} with today's pattern, {cyp2d6_partitions['mean over the 25 folds']:+.3f} over 5 partitions (p = {cyp2d6_partitions['p (corrected t, J = 25)']:.2f})", "fate": "motif block in A+fps; amine flag unused"},
    {"section": "7.3", "idea": "labels of the 10 nearest neighbours", "evidence": f"5 x 5: CYP3A4 {knn_range['CYP3A4'][0]:+.3f} to {knn_range['CYP3A4'][1]:+.3f}, CYP2D6 {knn_range['CYP2D6'][0]:+.3f} to {knn_range['CYP2D6'][1]:+.3f}; smallest p {knn_smallest_p:.2f}", "fate": "two stack members (8.1)"},
    {"section": "7.4", "idea": "predicted direct-inhibition pIC50 (the bridge)", "evidence": f"AUC {bridge_values.loc[v_naive, 'AUC']:.3f} in the naive evaluation, {bridge_values.loc[v_deploy, 'AUC']:.3f} as the test compounds will be seen; {corrected_population_test.mean_difference:+.4f} on the corrected population (p = {corrected_population_test.p_value:.2f})", "fate": "retracted; a shortcut"},
    {"section": "7.5", "idea": "threshold tuned inside the training part", "evidence": f"CYP3A4 MCC {rule_value('CYP3A4', 'fixed 0.5'):.3f} at 0.5, {rule_value('CYP3A4', 'tuned on the inner out-of-fold probabilities (the project rule)'):.3f} tuned inside the folds", "fate": "became the transfer of a call rate (8.4)"},
    {"section": "7.6", "idea": "query by committee as uncertainty", "evidence": f"best gain over the margin {best_committee_gain:+.3f} AUROC; {n_used} components may alter a prediction", "fate": "reported, never used to change a call"},
    {"section": "7.7", "idea": "AID 1851, BindingDB, ChEMBL, AID 884 and 891", "evidence": f"single partition up to {largest_single['change']:+.3f}; 5 x 5 smallest p {smallest_repeated_p:.2f}", "fate": "the TDC panel as the block ext_activity (3 members)"},
    {"section": "7.8", "idea": "GFN2-xTB numbers, qmdesc reactivity, pharmacophore fingerprint", "evidence": f"5 x 5 changes within {max(abs(change_range[0]), abs(change_range[1])):.3f} MCC; smallest p {smallest_null_p:.2f}", "fate": "not used; physics returns in Part X"},
    {"section": "7.9", "idea": "ChemBERTa embedding, shared-trunk network", "evidence": f"concatenated to M3: {concat_3a4['change']:+.3f} on CYP3A4 (p = {concat_3a4['p (corrected t)']:.2f}); fused: {fusion_rows.loc['CYP3A4', 'change']:+.3f}", "fate": "not used; the neural family returns in Part IX"},
    {"section": "7.10", "idea": "Boltz-2 structure for 50 compounds", "evidence": f"change {tier3_values['mean change']:+.3f} (p = {tier3_values['p (corrected t)']:.2f}); {selection_values['selected compounds that are also in the blinded test file (by name)']} of the 50 are test compounds", "fate": "not used; structure prediction is Part XI"},
])
show_table(ledger_vii)
n_ideas = len(ledger_vii)
'''


CELLS = [
    md("""
## 7.9 Learned representations: ChemBERTa embeddings and a shared-trunk network

> **Field:** Machine learning · Cheminformatics

*Where this section sits:* the ideas of 7.2 to 7.8 were columns that someone had designed, a flag, a neighbour's label, a quantum number, given to a tree model. When eight such additions had come back null, the project turned to the other family of ideas, in which the model learns the representation itself. Two experiments follow. In the first, a transformer that has read tens of millions of SMILES strings turns each compound into a vector of 768 numbers, and the vector is given to the classifier as features. In the second, a small neural network with one shared trunk and one head per enzyme is trained on both enzymes at once, alone and with the transformer's vector fed through a branch of its own. Part IV described both as models (sections 4.7 and 4.8); this section reports what they did here. The motive came from outside: the published CYP-specific models that the project's notes cite combine several learned representations (a fingerprint with a graph network in one, a graph, a convolution and a transformer in another), and the retrospective of a related challenge found classical gradient-boosted models ranked low among the best entries.

**The embedding.** ChemBERTa (the model `seyonec/ChemBERTa-zinc-base-v1`) is a RoBERTa transformer pretrained on about 77 million SMILES strings from the ZINC collection. It reads a SMILES token by token; the project keeps the mean of the vectors of the tokens, 768 numbers per compound. It needs no three-dimensional shape and so has no way of failing on an awkward molecule, and it takes about 35 milliseconds per compound on a processor (a few minutes for the training file). The extraction runs in a process of its own, because the neural-network library and LightGBM cannot share a process: the program ends with a segmentation fault (the same conflict as the one of 7.8). `cypmode/challenge/chemberta_features.py` extracts and caches, and `cypmode/challenge/chemberta_significance.py` evaluates in a process that never imports the library. This notebook reads the cache. The first cell asks three things of it: how many directions carry its variance (the principal components of section 4.9), and how well a linear classifier on the embedding alone separates the labels.
"""),
    code(CHEMBERTA_FACTS),
    md("""
The cache holds {{=n_embedded:,}} embeddings of {{=n_dims}} numbers, with {{=n_unusable}} labelled compounds missing. The 768 numbers are far from independent: {{=components_90}} principal components hold 90 percent of their variance and {{=components_99}} hold 99 percent, so the embedding is a thing of a few dozen dimensions written in 768. It also carries signal. A linear classifier on it alone reaches an AUC of {{=probe_table.iloc[0]['AUC of a linear classifier on the embedding alone']:.3f}} for CYP3A4 and {{=probe_table.iloc[1]['AUC of a linear classifier on the embedding alone']:.3f}} for CYP2D6, against {{=probe_table.iloc[0]['AUC of LightGBM on the descriptors (7.1)']:.3f}} and {{=probe_table.iloc[1]['AUC of LightGBM on the descriptors (7.1)']:.3f}} for LightGBM on the descriptors: a representation that has learned something about which compounds inhibit, though less than the descriptors know, and for CYP2D6 hardly more than chance in either case.

**What it did as features, and as a branch.** The next cell collects the two experiments. In the first, the 768 numbers are appended to M3 and handed to LightGBM, in the repeated-partition design of 6.6. In the second, the network of `cypmode/challenge/multitask.py` (a branch of one linear layer for the M3 columns and one for the embedding, joined before the shared trunk) is compared with the same trunk on M3 alone, on the same five folds.
"""),
    code(CHEMBERTA_TESTS),
    md("""
Appending the embedding lowers the CYP3A4 MCC from {{=concat_3a4['MCC without']:.3f}} to {{=concat_3a4['MCC with']:.3f}}, a change of {{=concat_3a4['change']:+.3f}} with a corrected p-value of {{=concat_3a4['p (corrected t)']:.2f}}. The record of the day had called the loss significant (Tukey's HSD, p = {{=example_tukey_p:.4f}}), and 6.6 showed why it was not established. A fusion branch does not rescue it: the network with the embedding scores {{=fusion_rows.loc['CYP3A4', 'change']:+.3f}} against the network without on CYP3A4 and {{=fusion_rows.loc['CYP2D6', 'change']:+.3f}} on CYP2D6, with corrected p-values of {{=fusion_rows.loc['CYP3A4', 'p (corrected t)']:.2f}} and {{=fusion_rows.loc['CYP2D6', 'p (corrected t)']:.2f}}. The point estimates fall in the same direction in the two designs and neither is established. The project's reading of the first, which it did not test, was that a tree model has no learned projection to compress a dense, highly correlated block, and spends its splits on noisy directions of the embedding; the fusion network exists to answer that, and its answer, within the noise, is that the embedding adds nothing there either. A simpler reading is also consistent with the numbers: what the embedding knows, the descriptors and the fingerprint already know.

**The network on its own.** The shared-trunk network of `multitask.py` has a trunk of two linear layers (64 units) shared by the two enzymes and one small head per enzyme, and its loss is a binary cross-entropy summed over the labelled enzymes of each compound only, so that a compound with a CYP3A4 label and none for CYP2D6 trains the CYP3A4 head and the trunk and sends no gradient to the other head. The idea is the one of multitask learning: the enzymes share chemistry, and the better-populated CYP3A4 labels may regularise the representation the CYP2D6 head reads. The hypothesis was benchmarked and not assumed. The next cell sets the stored five-fold results of the network beside those of independent LightGBM models, with the threshold of 0.5 and with the threshold tuned inside the training part (7.5), on M3 with the amine flag of the day and on M3 without it.
"""),
    code(SHARED_TRUNK),
    md("""
For CYP3A4 the independent LightGBM wins in every row: {{=architecture_value('independent LightGBM', 'M3 without the flag', 'CYP3A4'):.3f}} against {{=architecture_value('shared-trunk network', 'M3 without the flag', 'CYP3A4'):.3f}} for the network, on M3 without the flag with the tuned threshold, and {{=architecture_value('independent LightGBM', 'M3 with the amine flag', 'CYP3A4'):.3f}} against {{=architecture_value('shared-trunk network', 'M3 with the amine flag', 'CYP3A4'):.3f}} with it. For CYP2D6 the order is the opposite: {{=architecture_value('shared-trunk network', 'M3 without the flag', 'CYP2D6'):.3f}} for the network against {{=architecture_value('independent LightGBM', 'M3 without the flag', 'CYP2D6'):.3f}} for the independent model. The project of the day read this as a clean split, independent heads for CYP3A4 and the shared trunk for CYP2D6, and built its first assembled models on it (7.11). Two cautions belong here. The folds of the network are those of all the compounds and the folds of the LightGBM are those of each enzyme, so that the two sets of numbers are not paired and a difference of 0.07 to 0.1 is one to two standard deviations of a fold; 7.11 gives the properly paired test for CYP2D6. And the network's numbers changed once during the work, for a reason worth keeping: the audit of September found that two of its four cross-validation functions standardised the features with the mean and the spread of the whole data set and not of the training part, a leak of scale and not of labels, and the numbers were regenerated after the fix. A discrepancy between two stored files was first put down to the non-determinism of the network; it was checked and was not that. It was the stale file of the old code being compared with the new, and the network, run twice, gives byte-identical files. The explanation that sounds exotic is not evidence for a discrepancy that a simpler known cause explains completely.

*On what actually ran.* `cypmode/challenge/chemberta_features.py` (extraction and cache; `load_cached_chemberta_embeddings` is the loader that never imports the neural-network library), `chemberta_significance.py` (the evaluation of the concatenation, in a LightGBM-only process), `multitask.py` (`SharedTrunkModel`, `MultiBranchSharedTrunkModel`, the masked loss, and the cross-validation functions with and without threshold tuning; the tests of the module must run in a process of their own, as `pyproject.toml` records) and `tests/test_multitask_no_standardization_leak.py` (a test of the leak fix that was itself checked by breaking the code). The stored tables are `shared_trunk_m1.csv`, `shared_trunk_m1_chemberta_fusion.csv`, `shared_trunk_m1_threshold_tuning.csv` and `shared_trunk_threshold_tuning_no_amine.csv`. These experiments are on the mixed population of section 2.4.
"""),
    md("""
## 7.10 Does a predicted structure help a classifier? Boltz-2 for fifty compounds

> **Field:** Structural biology · Machine learning

*Where this section sits:* Part V checked the nitrogen screen against predicted structures: a nitrogen that sits within bonding distance of the heme iron in a Boltz-2 complex is a Type II binder (sections 5.5 and 1.9). A structure-based fact is the one kind of evidence that none of the fingerprints or descriptors of Part III can carry, and it was the founding idea of the project. This section asks the direct question of a classifier: if the structure of a compound bound to the enzyme is predicted, and three numbers are read from it (is a nitrogen of a motif coordinated to the iron, how far is the nearest, how confident is the prediction), do they improve the prediction of time-dependent inhibition? The structure predictions are the subject of Part XI; here they are the input.

**A budget, and a selection.** One Boltz-2 prediction of a compound with the heme takes 15 to 30 minutes on the machine of the project, so that the 3,584 labelled CYP3A4 compounds could not all be done. The project fixed a budget of 50 compounds and chose them where a structure had the most to add, the compounds on which the sequence-based model was most uncertain. The ranking is out of fold, each compound's uncertainty $|p - 0.5|$ coming from a LightGBM that never saw it (`cypmode/challenge/tier3_selection.py`; an earlier plan, to take every compound that either carries a motif or is uncertain, would have selected between 46 and 72 percent of the file and was dropped for that reason). The cell redoes the ranking, compares it with the list sent to the structure pipeline and checks the list against the blinded test file.
"""),
    code(TIER3_SELECTION),
    md("""
The ranking is redone and agrees with the stored list on {{=selection_values['of which the redone ranking also selects']}} of the {{=selection_values['compounds selected']}} compounds, {{=selection_values['share of the labelled CYP3A4 compounds']:.1%}} of the labelled CYP3A4 compounds, all of them within {{=selection_values['largest |p - 0.5| among the selected (out of fold)']:.3f}} of the decision boundary; {{=selection_values['positives among the selected']}} are positives. {{=selection_values['structure predictions that finished without an error']}} of the structure predictions finished without an error, and in {{=selection_values['of which the nitrogen of a Type II motif is within bonding distance of the iron']}} of them a motif nitrogen is within bonding distance of the iron; {{=selection_values['of which a motif nitrogen exists at all']}} have a motif nitrogen at all. And the list has {{=selection_values['selected compounds that are also in the blinded test file (by name)']}} compounds in common with the blinded test file, by name and by structure. That last number matters more than any other in this section, and it is explained below. Now the test the selection allows: three columns (coordinated or not, the distance to the nearest motif nitrogen with a far-away sentinel when there is none, the confidence of the prediction) are given to LightGBM together with M3, empty for every compound that is not among the 50, and the comparison is scored on the validation compounds of each fold that are among the 50, about ten per fold, in the 5 x 5 design.
"""),
    code(TIER3_TEST),
    md("""
On the {{=int(tier3_values['folds with scorable compounds'])}} folds that have compounds among the 50 the MCC is {{=tier3_values['mean MCC, M3 alone']:.3f}} for M3 alone and {{=tier3_values['mean MCC, M3 with the structural columns']:.3f}} with the structural columns, a change of {{=tier3_values['mean change']:+.3f}} with a corrected p-value of {{=tier3_values['p (corrected t)']:.2f}}. The spread of the per-fold MCC is {{=tier3_values['sd of the per-fold MCC (M3)']:.2f}}, as it must be when a fold has ten compounds, and the smallest change that this design could have found 8 times in 10 is {{=tier3_values['smallest change found 8 times in 10']:.2f}} MCC, larger than any MCC difference this part has met. The test is a null, and a null with almost no power: it says that the structural columns do not rescue the 50 compounds, and not that they could not help a larger set.

**The two weak blocks together.** The structural columns of this section and the read-across columns of 7.3 each came back null or negative on their own, and a pair of weak signals can be worth more than either. `cypmode/challenge/tier3_knn_significance.py` asks that question for both enzymes, on the same Tier 3-covered rows and in the same repeated design: four variants — M3, M3 with the structural columns, M3 with the read-across columns and M3 with both — are compared pairwise with Tukey's HSD. Of the {{=n_tier3_knn_pairs}} pairwise comparisons {{=n_tier3_knn_rejected}} reject the null of no difference, and the smallest adjusted p-value is {{=smallest_tier3_knn_p:.2f}}; the combination is no more visible than either block alone, at the same restricted power and for the same reason.

Two reasons make the question moot for the competition, and they agree. The statistical one is the one just given. The structural one is that the 50 compounds were chosen among the *training* compounds and the list has none in common with the blinded test file: the three columns would be empty for all 750 test compounds. A feature that does not exist for the compounds to be scored is not a feature, however well it works on the compounds that have it, and that is the first of the four rules of 7.4 again. The prediction of structures nevertheless became one of the two tracks of the challenge that the project entered, and Part XI is about it; what it asked of the structure was different (the pose of a ligand, scored against a crystal structure) and what it found there is the content of that part.

*On what actually ran.* `cypmode/challenge/tier3_selection.py` (`select_fixed_budget_oof` is the ranking that was used; `compute_selection` is the earlier fold-local policy, kept with its tests, one of which flips the validation labels and confirms that the policy does not change), `tier3_features.py` (the join of the structure results to the feature table, the sentinel for "no motif nitrogen", and the repeated test scored only on the covered compounds), and the structure pipeline of `cypmode/validation/` that produced `tier3_structural_results.{json,csv,parquet}`. The CYP2D6 analogue (`cyp2d6/`) was run the same way and gave the same kind of null. `cypmode/challenge/tier3_knn_significance.py` tests the two structural blocks together and writes its per-fold table and its Tukey comparison for each enzyme; the notebook reads the comparison. These experiments are on the mixed population of section 2.4.
"""),
    md("""
## 7.11 The first assembled models, the architecture test, and the decision to build the nested evaluation

> **Field:** Machine learning · Statistics

*Where this section sits:* by the end of September the project had a list of things that did not work and a few that were not yet shown not to. It assembled a first complete model from them, the one that wrote the first valid submission file, tested the one architectural choice it rested on, and then, in the first days of October, found three things about its own evaluation that led to Part VI. This section tells that sequence and closes Part VII with a table of everything the part tried.

**The first assembled models.** `cypmode/challenge/final_assembly.py` combined what the evidence of the day supported, per enzyme. CYP3A4 was an independent LightGBM on M3 without the amine flag, with the four read-across columns of 7.3 built for the test compounds from the whole training file and for the training compounds inside the training file leave-one-out, and a threshold tuned from the inner out-of-fold probabilities of the training file (7.5). CYP2D6 was the shared-trunk network of 7.9 on M3 without the flag, run in its own process by `final_assembly_cyp2d6.py` and read from its cache, the two enzymes' calls being merged and checked by the submission validator (`cypmode/challenge/submission.py`: the four columns, 750 rows, boolean calls). The choice for CYP2D6 rested on the single-partition comparison of 7.9, and the module that tests it properly, `cypmode/challenge/architecture_significance.py`, was written because the two stored files were not paired: it scores both models on exactly the same held-out CYP2D6 compounds in every fold of the 5 x 5 design. A second module, `cyp2d6_tuning_significance.py`, tries three changes of the network in the same design after the eight feature ideas had come back null (class weights, a smaller network with more dropout, an average over seeds). The cell reads both.
"""),
    code(ARCH_PAIRED),
    md("""
Over 25 folds the shared-trunk network scores {{=architecture_means.loc['Shared-trunk', 'mean']:.3f}} on CYP2D6 and the independent LightGBM {{=architecture_means.loc['Independent-head', 'mean']:.3f}}, a paired difference of {{=architecture_pair['meandiff']:+.3f}} with an interval of {{=architecture_pair['lower']:+.3f}} to {{=architecture_pair['upper']:+.3f}} and a corrected p-value of {{=architecture_pair['p-adj']:.3f}}. The single-partition numbers of 7.9 had suggested a gap of that size and the paired design confirms its size and not its certainty: the interval includes zero, and a design of this kind recognises a true difference of {{=architecture_floor:.2f}} MCC or more. The choice of the shared trunk for CYP2D6 was therefore a reasonable bet, supported by a suggestive result, and not a finding. The three changes of the network are indistinguishable from the baseline and from one another: the largest of the changes is {{=lever_largest:.3f}} MCC and the smallest adjusted p-value is {{=lever_best_p:.2f}}.

**What was found on 2026-10-04.** Three facts about the evaluation, each verified against the training file and the public boards, ended the period of this part, and each has a section in Part VI. The label is a deterministic function of two columns that the models had never used (2.3). A third of the CYP3A4 rows follow a different convention from the scored set, and every CYP3A4 number of this part had been computed on the mixture (2.4, and 7.4 for what it did). And a random fold is not like the test set: the blinded compounds are analogues of the top screening hits, they sit closer to the training file than a fold does, and the positive rates differ (6.4, 6.8, 2.10). The cell below measures the second of them, on the descriptors and the LightGBM of 7.1.
"""),
    code(MIXED_COST),
    md("""
Trained on the compounds with both arms, the model scores an AUC of {{=mixed_table.iloc[0]['mean AUC over the 5 folds']:.3f}} on the validation compounds with both arms; trained with the {{=len(rows_no_direct):,}} compounds of the other convention added, it scores {{=mixed_table.iloc[1]['mean AUC over the 5 folds']:.3f}} on the same compounds. Dropping them is worth {{=mixed_gain:+.3f}} AUC (corrected p = {{=mixed_paired.p_value:.3f}} over the five folds), which is larger than the gain of any feature of this part, and was found by looking at the data and not by adding a model. The selection of the model, which candidate, which blend, which threshold, had also been judged on the folds that chose it (6.2). The answer to all of it is the nested evaluation of 6.3, and the experiments of this part were not repeated in it one by one. They were turned into candidates and additions of the grid of Part VIII, whose gates (6.7) decide what stays.

**What the part tried.** The last cell collects the decisive number of each experiment from the cells above and says where the idea stands.
"""),
    code(VII_LEDGER),
    md("""
Of the {{=n_ideas}} ideas of the table, one is the reference, one was a shortcut, one (the threshold) is a real gain that comes from using the model better and not from new information, and of the other seven none moved the baseline by an amount that survives a design able to see it. Three survived as ingredients of the final grid for reasons other than their gain (the motif block, the read-across columns, the external panel block), and the others are not part of the final model. This is the main early finding of the project, and it should be read for what it is. It does not say that the ideas are wrong: the floor of the designs of this part was {{=mde_25_folds:.2f}} MCC for 25 folds on a comparison like that of 7.2, {{=architecture_floor:.2f}} for the architecture test above and {{=tier3_floor:.2f}} for the fifty compounds of 7.10, and an addition that helps by 0.01 or 0.02 was invisible to every one of them. It says that the project could not, with these designs, tell. The remainder of the work is the construction of a measurement that can (Part VI), a final model assembled under it (Part VIII), and a fair test of the same kind of ideas again, with the corrected statistics and in the corrected population (Parts IX and X).

*On what actually ran.* `cypmode/challenge/final_assembly.py` (CYP3A4; the module's docstring carries the account of the retraction of 7.4, in the project's own words), `final_assembly_cyp2d6.py` (CYP2D6, in its own process), `submission.py` (the validator and the builder of the file), `architecture_significance.py` and `cyp2d6_tuning_significance.py` (the paired tests, each run in three steps, one for each model library, writing to one shared file, and a third that combines it), and the stored tables `architecture_significance_raw.csv` and `tuning_significance_raw.csv` in `data/tdi_track/outputs/cyp2d6/`. The ledger above is computed from the cells of the part.
"""),
]
