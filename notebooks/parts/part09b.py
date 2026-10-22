# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IX (second part): frozen foundation-model embeddings (9.4), test-likeness weighting (9.5), floor-clipped arms (9.6), cross-isoform transfer of the arms (9.7), confidence-weighted training (9.8)."""

from notebooks.notebook_builder import code, md

CHEMELEON_FACTS = '''
# The frozen CheMeleon block (cypmode/challenge/frozen_embeddings.py). CheMeleon is a Chemprop message-passing network pretrained to predict a large set of computed molecular descriptors; here it is used frozen: every molecule
# becomes the mean-aggregated 2,048-number readout of its message-passing encoder, reduced to 64 principal components (fitted on all 6,895 structures, which needs no label) and stored as the block "chemeleon". The notebook
# reads the cache. Three facts: how much of the variance the 64 components keep; how well a linear classifier on them separates the labels in the five folds of 7.1 (as for ChemBERTa in 7.9); and how the block did in the nested design.
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

chemeleon_raw = np.load(ROOT / "data/tdi_track/outputs/caches/chemeleon_embeddings.npz", allow_pickle=True)["embeddings"].astype(np.float64)
chemeleon_pca = PCA(n_components=64).fit(chemeleon_raw)
variance_64 = float(chemeleon_pca.explained_variance_ratio_.sum())
variance_10 = float(chemeleon_pca.explained_variance_ratio_[:10].sum())
chemeleon_block = store8["chemeleon"][: len(train_file)]
chemeleon_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    labelled_rows = train_file[f"{iso}_is_TDI"].notna().to_numpy()
    X_c, y_c = chemeleon_block[labelled_rows], train_file.loc[labelled_rows, f"{iso}_is_TDI"].astype(int).to_numpy()
    oof_c = np.zeros(len(y_c))
    for fit_rows, held in StratifiedKFold(5, shuffle=True, random_state=42).split(X_c, y_c):
        oof_c[held] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(X_c[fit_rows], y_c[fit_rows]).predict_proba(X_c[held])[:, 1]
    gate = read_gate_log("embeddings", iso)
    diag = gate["diagnostic"] if gate is not None else pd.DataFrame()
    chemeleon_rows.append({"enzyme": iso, "linear probe on the 64 components, AUC": roc_auc_score(y_c, oof_c), "block alone in LightGBM, nested AUC at z0 1.0": float(diag[diag["z0"] == 1.0]["outer AUC alone"].iloc[0]) if len(diag) else np.nan, **stacked_summary("embeddings", iso)})
chemeleon_table = pd.DataFrame(chemeleon_rows)
show_table(chemeleon_table, floatfmt=".4f")
print(f"the 64 components keep {variance_64:.1%} of the variance of the 2,048 numbers, the first ten {variance_10:.1%}")
chemeleon_alone = read_gate_log("embeddings", "CYP3A4")["alone"]
'''

TESTLIKE = '''
# Test-likeness weights (cypmode/challenge/nested_runners.py, testlike_weights). Under covariate shift, the standard remedy is importance weighting by the density ratio p_test(x) / p_train(x) (section 4.1). It is approximated from structures
# alone: the ratio of a training compound's mean similarity to its 5 most similar blinded structures to its mean similarity to its 5 most similar other training compounds, tempered by alpha (0.5 or 1), clipped to [0.2, 5] and
# rescaled to mean 1. A compound that sits among the test compounds but alone in the training set gets a high weight; a member of a dense training series that the test set barely touches, a low one. Only the 750 structures are used,
# never labels. The cell computes the weights for the CYP3A4 pool and describes their distribution, with Kish's effective sample size (section 2.7).
from cypmode.challenge.nested_runners import TESTLIKE_NEIGHBOURS

bits_w = store8["morgan_bits"].astype(np.float32)
pool_w, test_w = plan8.pool, plan8.external
def tanimoto(a_rows, b_rows):
    overlap = bits_w[a_rows] @ bits_w[b_rows].T
    return overlap / (bits_w[a_rows].sum(1)[:, None] + bits_w[b_rows].sum(1)[None, :] - overlap)
to_test = np.sort(tanimoto(pool_w, test_w), axis=1)[:, -TESTLIKE_NEIGHBOURS:].mean(axis=1)
within_w = tanimoto(pool_w, pool_w)
np.fill_diagonal(within_w, -1.0)
to_train = np.sort(within_w, axis=1)[:, -TESTLIKE_NEIGHBOURS:].mean(axis=1)
ratio = to_test / np.maximum(to_train, 1e-3)
weight_rows = []
for alpha in (0.5, 1.0):
    w = np.clip(ratio**alpha, 0.2, 5.0)
    w = w / w.mean()
    weight_rows.append({"alpha": alpha, "weight, 5th percentile": float(np.percentile(w, 5)), "median": float(np.median(w)), "95th percentile": float(np.percentile(w, 95)), "effective sample size (of {:,})".format(len(w)): float(w.sum() ** 2 / (w**2).sum()),
                        "share at the lower clip": float((np.clip(ratio**alpha, 0.2, 5.0) <= 0.2).mean()), "share at the upper clip": float((np.clip(ratio**alpha, 0.2, 5.0) >= 5.0).mean())})
weight_table = pd.DataFrame(weight_rows)
show_table(weight_table, floatfmt=".3f")
testlike_stats = {"median to test": float(np.median(to_test)), "median to training": float(np.median(to_train)), "share of pool closer to the test than to the training": float((to_test > to_train).mean())}
testlike_gate = pd.DataFrame([{"enzyme": iso, **stacked_summary("test_likeness", iso)} for iso in ("CYP3A4", "CYP2D6")])
show_table(testlike_gate, floatfmt=".4f")
testlike_alone = read_gate_log("test_likeness", "CYP3A4")["alone"]
'''

ARMS_CLIP = '''
# Floor-clipped arms (cypmode/challenge/arms_experiment.py). The label depends on the two arms only through max(d, 4) and whether t clears 4 + log10 2 (section 2.3). Every reading below the assay floor 4 is therefore noise for
# the label, and a regressor that is asked to predict the arms spends capacity on it. Clipping both arms at 4 before fitting changes no label. The cell checks that on the data, and measures the noise below the floor.
from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD, direct_column, tdi_column

clip_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    t_arm, d_arm = train_file[tdi_column(iso)], train_file[direct_column(iso)]
    both = t_arm.notna() & d_arm.notna() & train_file[f"{iso}_is_TDI"].notna()
    label_given = train_file.loc[both, f"{iso}_is_TDI"].astype(bool).to_numpy()
    from_arms = (t_arm[both] - np.maximum(d_arm[both], ASSAY_FLOOR) > SHIFT_THRESHOLD).to_numpy()
    from_clipped = (np.maximum(t_arm[both], ASSAY_FLOOR) - np.maximum(d_arm[both], ASSAY_FLOOR) > SHIFT_THRESHOLD).to_numpy()
    sd_d = train_file.loc[both, direct_column(iso) + "_std"]
    below = (d_arm[both] < ASSAY_FLOOR).to_numpy()
    clip_rows.append({"enzyme": iso, "compounds with both arms": int(both.sum()), "labels reproduced by the rule": float((from_arms == label_given).mean()), "labels reproduced with both arms clipped at 4": float((from_clipped == label_given).mean()),
                      "direct readings below the floor": float(below.mean()), "median sd of a direct reading above / below the floor": f"{np.nanmedian(sd_d[~below]):.2f} / {np.nanmedian(sd_d[below]):.2f}"})
clip_table = pd.DataFrame(clip_rows)
show_table(clip_table, floatfmt=".3f")
arms_gate = pd.DataFrame([{"enzyme": iso, **stacked_summary("arms_experiment", iso)} for iso in ("CYP3A4", "CYP2D6")])
show_table(arms_gate, floatfmt=".4f")
arms_alone = read_gate_log("arms_experiment", "CYP3A4")["alone"]
'''

XISO = '''
# Cross-isoform transfer of the arms (cypmode/challenge/xiso_transfer.py). The shift t - d of different enzymes is almost uncorrelated, and an earlier look at that correlation had dropped the idea. But the arms themselves are potencies,
# and potency on one P450 says something about potency on another (shared lipophilicity, basicity, size). The cell measures how much, over the compounds measured on both enzymes of a pair, for the TDI arm, the direct arm and the
# shift. The idea tested is that the arms of the other enzymes, predicted from structure with cross-fitted regressors, help the classifier of the target enzyme.
from cypmode.challenge.tdi_targets import shift_target

enzymes_x = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
corr_rows = []
for a_i, iso_a in enumerate(enzymes_x):
    for iso_b in enzymes_x[a_i + 1:]:
        row = {"pair of enzymes": f"{iso_a} and {iso_b}"}
        for label, getter in (("TDI arm", tdi_column), ("direct arm", direct_column)):
            both = train_file[getter(iso_a)].notna() & train_file[getter(iso_b)].notna()
            row[f"{label}: compounds"] = int(both.sum())
            row[f"{label}: correlation"] = float(train_file.loc[both, getter(iso_a)].corr(train_file.loc[both, getter(iso_b)])) if both.sum() > 20 else np.nan
        shift_a, shift_b = shift_target(train_file[direct_column(iso_a)], train_file[tdi_column(iso_a)]), shift_target(train_file[direct_column(iso_b)], train_file[tdi_column(iso_b)])
        both_s = pd.Series(shift_a).notna().to_numpy() & pd.Series(shift_b).notna().to_numpy()
        row["shift: compounds"] = int(both_s.sum())
        row["shift: correlation"] = float(np.corrcoef(np.asarray(shift_a)[both_s], np.asarray(shift_b)[both_s])[0, 1]) if both_s.sum() > 20 else np.nan
        corr_rows.append(row)
corr_table = pd.DataFrame(corr_rows)
show_table(corr_table, floatfmt=".2f")
xiso_gate = pd.DataFrame([{"enzyme": iso, **stacked_summary("xiso_transfer", iso)} for iso in ("CYP3A4", "CYP2D6")])
show_table(xiso_gate, floatfmt=".4f")
xiso_alone = read_gate_log("xiso_transfer", "CYP3A4")["alone"]
'''

CONFIDENCE = '''
# Training on label confidence (cypmode/challenge/confidence_experiment.py, label_confidence.py). The official score counts only the compounds whose label is assigned with confidence; the project's yardstick (6.4) does the same. Does it
# help to train that way too? Two variants of the deep LightGBM on feature sets A and A+fps, with |z| computed from each training row's own measurement noise: "cdrop10" trains only on rows with |z| >= 1, "csoft" on every row
# weighted by min(1, |z| / 2). The cell counts what each variant does to the training rows and reads the nested comparison with the plain model.
from cypmode.challenge.label_confidence import training_weights

confidence_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    z_iso = confidence_z(train_file, iso)
    z_pool = z_iso[~np.isnan(z_iso)]
    keep, weight = training_weights(z_iso[~np.isnan(z_iso)], "drop1.0")
    soft_keep, soft_weight = training_weights(z_iso[~np.isnan(z_iso)], "soft")
    confidence_rows.append({"enzyme": iso, "labelled compounds with a z": len(z_pool), "dropped by cdrop10 (|z| < 1)": float(1 - keep.mean()), "mean weight under csoft": float(soft_weight[soft_keep].mean()), "share of rows with weight 1 under csoft": float((soft_weight[soft_keep] >= 1.0).mean())})
confidence_table = pd.DataFrame(confidence_rows)
show_table(confidence_table, floatfmt=".3f")
PAIR_CONF = re.compile(r"z0=(\\d\\.\\d+)\\s+(\\S+)\\s+vs plain: AUC ([\\d.]+) -> ([\\d.]+) \\(([+-][\\d.]+), p=([\\d.]+), better (\\d+)/(\\d+)\\)")
conf_rows = []
for iso in ("CYP3A4", "CYP2D6"):
    path = LOGS9 / f"confidence_{iso}_r3.log"
    if path.exists():
        for z0, name, before, after, gain, p, k, n in PAIR_CONF.findall(path.read_text(errors="ignore")):
            conf_rows.append({"enzyme": iso, "z0": float(z0), "variant": name, "AUC plain": float(before), "AUC variant": float(after), "gain": float(gain), "p (corrected t)": float(p), "folds better": f"{k} of {n}"})
conf_results = pd.DataFrame(conf_rows)
show_table(conf_results, floatfmt=".4f")
conf_largest = float(conf_results["gain"].abs().max()) if len(conf_results) else np.nan
conf_smallest_p = float(conf_results["p (corrected t)"].min()) if len(conf_results) else np.nan
conf_best = conf_results.sort_values("gain", ascending=False).iloc[0] if len(conf_results) else None
'''

CELLS = [
    md("""
## 9.4 A frozen foundation model: CheMeleon embeddings

> **Field:** Machine learning · Cheminformatics

*Where this section sits:* 7.9 tried a pretrained representation, the ChemBERTa embedding of the SMILES text, and found that appending 768 dense numbers to a tree model did not help. The lesson the project drew was about *form*, a dense block should be reduced before it is used, and about *kind*: a model that has read only text knows nothing of the physics of the molecule. This section tries a representation of a different kind, from a model that was pretrained to predict computed molecular descriptors, and uses it in the reduced form.

**What a foundation model is, and what frozen means.** A foundation model (section 4.9) is a model pretrained once, on a very large collection of data and a general task, so that its internal representation can be reused for many specific tasks with few labels of their own. CheMeleon (Burns and co-workers; the weights are deposited at Zenodo under the DOI 10.5281/zenodo.15426600) is a Chemprop directed message-passing network (section 4.8) pretrained on a large set of molecular descriptors that were computed from structures, so that its representation has been shaped to reproduce quantities that can be calculated from a molecule, the kind of information that the descriptors of Part III also hold. Using it **frozen** means that its weights are not changed by the project's few thousand labels: each molecule is passed through the encoder once, the mean of the encoder's output over the atoms (the *readout*, 2,048 numbers) is taken, and that is the representation. The project reduces the 2,048 numbers to 64 principal components (section 4.9), fitted on all 6,895 structures (this needs no label), and appends them to the feature sets A and A+fps as the block `chemeleon` (`cypmode/challenge/frozen_embeddings.py`). The extraction runs in a process of its own, under the environment that has the neural-network library, and writes a cache; the evaluation reads the cache (the rule of 7.8). The cell reads the cache and asks the same simple questions as for ChemBERTa, and then reads the nested comparison.
"""),
    code(CHEMELEON_FACTS),
    md("""
The 64 components keep {{=variance_64:.1%}} of the variance of the 2,048 numbers. A linear classifier on them alone separates the labels with an AUC of {{=chemeleon_table.iloc[0]['linear probe on the 64 components, AUC']:.3f}} for CYP3A4 and {{=chemeleon_table.iloc[1]['linear probe on the 64 components, AUC']:.3f}} for CYP2D6, against {{=probe_table.iloc[0]['AUC of a linear classifier on the embedding alone']:.3f}} and {{=probe_table.iloc[1]['AUC of a linear classifier on the embedding alone']:.3f}} for the ChemBERTa vector of 7.9, which has twelve times as many numbers. In a LightGBM, the block alone reaches a nested AUC of {{=chemeleon_table.iloc[0]['block alone in LightGBM, nested AUC at z0 1.0']:.3f}} on CYP3A4, so that the block carries signal of its own; but added to the default stack it changes the AUC by {{=chemeleon_table.iloc[0]['gain at z0 1.0']:+.4f}} (corrected p = {{=chemeleon_table.iloc[0]['p at z0 1.0']:.2f}}) at $z_0 = 1.0$ and {{=chemeleon_table.iloc[0]['gain at z0 1.96']:+.4f}} (p = {{=chemeleon_table.iloc[0]['p at z0 1.96']:.2f}}) at $z_0 = 1.96$, and for CYP2D6 by {{=chemeleon_table.iloc[1]['gain at z0 1.0']:+.4f}} and {{=chemeleon_table.iloc[1]['gain at z0 1.96']:+.4f}}. The gate is {{=chemeleon_table.iloc[0]['gate']}} for CYP3A4 and {{=chemeleon_table.iloc[1]['gate']}} for CYP2D6. A representation adds to a stack only if it knows something that the members do not. The simplest reading is that the 23 members, which read descriptors, fingerprints and the other blocks, already hold what the block knows, and that a network pretrained on computed descriptors repeats much of what the descriptors of Part III contain; the experiment does not test that reading, it shows only that the block adds nothing measurable. The fine-tuned CheMeleon, in which the whole network is trained on the project's labels and not frozen, is a different candidate and belongs to the neural family of 9.9.

*On what actually ran.* `cypmode/challenge/frozen_embeddings.py` (`extract` under the neural-network environment, the 64-component reduction `N_COMPONENTS`, the candidates `lgbm_chem_deep`, `lgbm_A+chem_deep` and `lgbm_A+fps+chem_deep`, the gate through `addition_experiment.py`), the cache `chemeleon_embeddings.npz`, and the logs `frozen_embeddings_extract.log` and `embeddings_<enzyme>.log`. The queue job is `050_frozen_embeddings`; `084_tdi_stack_reevaluation` repeats the stacked comparisons of this section and of 9.5 to 9.7 on the final plan. The weights are those of the Zenodo record named above, loaded into Chemprop's message-passing encoder with the hyperparameters stored beside them.
"""),
    md("""
## 9.5 Test-likeness weighting: importance weights from structures alone

> **Field:** Statistics · Machine learning

*Where this section sits:* 6.4 re-weighted the *validation* compounds so that the score looks like the score on the blinded set. This section asks whether the *training* compounds should be weighted the same way, so that the model spends its capacity where the blinded set lives. The blinded compounds sit closer to the training file than the compounds of a cross-validation fold do (section 6.8), and the argument for weighting is that a model fitted with equal weights spends much of its capacity on dense parts of the training set that the blinded compounds hardly touch.

**Importance weighting for the training rows.** Section 2.7 weighted the validation rows. If a model is judged by its average loss $L(x)$ over the test distribution $p_{\\text{test}}(x)$ but fitted on training compounds drawn from $p_{\\text{train}}(x)$, then $E_{\\text{test}}[L] = E_{\\text{train}}[w\\,L]$ with the *density ratio* $w(x) = p_{\\text{test}}(x)/p_{\\text{train}}(x)$ (section 4.1): weight each training compound by how much more likely it is under the test distribution than under the training one. The ratio cannot be computed, because the densities of molecules are unknown, so it is approximated from structures alone. For a training compound $i$ let $a_i$ be its mean Tanimoto similarity to its five most similar *blinded* structures and $b_i$ its mean similarity to its five most similar *other training* compounds. The weight is $w_i = \\mathrm{clip}\\big((a_i / b_i)^{\\alpha},\\ 0.2,\\ 5\\big)$, rescaled to have mean 1, with $\\alpha \\in \\{0.5, 1\\}$ tempering the ratio. A compound that sits among the test compounds but alone in the training set gets a high weight, and a member of a dense series that the test set barely touches a low one. Only the 750 structures are used, never a label (`testlike_weights` in `cypmode/challenge/nested_runners.py`). The cell computes the weights once for the whole CYP3A4 pool; in the nested design each task computes them from its own training rows.
"""),
    code(TESTLIKE),
    md("""
The weights are a mild reweighting. The median training compound has a mean similarity of {{=testlike_stats['median to test']:.3f}} to its five nearest blinded structures and {{=testlike_stats['median to training']:.3f}} to its five nearest other training compounds, so that the raw ratio is below 1 for most of the pool ({{=testlike_stats['share of pool closer to the test than to the training']:.0%}} of the compounds are closer to the blinded structures than to the rest of the training); the rescaling to mean 1 turns that into weights, and at $\\alpha = 1$ they run from {{=weight_table.iloc[1]['weight, 5th percentile']:.2f}} (5th percentile) to {{=weight_table.iloc[1]['95th percentile']:.2f}} (95th). {{=weight_table['share at the lower clip'].max():.1%}} of the compounds reach the lower clip and {{=weight_table['share at the upper clip'].max():.1%}} the upper, and the effective sample size is {{=weight_table.iloc[1].iloc[4]:,.0f}} of {{=len(pool_w):,}}, that is {{=weight_table.iloc[1].iloc[4] / len(pool_w):.0%}} of the pool. A reweighting of this size moves the fit little, and the nested comparison finds no measurable change: alone, for CYP3A4 (the changes of the AUC of the four weighted candidates against their plain counterparts are between {{=testlike_alone['gain'].min():+.4f}} and {{=testlike_alone['gain'].max():+.4f}}) and in the stack ({{=testlike_gate.iloc[0]['gain at z0 1.0']:+.4f}} and {{=testlike_gate.iloc[0]['gain at z0 1.96']:+.4f}} for CYP3A4, {{=testlike_gate.iloc[1]['gain at z0 1.0']:+.4f}} and {{=testlike_gate.iloc[1]['gain at z0 1.96']:+.4f}} for CYP2D6); the gate is {{=testlike_gate.iloc[0]['gate']}} for CYP3A4. A sharper weighting would trade the bias it removes for variance, since a handful of compounds would then carry the fit, which is the usual price of importance weights. The effective sample size shows that the price paid here is small, and the measured benefit is nil.

*On what actually ran.* `cypmode/challenge/test_likeness_training.py` (the four candidates `lgbm_{A,A+fps}_deep_testlike{0.5,1.0}` and the alone pairs), `testlike_weights` and `TESTLIKE_NEIGHBOURS` in `nested_runners.py`, and the log `test_likeness_<enzyme>.log`. The queue job is `060_test_likeness_training`.
"""),
    md("""
## 9.6 Floor-clipped arms: not spending capacity below the assay floor

> **Field:** Statistics · Machine learning

*Where this section sits:* 8.1 introduced regression members that predict the two arms and score $t - \\max(d, 4)$. This section asks whether they should be fitted on the arms as measured or on the arms clipped at the assay floor, and it is the cleanest example in Part IX of an idea whose premise is true and whose effect is nil.

**The idea.** The label depends on the arms only through $\\max(d, 4)$ and through whether $t$ clears $4 + \\log_{10} 2$. A reading below the floor of 4 is, for the label, just "below the floor", and its exact value is noise: the project's measurements below the floor are about ten times as noisy as those above (section 2.5). A regressor fitted by squared error on the raw arms spends effort on reproducing that noise. If both arms are replaced by $\\max(\\cdot, 4)$ before fitting, no label changes, and the regressor is relieved of the part of the target that carries no information. The cell checks the first claim on the data, that the rule reproduces every label from the raw arms and from the clipped arms, and measures the share and the noise of the readings below the floor.
"""),
    code(ARMS_CLIP),
    md("""
The rule reproduces {{=clip_table.iloc[0]['labels reproduced by the rule']:.1%}} of the CYP3A4 labels and {{=clip_table.iloc[1]['labels reproduced by the rule']:.1%}} of the CYP2D6 labels from the raw arms, and the same with both arms clipped at 4 ({{=clip_table.iloc[0]['labels reproduced with both arms clipped at 4']:.1%}} and {{=clip_table.iloc[1]['labels reproduced with both arms clipped at 4']:.1%}}): clipping changes no label, as claimed. The readings below the floor are {{=clip_table.iloc[0]['direct readings below the floor']:.0%}} of the direct readings for CYP3A4 and {{=clip_table.iloc[1]['direct readings below the floor']:.0%}} for CYP2D6, and their median standard deviation is {{=clip_table.iloc[0]['median sd of a direct reading above / below the floor']}} (above / below the floor) for CYP3A4, about ten times larger below. For CYP3A4 two-fifths of the direct target is therefore below the floor and noisy. The clipped regressors (`lgbm_reg_arms_clip_A` and `lgbm_reg_arms_clip_A+fps`) were compared with the unclipped ones alone and in the stack: alone, for CYP3A4, the AUC changes by between {{=arms_alone['gain'].min():+.4f}} and {{=arms_alone['gain'].max():+.4f}}, with corrected p-values of at least {{=arms_alone['p'].min():.2f}}; in the stack by {{=arms_gate.iloc[0]['gain at z0 1.0']:+.4f}} and {{=arms_gate.iloc[0]['gain at z0 1.96']:+.4f}} for CYP3A4 and {{=arms_gate.iloc[1]['gain at z0 1.0']:+.4f}} and {{=arms_gate.iloc[1]['gain at z0 1.96']:+.4f}} for CYP2D6. The gate is {{=arms_gate.iloc[0]['gate']}} for CYP3A4 and {{=arms_gate.iloc[1]['gate']}} for CYP2D6. The clipping is a correct statement about the label, and the measurement says that the models do not need it; one reading is that a tree ensemble fitted by squared error places its splits where the signal is and leaves the noise below the floor in leaves that matter little, but the experiment does not test that reading.

*On what actually ran.* `cypmode/challenge/arms_experiment.py` (the family `lgbm_reg_arms_clip` of `nested_runners.py`, which clips the two arms at `ASSAY_FLOOR` before fitting), `tdi_targets.py` (the label rule) and the log `arms_experiment_<enzyme>.log`. The queue job is `078_arm_variants`.
"""),
    md("""
## 9.7 Cross-isoform transfer of the assay arms

> **Field:** Cheminformatics · Machine learning

*Where this section sits:* 9.6 used the arms of one enzyme. The training file also holds the arms of three other enzymes, measured on overlapping sets of compounds, and a compound that is a potent inhibitor of CYP2C9 is often a potent inhibitor of CYP3A4 (shared lipophilicity, size, basicity). This section asks whether the arms of the *other* enzymes, which the blinded compounds will not have measured and which can only be predicted from structure, help the classifier of the target enzyme.

**The construction** (`cypmode/challenge/xiso_transfer.py`). In a first stage a LightGBM regressor per other enzyme and per arm predicts the floor-clipped pIC50 from the 2D features. The compounds that were measured on that enzyme get *cross-fitted* predictions (five folds), so that no training compound receives a prediction from a model that saw its own measurement; every other compound receives the prediction of a model fitted on all the rows allowed in the task. The allowed rows are the training rows of the task and every compound outside the target's pool, never an evaluation row, and the other-enzyme measurements of an evaluation row are never read. In a second stage the target-enzyme classifier, a deep LightGBM, is given the 2D features and the six predicted arms. The idea was dropped once on the evidence of one number, the correlation of the *shifts* between enzymes, which is near zero. The arms are what is correlated, and the cell measures how much.
"""),
    code(XISO),
    md("""
The arms correlate strongly between some enzymes and not at all between others. CYP2C9 and CYP3A4 share {{=corr_table[corr_table['pair of enzymes'] == 'CYP2C9 and CYP3A4'].iloc[0]['direct arm: compounds']}} compounds measured on both and their direct arms correlate at {{=corr_table[corr_table['pair of enzymes'] == 'CYP2C9 and CYP3A4'].iloc[0]['direct arm: correlation']:.2f}} (the TDI arms at {{=corr_table[corr_table['pair of enzymes'] == 'CYP2C9 and CYP3A4'].iloc[0]['TDI arm: correlation']:.2f}}); the shift, the difference of the arms, only at {{=corr_table[corr_table['pair of enzymes'] == 'CYP2C9 and CYP3A4'].iloc[0]['shift: correlation']:.2f}}, which is why the earlier look at the shift had dismissed the idea. CYP2D6 is the exception: its direct arm correlates with that of every other enzyme at no more than {{=corr_table[corr_table['pair of enzymes'].str.contains('CYP2D6')]['direct arm: correlation'].abs().max():.2f}} in absolute value, so that there is almost nothing to transfer to the enzyme that needs the help most. For CYP3A4 the transfer has a real basis, and still the nested comparison finds nothing: alone, the AUC changes by between {{=xiso_alone['gain'].min():+.4f}} and {{=xiso_alone['gain'].max():+.4f}} (p of at least {{=xiso_alone['p'].min():.2f}}), in the stack by {{=xiso_gate.iloc[0]['gain at z0 1.0']:+.4f}} and {{=xiso_gate.iloc[0]['gain at z0 1.96']:+.4f}}, and the gate is {{=xiso_gate.iloc[0]['gate']}} for CYP3A4 and {{=xiso_gate.iloc[1]['gate']}} for CYP2D6. The predicted arms of CYP2C9 are functions of the same structure that the CYP3A4 classifier already reads. A feature derived from the structure by another model adds information only if that model knew something that the classifier could not learn from the structure and the CYP3A4 labels; one reading of the result is that it did not, but the experiment measures only that the transfer adds nothing.

*On what actually ran.* `cypmode/challenge/xiso_transfer.py` (`fit_predict_xiso`, the family `lgbm_xiso`, the cross-fitting and the allowed-rows rule), `ISOFORM_ORDER` and the candidates `xiso_{A,A+fps}_deep` in `nested_runners.py`, and the log `xiso_transfer_<enzyme>.log`. The queue job is `076_xiso_arm_transfer`.
"""),
    md("""
## 9.8 Training on label confidence

> **Field:** Statistics · Machine learning

*Where this section sits:* 6.4 restricted the *score* to compounds whose label is assigned with confidence, because the challenge does the same. This section asks the natural follow-up: should the *model* be trained that way too? A label close to the threshold is nearly a coin flip, and training on it adds noise. The experiment was pre-registered: the candidates and the rule (an improvement counts only if the corrected p-value is below 0.05 at both cut-offs for the enzyme it is claimed for) were fixed before any variant was run (`cypmode/challenge/confidence_experiment.py`).

**The two variants.** With $z_i$ the confidence of section 2.5, the distance of the shift of training row $i$ from the threshold $\\log_{10} 2$ in units of the standard deviation of that shift (computed from the standard deviations of the two arms), the deep LightGBM is fitted (a) only on the rows with $|z_i| \\ge 1$ (`cdrop10`), or (b) on every row weighted by $|z_i|/2$, limited to the range 0.1 to 1 (`csoft`): a row whose label is within one standard deviation of the threshold counts for half or less, and one beyond two standard deviations for the whole. A row with no value of $z$ (its TDI arm was not measured) is dropped by the first variant and gets the weight 0.1 in the second. The cell counts what each does to the rows and reads the nested comparison, on feature sets A and A+fps, for both enzymes.
"""),
    code(CONFIDENCE),
    md("""
Dropping the rows with $|z| < 1$ removes {{=confidence_table.iloc[0]['dropped by cdrop10 (|z| < 1)']:.1%}} of the CYP3A4 training rows and {{=confidence_table.iloc[1]['dropped by cdrop10 (|z| < 1)']:.1%}} of the CYP2D6 rows, and the soft weighting has a mean weight of {{=confidence_table.iloc[0]['mean weight under csoft']:.2f}} and {{=confidence_table.iloc[1]['mean weight under csoft']:.2f}}. The comparison finds no effect that survives the rule. For CYP3A4 the changes of the AUC are within {{=conf_results[conf_results['enzyme'] == 'CYP3A4']['gain'].abs().max():.4f}} of zero. For CYP2D6 all {{=int((conf_results[conf_results['enzyme'] == 'CYP2D6']['gain'] > 0).sum())}} of the {{=len(conf_results[conf_results['enzyme'] == 'CYP2D6'])}} comparisons are positive, up to {{=conf_results[conf_results['enzyme'] == 'CYP2D6']['gain'].max():+.4f}}, and their corrected p-values are between {{=conf_results[conf_results['enzyme'] == 'CYP2D6']['p (corrected t)'].min():.2f}} and {{=conf_results[conf_results['enzyme'] == 'CYP2D6']['p (corrected t)'].max():.2f}}. A consistent sign with p-values this large is what a small true effect looks like when the design cannot see it, and also what noise looks like when the eight comparisons are correlated (they share the same folds and nearly the same models), so that the eight signs are not eight independent votes. The pre-registered rule is not met for either enzyme, and the variants are not part of the model.

*On what actually ran.* `cypmode/challenge/confidence_experiment.py` (the candidates `lgbm_{A,A+fps}_deep_{cdrop10,csoft}`, the pairs, the corrected t), `label_confidence.py` (`confidence_z`, `training_weights`) and the logs `confidence_<enzyme>_r3.log`, written by the run scripts `scripts/run_tdi_then_structure.sh`, `scripts/run_tdi_power_then_structure.sh` and `scripts/run_tdi_context_then_structure.sh`. The same experiment holds a second family of candidates that add the alert-context block (`tdi_alert_context`) to the feature sets, whose result is in the same logs and is part of the ledger of 9.11.
"""),
]
