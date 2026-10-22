# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IV (third part): message passing and attention, principal components and tabular in-context learning, the metrics."""

from notebooks.notebook_builder import code, md

MPNN = '''
# A directed message-passing network (the form Chemprop uses), forward pass only, with random weights, for the example compound. An atom starts as a vector of nine numbers (which of five elements it is,
# its number of heavy neighbours, aromatic, in a ring, hydrogens); a bond as four numbers (its type). Every bond gives two directed edges. An edge u -> v starts as relu(W_i [x_u, b_uv]); in each step it
# is updated with the sum of the edges that point into u from anywhere but v. At the end every atom adds up the edges pointing into it, and the molecule is the mean of its atoms' states.
ELEMENTS = ["C", "N", "O", "S", "Br"]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

def atom_vector(atom):
    return np.array([float(atom.GetSymbol() == e) for e in ELEMENTS] + [atom.GetDegree() / 4, float(atom.GetIsAromatic()), float(atom.IsInRing()), atom.GetTotalNumHs() / 3])

def bond_vector(bond):
    return np.array([float(bond.GetBondType() == t) for t in BOND_TYPES])

def make_weights(hidden, seed=0):
    rng = np.random.default_rng(seed)
    return (np.abs(rng.normal(0, 0.5, (hidden, 9 + 4))), np.abs(rng.normal(0, 0.5 / np.sqrt(hidden), (hidden, hidden))), np.abs(rng.normal(0, 0.5, (hidden, 9 + hidden))))

def message_passing(molecule, weights, steps, x=None):
    W_i, W_h, W_o = weights
    x = np.array([atom_vector(a) for a in molecule.GetAtoms()]) if x is None else x
    neighbours = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()] for a in molecule.GetAtoms()}
    directed = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_vector(b)) for b in molecule.GetBonds()] + [(b.GetEndAtomIdx(), b.GetBeginAtomIdx(), bond_vector(b)) for b in molecule.GetBonds()]
    index = {(u, v): k for k, (u, v, _) in enumerate(directed)}
    h0 = np.array([np.maximum(0, W_i @ np.concatenate([x[u], f])) for u, v, f in directed])
    h = h0.copy()
    for _ in range(steps):
        updated = np.empty_like(h)
        for k, (u, v, _) in enumerate(directed):
            incoming = [index[(w, u)] for w in neighbours[u] if w != v]               # the messages into u from every neighbour except the one this edge points to
            updated[k] = np.maximum(0, h0[k] + W_h @ (h[incoming].sum(axis=0) if incoming else np.zeros(h.shape[1])))
        h = updated
    states = np.array([np.maximum(0, W_o @ np.concatenate([x[v], sum(h[index[(w, v)]] for w in neighbours[v])])) for v in range(molecule.GetNumAtoms())])
    return states, states.mean(axis=0), len(directed)

weights = make_weights(24)
states, vector, n_directed = message_passing(mol, weights, steps=3)
print(f"{mol.GetNumBonds()} bonds give {n_directed} directed edges; the molecule is described by a vector of {vector.shape[0]} numbers, whatever its size")

# Renumbering the atoms must not change the molecule vector (the readout is a mean over atoms).
rng = np.random.default_rng(4)
permutation = [int(i) for i in rng.permutation(mol.GetNumAtoms())]
renumbered = Chem.RenumberAtoms(mol, permutation)
_, vector_renumbered, _ = message_passing(renumbered, weights, steps=3)
invariant = bool(np.allclose(vector, vector_renumbered))
print(f"the vector is the same after a random renumbering of the atoms: {invariant}")

# The receptive field: change the features of one atom (the sulfur) and see which atoms' final states change, for 0 to 3 message-passing steps.
distance = Chem.GetDistanceMatrix(mol)
centre = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"][0]
field_rows = []
for steps in range(4):
    base, _, _ = message_passing(mol, weights, steps)
    x_changed = np.array([atom_vector(a) for a in mol.GetAtoms()])
    x_changed[centre] += 1.0
    changed, _, _ = message_passing(mol, weights, steps, x=x_changed)
    affected = np.flatnonzero(np.abs(changed - base).max(axis=1) > 1e-12)
    field_rows.append({"message-passing steps": steps, "atoms whose state changes": len(affected), "farthest of them, in bonds from the changed atom": int(distance[centre, affected].max())})
show_table(pd.DataFrame(field_rows))
n_atoms_affected = field_rows[-1]["atoms whose state changes"]
'''

ATTENTION = r'''
# Self-attention written out on the canonical SMILES of the example compound. The string is cut into tokens (an atom symbol, a bond, a bracket, a ring digit); every token gets a vector (random here,
# learned in a real model) plus a vector that encodes its position; and one attention head mixes the tokens. The weights are random, so the pattern means nothing chemically: the cell shows the mechanics.
import re

token_regex = re.compile(r"\[[^\]]+\]|Br|Cl|@@|%\d\d|[A-Za-z]|\d|[()=#+\-/\\.:~@*$]")
canonical_smiles = Chem.MolToSmiles(mol)
tokens = token_regex.findall(canonical_smiles)
assert "".join(tokens) == canonical_smiles
vocabulary = {t: i for i, t in enumerate(sorted(set(tokens)))}
d_model = 16
rng = np.random.default_rng(2)
embedding = rng.normal(size=(len(vocabulary), d_model))
position = np.arange(len(tokens))[:, None]
frequency = 1 / 10000 ** (np.arange(0, d_model, 2) / d_model)
positional = np.zeros((len(tokens), d_model))
positional[:, 0::2], positional[:, 1::2] = np.sin(position * frequency), np.cos(position * frequency)
X_tokens = embedding[[vocabulary[t] for t in tokens]] + positional
W_q, W_k, W_v = (rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model)) for _ in range(3))
Q, Kmat, V = X_tokens @ W_q, X_tokens @ W_k, X_tokens @ W_v

def softmax(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

attention = softmax(Q @ Kmat.T / np.sqrt(d_model))
mixed = attention @ V
print(f"{len(tokens)} tokens ({len(vocabulary)} different); attention matrix {attention.shape[0]} x {attention.shape[1]}; every row sums to 1: {bool(np.allclose(attention.sum(axis=1), 1))}; output {mixed.shape}")
print("the first tokens:", tokens[:12])

# Why the scores are divided by the square root of the key length: a dot product of two vectors of d independent numbers with variance 1 has variance d, and large scores push the softmax to a one-hot vector.
scale_rows = []
for d_k in (16, 64, 256):
    q, k = rng.normal(size=(20000, d_k)), rng.normal(size=(20000, d_k))
    dots = (q * k).sum(axis=1)
    queries, keys = rng.normal(size=(20, d_k)), rng.normal(size=(20, d_k))
    raw, scaled = softmax(queries @ keys.T), softmax(queries @ keys.T / np.sqrt(d_k))
    scale_rows.append({"key length d_k": d_k, "variance of q . k": float(dots.var()), "variance of q . k / sqrt(d_k)": float((dots / np.sqrt(d_k)).var()),
                       "mean largest attention weight, unscaled": float(raw.max(axis=1).mean()), "scaled": float(scaled.max(axis=1).mean())})
show_table(pd.DataFrame(scale_rows), floatfmt=".2f")
n_tokens, n_vocabulary = len(tokens), len(vocabulary)
'''

PCA_CELL = '''
# Principal component analysis by hand on the 217 descriptors of all 6895 compounds (no label is used): standardise (clipped at 5 standard deviations), form the covariance matrix, take its eigenvectors, and
# compare with scikit-learn.
from sklearn.decomposition import PCA

desc_all = store["rdkit_desc"].astype(np.float64)
mu, sd = desc_all.mean(axis=0), desc_all.std(axis=0)
sd[sd == 0] = 1.0
Z = np.clip((desc_all - mu) / sd, -5, 5)
Zc = Z - Z.mean(axis=0)
covariance = Zc.T @ Zc / (len(Zc) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
explained = eigenvalues / eigenvalues.sum()
library = PCA(n_components=10).fit(Z)
scores_two = Zc @ eigenvectors[:, :2]
pca_rows = [{"quantity": "largest difference in the explained-variance shares of the first 10 components, by hand against scikit-learn", "value": f"{np.abs(explained[:10] - library.explained_variance_ratio_).max():.1e}"},
            {"quantity": "the first 10 eigenvectors are orthonormal (largest deviation of V'V from the identity)", "value": f"{np.abs(eigenvectors[:, :10].T @ eigenvectors[:, :10] - np.eye(10)).max():.1e}"},
            {"quantity": "variance of the data along the first component, and its eigenvalue", "value": f"{np.var(Zc @ eigenvectors[:, 0], ddof=1):.3f} and {eigenvalues[0]:.3f}"},
            {"quantity": "share of the variance in the first 2, 10 and 64 components", "value": f"{explained[:2].sum():.1%}, {explained[:10].sum():.1%}, {explained[:64].sum():.1%}"}]
show_table(pd.DataFrame(pca_rows))
fig, axis = plt.subplots(figsize=(6.5, 4.6))
n_train = len(train)
axis.scatter(scores_two[:n_train, 0], scores_two[:n_train, 1], s=4, color=theme["blue"], alpha=0.35, label="training compounds")
axis.scatter(scores_two[n_train:, 0], scores_two[n_train:, 1], s=7, color=theme["red"], alpha=0.7, label="test compounds")
axis.set(xlabel=f"first component ({explained[0]:.0%} of the variance)", ylabel=f"second component ({explained[1]:.0%})")
axis.legend(frameon=False)
tidy(axis, theme)
fig.tight_layout()
share_two, share_ten, share_64 = float(explained[:2].sum()), float(explained[:10].sum()), float(explained[:64].sum())
'''

CONFUSION = '''
# Out-of-fold probabilities of the project's deep LightGBM on feature set A (every compound is scored by a model that did not see it), and the numbers that describe them as calls: the 37 % of compounds with
# the highest scores are called positive (the call rate the project's CYP3A4 model uses, section 8.4).
from cypmode.challenge.nested_runners import LGBM_PRESETS

oof = np.zeros(len(y_ml))
for fit_rows, held in folds_ml:
    oof[held] = LGBMClassifier(verbose=-1, n_jobs=2, random_state=0, **LGBM_PRESETS["deep"]).fit(set_a[fit_rows], y_ml[fit_rows]).predict_proba(set_a[held])[:, 1]
called = oof >= np.quantile(oof, 1 - 0.37)
TP, FP = int((called & (y_ml == 1)).sum()), int((called & (y_ml == 0)).sum())
FN, TN = int((~called & (y_ml == 1)).sum()), int((~called & (y_ml == 0)).sum())
n_all = TP + FP + FN + TN
accuracy, precision, recall = (TP + TN) / n_all, TP / (TP + FP), TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
mcc_formula = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
from sklearn.metrics import matthews_corrcoef
mcc_library, mcc_pearson = matthews_corrcoef(y_ml, called.astype(int)), np.corrcoef(y_ml, called.astype(int))[0, 1]
show_table(pd.DataFrame([{"": "called positive", "is positive": TP, "is negative": FP}, {"": "called negative", "is positive": FN, "is negative": TN}]))
metric_rows = [{"metric": "accuracy (TP + TN) / n", "value": accuracy}, {"metric": "precision TP / (TP + FP)", "value": precision}, {"metric": "recall TP / (TP + FN)", "value": recall}, {"metric": "F1, the harmonic mean of precision and recall", "value": f1},
               {"metric": "MCC by the formula", "value": mcc_formula}, {"metric": "MCC by scikit-learn", "value": mcc_library}, {"metric": "Pearson correlation of the label with the call", "value": mcc_pearson}]
show_table(pd.DataFrame(metric_rows), floatfmt=".4f")
share_called, share_positive = float(called.mean()), float(y_ml.mean())
accuracy_if_all_negative = 1 - share_positive
'''

ROC = '''
# The ROC curve, the area under it computed three ways, and the precision-recall curve with average precision. The three ways: the trapezoid rule on the curve; the probability that a random positive is scored
# above a random negative (all positive-negative pairs, ties counted half); and scikit-learn.
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

fpr, tpr, _ = roc_curve(y_ml, oof)
auc_trapezoid = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
positives, negatives = oof[y_ml == 1], oof[y_ml == 0]
pair_compare = positives[:, None] - negatives[None, :]
auc_pairs = float((pair_compare > 0).mean() + 0.5 * (pair_compare == 0).mean())
auc_library = float(roc_auc_score(y_ml, oof))
order = np.argsort(-oof, kind="stable")
hits = y_ml[order]
precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
ap_formula = float((precision_at_k * hits).sum() / hits.sum())                  # the mean, over the positives, of the precision at the rank of each positive
ap_library = float(average_precision_score(y_ml, oof))
print(f"AUC by the trapezoid rule {auc_trapezoid:.5f}; as P(positive scored above negative) {auc_pairs:.5f}; scikit-learn {auc_library:.5f}")
print(f"average precision by the formula {ap_formula:.5f}; scikit-learn {ap_library:.5f}; the value for a model that ranks at random is the positive rate, {y_ml.mean():.3f}")
precision_curve, recall_curve, _ = precision_recall_curve(y_ml, oof)
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9))
left.plot(fpr, tpr, color=theme["blue"], lw=2.2)
left.plot([0, 1], [0, 1], color=theme["muted"], ls=":", lw=1)
left.set(xlabel="false-positive rate", ylabel="true-positive rate", title=f"ROC curve, AUC {auc_library:.3f}")
right.plot(recall_curve, precision_curve, color=theme["orange"], lw=2.2)
right.axhline(y_ml.mean(), color=theme["muted"], ls=":", lw=1)
right.set(xlabel="recall", ylabel="precision", title=f"precision-recall curve, AP {ap_library:.3f}", ylim=(0, 1))
for axis in (left, right):
    tidy(axis, theme)
fig.tight_layout()
'''

PREVALENCE_MCC = '''
# The same ranking quality at different positive rates. Scores of positives are N(m, 1) and of negatives N(0, 1), with m chosen so that the AUC is 0.75 whatever the positive rate (AUC = Phi(m / sqrt 2)).
# For each rate, the best MCC over all thresholds and the average precision are computed on 400000 simulated compounds.
from scipy.stats import norm

m_shift = np.sqrt(2) * norm.ppf(0.75)
rng = np.random.default_rng(8)
prev_rows = []
for rate in (0.5, 0.3, 0.2, 0.07):
    n_sim = 400_000
    labels_sim = rng.random(n_sim) < rate
    scores_sim = rng.normal(size=n_sim) + m_shift * labels_sim
    order_sim = np.argsort(-scores_sim)
    sorted_labels = labels_sim[order_sim]
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(~sorted_labels)
    fn_cum, tn_cum = sorted_labels.sum() - tp_cum, (~sorted_labels).sum() - fp_cum
    denominator = np.sqrt((tp_cum + fp_cum) * (tp_cum + fn_cum) * (tn_cum + fp_cum) * (tn_cum + fn_cum).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc_curve = (tp_cum * tn_cum - fp_cum * fn_cum) / denominator
    prev_rows.append({"positive rate": rate, "AUC": float(roc_auc_score(labels_sim, scores_sim)), "best MCC over all thresholds": float(np.nanmax(mcc_curve)),
                      "average precision": float(average_precision_score(labels_sim, scores_sim)), "average precision of a random ranking": rate})
show_table(pd.DataFrame(prev_rows), floatfmt=".3f")
mcc_at_rate = {r["positive rate"]: r["best MCC over all thresholds"] for r in prev_rows}
'''

CALIBRATION = '''
# Probabilities as probabilities. The Brier score (mean squared difference between the probability and the label) against that of always predicting the positive rate; the expected calibration error with ten
# bins; and the calibration slope, the coefficient of a logistic regression of the label on the logit of the probability (1 for a calibrated model). The project's own function is called as well.
from cypmode.challenge.calibration_experiment import calibration_metrics

brier = float(np.mean((oof - y_ml) ** 2))
brier_reference = float(y_ml.mean() * (1 - y_ml.mean()))
bin_index = np.minimum((oof * 10).astype(int), 9)
bin_rows, ece_by_hand = [], 0.0
for b in range(10):
    inside = bin_index == b
    if inside.any():
        ece_by_hand += inside.mean() * abs(oof[inside].mean() - y_ml[inside].mean())
        bin_rows.append({"probability bin": f"{b / 10:.1f} to {(b + 1) / 10:.1f}", "compounds": int(inside.sum()), "mean predicted probability": float(oof[inside].mean()), "share that are positive": float(y_ml[inside].mean())})
show_table(pd.DataFrame(bin_rows), floatfmt=".3f")
project = calibration_metrics(y_ml.astype(float), oof, np.ones(len(oof)))
print(f"Brier score {brier:.4f} against {brier_reference:.4f} for the constant prediction (skill {1 - brier / brier_reference:.1%}); ECE by hand {ece_by_hand:.4f}, by the project's function {project['ece']:.4f}; calibration slope {project['slope']:.2f}, intercept {project['intercept']:.2f}")
brier_skill, ece_value, slope_value = 1 - brier / brier_reference, project["ece"], project["slope"]
calibration_verdict = "meet" if (ece_value <= 0.05 and 0.8 <= slope_value <= 1.2) else "do not meet"
'''

CELLS = [
    md("""
## 4.8 Message-passing networks on molecular graphs, and networks on text

> **Field:** Machine learning · Cheminformatics

**Learning on the graph.** The fingerprints of Part III are made by fixed rules. A **message-passing neural network** (Gilmer and co-workers, 2017) learns its own. Every atom carries a vector of numbers, its **state**, initially made of the properties of the atom. In each of $T$ steps, every atom receives **messages** from its bonded neighbours, combines them with its own state and updates the state with a small neural network whose weights are learned. After $T$ steps the state of an atom has taken in everything within $T$ bonds, which is exactly the idea of the Morgan environment of
section 3.1 but with learned features instead of hashes. A final **readout** combines the states of all atoms, by a sum or a mean, into one vector for the molecule, which a small network turns into the prediction. Because the readout does not depend on the order of the atoms, the model gives the same answer however the atoms are numbered. **Chemprop** (Yang and co-workers, 2019) uses a variant in which the messages live on the *directed bonds*: the state $h_{u\\to v}$ of the bond from $u$ to $v$ starts as
$\\mathrm{ReLU}(W_i\\,[x_u, b_{uv}])$, where $x_u$ describes atom $u$, $b_{uv}$ the bond, $[\\cdot,\\cdot]$ joins two vectors and $W_i$ is a weight matrix, and is updated for $T$ steps, step $t$ giving the state $h^{(t)}$, by

$$h_{u\\to v}^{(t+1)} \\;=\\; \\mathrm{ReLU}\\Bigl(h_{u\\to v}^{(0)} + W_h \\sum_{k \\in N(u)\\setminus\\{v\\}} h_{k\\to u}^{(t)}\\Bigr),$$

the sum being over the neighbours $k$ of $u$ other than $v$: a bond does not send back to $v$ what it has just heard from $v$, which prevents the message from running in circles. Finally the state of an atom is $h_v = \\mathrm{ReLU}\\bigl(W_o\\,[x_v, \\sum_{k \\in N(v)} h_{k\\to v}^{(T)}]\\bigr)$ and the molecule is the mean of its atoms' states. The cell writes this forward pass for the example compound, with random weights, and checks the two properties just named:
order independence, and that the reach of an atom grows by one bond per step.
"""),
    code(MPNN),
    md("""
The molecule is described by a vector of the same length, whatever its size, and the vector is unchanged by a renumbering of the atoms ({{=invariant}}). The reach grows by one bond with each step, starting from one bond for no steps at all, because the final step already reads the bonds that arrive at an atom: after three steps the change of one atom reaches {{=field_rows[-1]['farthest of them, in bonds from the changed atom']}} bonds away and alters {{=n_atoms_affected}} atoms.
When the weights are learned from labelled molecules, the network chooses which features of the neighbourhood matter. The project uses Chemprop in two ways: trained from scratch on the eight potency endpoints (the candidate `chemprop_mt_shift` of `cypmode/challenge/nested_runners_chemprop.py`, which scores a compound by $\\hat t - \\max(\\hat d, 4)$), and, as a frozen encoder, **CheMeleon**, a Chemprop network that was pretrained by others on a large set of computed molecular descriptors (Burns and co-workers): each molecule is turned into its 2048-number readout, which is reduced to 64 principal components (section 4.9) and offered to the other models as the block `chemeleon` (section 9.4).

**Networks on text and self-attention.** A SMILES string (section 2.1) is a text, and models that read text can read it. The most successful of them are **transformers** (Vaswani and co-workers, 2017). A text is cut into **tokens**, here atoms, bonds, brackets and ring digits, and every token gets a vector from a table, to which a vector encoding the *position* of the token is added, since the machinery below does not otherwise know the order. The heart of a transformer is **self-attention**:
each token looks at all the others and takes a weighted average of what it finds. From the token vectors $X$ (one row per token) three linear maps give **queries** $Q = XW_Q$, **keys** $K = XW_K$ and **values** $V = XW_V$. The score of token $j$ for token $i$ is the dot product of the query of $i$ with the key of $j$, the scores of a row are turned into weights that sum to 1 by the **softmax** $\\mathrm{softmax}(s)_j = e^{s_j}/\\sum_l e^{s_l}$, and the output of token $i$ is the weighted average of the values:

$$\\mathrm{Attention}(Q, K, V) \\;=\\; \\mathrm{softmax}\\!\\Bigl(\\frac{Q K^{\\top}}{\\sqrt{d_k}}\\Bigr)\\,V ,$$

with $d_k$ the length of a key. The division by $\\sqrt{d_k}$ has a reason that the cell shows: if the entries of $q$ and $k$ are independent with variance 1, the dot product $q\\cdot k = \\sum_{l=1}^{d_k} q_l k_l$ has variance $d_k$, so for long keys the scores are large, the softmax saturates to a one-hot vector and its gradients vanish; dividing by $\\sqrt{d_k}$ brings the variance back to 1.
A transformer layer repeats this with several heads in parallel (each with its own maps), adds a small network applied to every token, and wraps both in residual connections and normalisation. **ChemBERTa** (Chithrananda and co-workers, 2020, who pretrained on 77 million SMILES strings from PubChem; the checkpoint used here is the smaller release `seyonec/ChemBERTa-zinc-base-v1`, `cypmode/challenge/chemberta_features.py`, whose model card states 100 thousand SMILES strings from ZINC, a public catalogue of purchasable compounds) is a transformer of the RoBERTa kind that was pretrained by *masked-language modelling* on those strings: a fraction of the tokens of each string is hidden and
the network is trained to predict them, which forces it to learn the grammar and the chemistry of the strings without any label. The project took its 768-number mean-pooled embedding of every compound. Joined naively to the other features it lowered the AUC on CYP3A4, by an amount that the corrected test of Part VI does not separate from chance (section 7.9), and the project reduces such embeddings before use (as with CheMeleon). The cell writes the attention of one head on the example's SMILES.
"""),
    code(ATTENTION),
    md("""
The SMILES of the example is cut into {{=n_tokens}} tokens of {{=n_vocabulary}} kinds; the attention matrix has one row of weights per token and every row sums to 1. The table shows the reason for the division: the variance of the dot product grows with the key length and the largest attention weight of an unscaled softmax approaches 1 for long keys, while the scaled scores keep a variance of 1 and a softer pattern.

## 4.9 Principal component analysis, and tabular in-context learning

> **Field:** Machine learning · Mathematics (linear algebra)

**Principal component analysis** (PCA; Pearson, 1901; Hotelling, 1933) finds the directions in which a table varies most, so that a table of 217 correlated columns can be summarised by a few new columns that lose little. Take a table $X$ with $n$ rows and $p$ columns, each column with its mean subtracted (and, for the descriptors, divided by its standard deviation, so that no column dominates by its units). The variation along a direction given by a unit vector $w$ (length 1) is the variance of the projections $Xw$, which is
$w^{\\top} C\\, w$ with the **covariance matrix** $C = X^{\\top}X/(n - 1)$, whose entry $C_{ab}$ says how much columns $a$ and $b$ vary together. To find the direction of the most variance, maximise $w^{\\top}Cw$ subject to $w^{\\top}w = 1$. With a multiplier $\\lambda$ for the constraint, the maximum is where the gradient of $w^{\\top}Cw - \\lambda(w^{\\top}w - 1)$ vanishes, $2Cw - 2\\lambda w = 0$, that is

$$C\\,w = \\lambda\\,w :$$

$w$ is an **eigenvector** of $C$ and the variance along it, $w^{\\top}Cw = \\lambda$, is its **eigenvalue**. The first principal component is the eigenvector with the largest eigenvalue; the next is the eigenvector of the next largest eigenvalue, which is automatically perpendicular to the first because $C$ is symmetric; and so on. The share of the total variance on component $k$ is $\\lambda_k/\\sum_l \\lambda_l$, and keeping the first $m$ components keeps $\\sum_{k \\le m}\\lambda_k/\\sum_l \\lambda_l$ of it. The cell does it with a matrix library routine for eigenvectors and checks the results against scikit-learn.
"""),
    code(PCA_CELL),
    md("""
The first two components carry {{=share_two:.0%}} of the variance of the descriptors, the first ten {{=share_ten:.0%}} and the first 64 {{=share_64:.0%}}; the hand calculation agrees with scikit-learn to rounding. The picture is another view of the covariate shift of section 2.7: the test compounds do not occupy the same part of the plane as the training compounds. PCA is used twice in the project: to reduce the 2048-number embedding of CheMeleon to 64 components (`cypmode/challenge/frozen_embeddings.py`; the components are fitted to the structures of
both files, which needs no label), and, in the form called *truncated SVD* (the **singular value decomposition**, a factorisation of the table itself that finds the same directions as the eigenvectors of $C$, here without subtracting the mean, which suits sparse count tables), to reduce the Morgan counts for the next model.

**Tabular in-context learning.** The models of this part are *fitted* to each training set: their parameters are adjusted by an optimisation. **TabICL** (Qu and co-workers, 2025; related to TabPFN, Hollmann and co-workers) is different: it is a **foundation model**, a model pretrained once on a very large collection of data and a general task and then reused for many specific tasks, here a transformer pretrained on millions of artificial tables generated from random processes, to read a table of labelled rows together with unlabelled rows to be predicted and to output the missing labels in one forward pass. The labelled rows are the **context**, in the way the text before a question is the context
for a language model, and no weight changes when a new table arrives: "training" is only the reading of the context, which is why the method is called *in-context learning*. Since the networks are pretrained on a wide range of tables, they are strong on small tables, such as the 2300 labelled compounds of one task. The project's candidate `tabicl_svd` (`cypmode/challenge/nested_runners_torch.py`) gives it 128 components of the truncated SVD of the logarithm of the Morgan counts together with the quantile-normalised descriptors, with two
ensemble members on the CPU. It runs in the separate torch environment, so the cell does not execute it; its result is one of the candidates of the nested evaluation (section 8.1).

## 4.10 Metrics: how a model is judged

> **Field:** Statistics · Machine learning

A model's output has to be turned into numbers that say how good it is. Which number depends on what the model will be used for, and the choice matters because the challenge's labels are unbalanced (section 2.10).

**The confusion matrix and the numbers computed from it.** If a model's scores are turned into calls by a threshold, every compound falls into one of four cells: true positive (TP, called positive and is), false positive (FP), false negative (FN) and true negative (TN). From them: **accuracy** $(TP + TN)/n$, **precision** $TP/(TP + FP)$ (how much to trust a positive call), **recall** $TP/(TP + FN)$ (how many of the positives are found) and **F1**, their harmonic mean $2PR/(P + R)$.
Accuracy is misleading when one class is rare: a model that calls everything negative has an accuracy of $1 - \\pi$, with $\\pi$ the positive rate, and finds nothing. The **Matthews correlation coefficient** (MCC; Matthews, 1975) uses all four cells:

$$\\mathrm{MCC} \\;=\\; \\frac{TP\\cdot TN - FP\\cdot FN}{\\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} .$$

It is the ordinary (Pearson) correlation between two yes/no variables, the label $y$ and the call $\\hat y$, which is why it is 0 for a model that is no better than chance, 1 for a perfect one and $-1$ for a perfectly wrong one. To see this, call $n = TP + FP + FN + TN$. The covariance of $y$ and $\\hat y$ is $E[y\\hat y] - E[y]E[\\hat y] = TP/n - (TP + FN)(TP + FP)/n^2 = (TP\\cdot TN - FP\\cdot FN)/n^2$, because $n\\,TP - (TP + FN)(TP + FP) = TP\\cdot TN - FP\\cdot FN$ when $n$ is expanded; and
the variance of a yes/no variable with probability $P$ of yes is $P(1 - P)$, so the product of the two standard deviations is $\\sqrt{(TP + FN)(TN + FP)(TP + FP)(TN + FN)}/n^2$; the ratio is the formula. **The challenge scores the MCC of the calls**, so the TDI track is judged at one chosen threshold. The cell computes all of them for a real set of out-of-fold predictions.
"""),
    code(CONFUSION),
    md("""
The 37 % of compounds with the highest scores are called positive: {{=TP}} true positives, {{=FP}} false positives, {{=FN}} false negatives and {{=TN}} true negatives. The accuracy is {{=accuracy:.3f}}, against {{=accuracy_if_all_negative:.3f}} for calling every compound negative: accuracy would not have told the model from the useless one. The MCC computed by the formula, by the library and as a correlation coefficient are the same number ({{=mcc_formula:.4f}}, {{=mcc_library:.4f}}, {{=mcc_pearson:.4f}}).

**Threshold-free metrics.** Calls depend on a threshold, which has to be chosen, while the ranking that the model produces does not. The **ROC curve** (receiver operating characteristic) plots, as the threshold runs from high to low, the **true-positive rate** (recall) against the **false-positive rate** $FP/(FP + TN)$; a ranking of the positives above the negatives gives a curve that hugs the upper left corner, a random ranking the diagonal. The **area under the ROC curve** (AUC) has a meaning that makes it easy to interpret. With $S^{+}$ and $S^{-}$ the scores of a random positive and a random negative compound, and threshold $\\tau$, the true-positive rate is
$P(S^{+} > \\tau)$ and the false-positive rate is $P(S^{-} > \\tau)$, so that, writing $f^{-}$ for the density of the negative scores,

$$\\mathrm{AUC} \\;=\\; \\int \\mathrm{TPR}\\; d(\\mathrm{FPR}) \\;=\\; \\int P(S^{+} > \\tau)\\,f^{-}(\\tau)\\,d\\tau \\;=\\; P(S^{+} > S^{-}) :$$

the AUC is the probability that a random positive compound is scored above a random negative one (with ties counting half), and this is the **Mann-Whitney** statistic (Mann and Whitney, 1947) divided by the number of pairs. It is 0.5 for chance and independent of the positive rate. The **precision-recall curve** plots precision against recall, and the **average precision** (AP) is its area as a step sum, $\\mathrm{AP} = \\sum_n (R_n - R_{n-1})P_n$, which for a ranking without ties is the average, over the positive compounds, of the precision at the position of each one. Unlike the AUC, it depends on the positive rate:
a random ranking has an AP equal to $\\pi$. The cell computes the AUC three ways and the AP two ways.
"""),
    code(ROC),
    md("""
The three AUCs agree ({{=auc_trapezoid:.5f}} by the trapezoid rule, {{=auc_pairs:.5f}} as a probability over all pairs, {{=auc_library:.5f}} from the library), and so do the two APs ({{=ap_formula:.5f}} and {{=ap_library:.5f}}). **Why the MCC of a rare class is low even for a good ranking.** The AUC does not depend on the positive rate, but the MCC at the best threshold does, and so does the AP. The cell holds the ranking quality fixed (an AUC of 0.75) and changes only the positive rate:
"""),
    code(PREVALENCE_MCC),
    md("""
The same ranking quality gives a best MCC of {{=mcc_at_rate[0.5]:.2f}} at a positive rate of 0.5, {{=mcc_at_rate[0.3]:.2f}} at 0.3 (about that of CYP3A4 on the test set) and only {{=mcc_at_rate[0.07]:.2f}} at 0.07 (about that of CYP2D6): the low scores of the second enzyme on the board are in part a consequence of its rarity and not only of a weaker model (section 8.7). A number is comparable only to numbers at the same positive rate.

**Probabilities as probabilities.** A model that outputs 0.8 should be right about 80 % of the time on compounds to which it gives 0.8. That property is **calibration**, and it matters when a probability is used for anything other than a ranking. Three numbers measure it. The **Brier score** (Brier, 1950) is the mean squared difference between probability and label, $\\mathrm{BS} = \\frac1n\\sum_i (p_i - y_i)^2$; a model that always says the positive rate $\\pi$ has $\\mathrm{BS}_0 = \\pi(1 - \\pi)$, and
$1 - \\mathrm{BS}/\\mathrm{BS}_0$ is the skill relative to that. The **expected calibration error** divides the probabilities into bins (here ten) and averages the gap between the mean probability and the observed positive share in a bin, weighted by the size of the bin, $\\mathrm{ECE} = \\sum_b (n_b/n)\\,\\lvert \\bar p_b - \\bar y_b\\rvert$. The **calibration slope** is the coefficient $\\beta$ of a logistic regression of the label on the logit of the probability, $P(y = 1) = \\sigma(\\alpha + \\beta\\,\\mathrm{logit}(p))$; for a calibrated model $\\alpha = 0$ and $\\beta = 1$, a slope below 1 means the model is
over-confident (its probabilities are too extreme) and above 1 under-confident. The project's rule for a usable probability is an ECE of at most 0.05 and a slope between 0.8 and 1.2 (section 8.5).
"""),
    code(CALIBRATION),
    md("""
For these out-of-fold predictions the Brier score is {{=brier:.4f}} against {{=brier_reference:.4f}} for the constant prediction (a skill of {{=brier_skill:.1%}}), the ECE is {{=ece_value:.3f}} and the calibration slope is {{=slope_value:.2f}}, and the project's own function gives the same ECE as the hand calculation. The raw probabilities of the model {{=calibration_verdict}} the project's rule (an ECE of at most 0.05 and a slope between 0.8 and 1.2), which is why calibration is a separate step of the final model (section 8.5). These are the numbers by which every model in the rest of the notebook is judged, always on compounds that the model did not see.
"""),
]
