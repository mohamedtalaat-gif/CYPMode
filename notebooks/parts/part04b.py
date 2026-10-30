# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part IV (second part): neighbours and kernels, regression for the two arms, neural networks."""

from notebooks.notebook_builder import code, figure, md

KERNEL = '''
# The Tanimoto kernel on the 2334 compounds, and the two models built on it. K[i, j] is the Tanimoto similarity of compounds i and j (section 3.2). A kernel must be positive semi-definite: a quadratic
# form a'Ka is never negative, which is the same as K having no negative eigenvalue. The check is made on 600 compounds in double precision.
from sklearn.svm import SVC

from cypmode.challenge.grouping import morgan_bit_matrix
from cypmode.challenge.validation_matching import tanimoto_matrix

bits_ml = morgan_bit_matrix(train.loc[rows_ml, "SMILES"].tolist())
K = tanimoto_matrix(bits_ml)
eigenvalues = np.linalg.eigvalsh(K[:600, :600].astype(np.float64))
print(f"kernel matrix {K.shape[0]} x {K.shape[1]}, diagonal all 1: {bool(np.allclose(np.diag(K), 1))}, symmetric: {bool(np.allclose(K, K.T))}; smallest eigenvalue of a 600-compound block: {eigenvalues.min():.2e}")
smallest_eigenvalue = float(eigenvalues.min())

index_column = np.arange(len(y_ml), dtype=np.float64)[:, None]                  # the "features" handed to the models are the compound numbers; the kernel is read from K

def knn_fit_predict(k):
    def fit_predict(train_index, train_y, test_index):
        a, b = train_index[:, 0].astype(int), test_index[:, 0].astype(int)
        sims = K[np.ix_(b, a)]
        top = np.argsort(-sims, axis=1)[:, :k]
        w = np.take_along_axis(sims, top, axis=1)
        return (w * train_y[top]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)       # the similarity-weighted share of positive neighbours
    return fit_predict

def svm_fit_predict(C):
    def fit_predict(train_index, train_y, test_index):
        a, b = train_index[:, 0].astype(int), test_index[:, 0].astype(int)
        model = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(a, a)], train_y)
        return model.decision_function(K[np.ix_(b, a)])
    return fit_predict

kernel_rows = []
for k in (1, 5, 15, 40, 100):
    resubstitution = roc_auc_score(y_ml, knn_fit_predict(k)(index_column, y_ml, index_column))              # every compound scored against all compounds, itself included
    mean_auc, sd_auc = cross_validated_auc(knn_fit_predict(k), index_column)
    kernel_rows.append({"model": f"nearest neighbours, k = {k}", "AUC on the compounds it was fitted on": resubstitution, "mean AUC on held-out compounds": mean_auc, "standard deviation": sd_auc})
for C in (0.3, 1.0, 3.0):
    fit_rows, held = folds_ml[0]
    fitted = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(fit_rows, fit_rows)], y_ml[fit_rows])
    resubstitution = roc_auc_score(y_ml[fit_rows], fitted.decision_function(K[np.ix_(fit_rows, fit_rows)]))
    mean_auc, sd_auc = cross_validated_auc(svm_fit_predict(C), index_column)
    kernel_rows.append({"model": f"support vector machine, Tanimoto kernel, C = {C:g} ({len(fitted.support_) / len(fit_rows):.0%} of the compounds are support vectors)", "AUC on the compounds it was fitted on": resubstitution,
                        "mean AUC on held-out compounds": mean_auc, "standard deviation": sd_auc})
show_table(pd.DataFrame(kernel_rows), floatfmt=".3f")
knn_best = max(kernel_rows[:5], key=lambda r: r["mean AUC on held-out compounds"])
svm_best = max(kernel_rows[5:], key=lambda r: r["mean AUC on held-out compounds"])
knn_one = kernel_rows[0]
'''

ARMS = '''
# Regression for the two assay arms. First the claim of the text: clipping both arms at the floor of 4 leaves every label unchanged. Then four ways of scoring a compound, in the five folds of the other
# models: the classifier on the label; a regressor of the shift s = t - max(d, 4); two regressors, one per arm, scored by t_hat - max(d_hat, 4); and the same with both arms clipped at 4 before fitting.
from lightgbm import LGBMRegressor

from cypmode.challenge.nested_runners import LGBM_PRESETS
from cypmode.challenge.tdi_targets import label_from_shift

d_arm, t_arm = view_ml["direct"].to_numpy(), view_ml["tdi"].to_numpy()
label_original = label_from_shift(t_arm - np.maximum(d_arm, ASSAY_FLOOR))
label_clipped = label_from_shift(np.maximum(t_arm, ASSAY_FLOOR) - np.maximum(np.maximum(d_arm, ASSAY_FLOOR), ASSAY_FLOOR))
n_label_changes = int((label_original != label_clipped).sum())
print(f"labels that change when both arms are clipped at {ASSAY_FLOOR:g}: {n_label_changes} of {len(label_original)}; share of the t values below the floor: {(t_arm < ASSAY_FLOOR).mean():.1%}, of the d values: {(d_arm < ASSAY_FLOOR).mean():.1%}")

params = LGBM_PRESETS["deep"]
def regressor():
    return LGBMRegressor(verbose=-1, n_jobs=2, random_state=0, **params)

def fit_two_arms(train_x, train_rows, test_x, clip):
    t_target, d_target = t_arm[train_rows], d_arm[train_rows]
    if clip:
        t_target, d_target = np.maximum(t_target, ASSAY_FLOOR), np.maximum(d_target, ASSAY_FLOOR)
    return regressor().fit(train_x, t_target).predict(test_x), regressor().fit(train_x, d_target).predict(test_x)

arm_scores = {"classifier on the label": [], "regressor of the shift s": [], "two arms, t_hat - max(d_hat, 4)": [], "two arms clipped at 4, t_hat - max(d_hat, 4)": []}
arm_r2 = {"t": [], "d": []}
truths = []
for fit_rows, held in folds_ml:
    truths.append(y_ml[held])
    arm_scores["classifier on the label"].append(LGBMClassifier(verbose=-1, n_jobs=2, random_state=0, **params).fit(set_a[fit_rows], y_ml[fit_rows]).predict_proba(set_a[held])[:, 1])
    arm_scores["regressor of the shift s"].append(regressor().fit(set_a[fit_rows], shift_ml[fit_rows]).predict(set_a[held]))
    t_hat, d_hat = fit_two_arms(set_a[fit_rows], fit_rows, set_a[held], clip=False)
    arm_scores["two arms, t_hat - max(d_hat, 4)"].append(t_hat - np.maximum(d_hat, ASSAY_FLOOR))
    arm_r2["t"].append(1 - np.sum((t_arm[held] - t_hat) ** 2) / np.sum((t_arm[held] - t_arm[held].mean()) ** 2))
    arm_r2["d"].append(1 - np.sum((d_arm[held] - d_hat) ** 2) / np.sum((d_arm[held] - d_arm[held].mean()) ** 2))
    t_hat, d_hat = fit_two_arms(set_a[fit_rows], fit_rows, set_a[held], clip=True)
    arm_scores["two arms clipped at 4, t_hat - max(d_hat, 4)"].append(t_hat - np.maximum(d_hat, ASSAY_FLOOR))
arm_rows = [{"way of scoring a compound": name, "mean AUC (5 folds)": float(np.mean([roc_auc_score(t, s) for t, s in zip(truths, scores)])),
             "standard deviation": float(np.std([roc_auc_score(t, s) for t, s in zip(truths, scores)]))} for name, scores in arm_scores.items()]
show_table(pd.DataFrame(arm_rows), floatfmt=".3f")
print(f"R squared of the arm regressors on held-out compounds: t {np.mean(arm_r2['t']):.2f}, d {np.mean(arm_r2['d']):.2f}")
arm_auc = {r["way of scoring a compound"]: r["mean AUC (5 folds)"] for r in arm_rows}
r2_t, r2_d = float(np.mean(arm_r2["t"])), float(np.mean(arm_r2["d"]))
'''

TINY_NET = '''
# A neural network with one hidden layer, written out with its gradient. Input x (p numbers) -> hidden layer h = tanh(W1 x + b1) of H units -> logit z = w2 . h + b2 -> p = sigmoid(z); loss = log-loss plus
# an L2 penalty on the weights. The gradient is derived in the text; here it is compared with a finite-difference estimate for every single weight, and then the network is trained with Adam.
class TinyNet:
    def __init__(self, n_inputs, hidden, seed=0, l2=1e-3):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_inputs), (hidden, n_inputs)); self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden); self.b2 = 0.0
        self.l2 = l2

    def forward(self, X):
        self.X = X
        self.h = np.tanh(X @ self.W1.T + self.b1)
        self.z = self.h @ self.w2 + self.b2
        return self.z

    def loss(self, X, y):
        z = self.forward(X)
        return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * self.l2 * (np.sum(self.W1**2) + np.sum(self.w2**2)))    # log(1 + e^z) - y z is the log-loss written with the logit

    def gradients(self, X, y):
        z = self.forward(X)
        n = len(y)
        delta2 = (sigmoid(z) - y) / n                                                # dL/dz for every compound, averaged
        grad_w2 = self.h.T @ delta2 + self.l2 * self.w2
        grad_b2 = delta2.sum()
        delta1 = np.outer(delta2, self.w2) * (1 - self.h**2)                         # back through the tanh: d tanh(a)/da = 1 - tanh(a)^2
        grad_W1 = delta1.T @ X + self.l2 * self.W1
        grad_b1 = delta1.sum(axis=0)
        return grad_W1, grad_b1, grad_w2, grad_b2

# Gradient check on 40 compounds and 6 descriptors, every weight perturbed in turn.
rng = np.random.default_rng(1)
Xc, yc = rng.normal(size=(40, 6)), (rng.random(40) < 0.4).astype(float)
net = TinyNet(6, 5, seed=2)
analytic = net.gradients(Xc, yc)
worst = 0.0
for array, grad in zip((net.W1, net.b1, net.w2), analytic[:3]):
    flat, flat_grad = array.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        saved = flat[i]
        flat[i] = saved + 1e-6; up = net.loss(Xc, yc)
        flat[i] = saved - 1e-6; down = net.loss(Xc, yc)
        flat[i] = saved
        worst = max(worst, abs((up - down) / 2e-6 - flat_grad[i]))
saved_b2 = net.b2
net.b2 = saved_b2 + 1e-6; up = net.loss(Xc, yc); net.b2 = saved_b2 - 1e-6; down = net.loss(Xc, yc); net.b2 = saved_b2
worst = max(worst, abs((up - down) / 2e-6 - analytic[3]))
n_weights_checked = net.W1.size + net.b1.size + net.w2.size + 1
print(f"largest difference between the analytic gradient and the finite difference over all {n_weights_checked} weights: {worst:.1e}")
gradient_gap = worst

def train_net(X, y, hidden=32, epochs=150, rate=0.01, l2=1e-3, seed=0, watch=None):
    """Full-batch Adam; returns the network and, if watch = (X_watch, y_watch) is given, the AUC on the watched compounds every 25 epochs."""
    net = TinyNet(X.shape[1], hidden, seed=seed, l2=l2)
    state = [[np.zeros_like(a), np.zeros_like(a)] for a in (net.W1, net.b1, net.w2)] + [[np.zeros(1), np.zeros(1)]]
    history = []
    for t in range(1, epochs + 1):
        grads = net.gradients(X, y)
        params = [net.W1, net.b1, net.w2, np.array([net.b2])]
        for (m, v), param, grad in zip(state, params, grads):
            g = np.asarray(grad, dtype=float).reshape(param.shape)
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g**2
            param -= rate * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        net.b2 = float(params[3][0])
        if watch is not None and t % 25 == 0:
            history.append({"epoch": t, "AUC on the fitted compounds": roc_auc_score(y, net.forward(X)), "AUC on the held-out compounds": roc_auc_score(watch[1], net.forward(watch[0]))})
    return net, history

def standardise(train_x, other_x):
    mu, sd = train_x.mean(axis=0), train_x.std(axis=0)
    sd[sd == 0] = 1.0
    return np.clip((train_x - mu) / sd, -5, 5), np.clip((other_x - mu) / sd, -5, 5)

def net_fit_predict(hidden, epochs, l2):
    def fit_predict(train_x, train_y, test_x):
        a, b = standardise(train_x, test_x)
        net, _ = train_net(a, train_y.astype(float), hidden=hidden, epochs=epochs, l2=l2)
        return net.forward(b)
    return fit_predict

fit_rows, held = folds_ml[0]
a, b = standardise(desc_ml[fit_rows], desc_ml[held])
_, net_history = train_net(a, y_ml[fit_rows].astype(float), hidden=32, epochs=200, l2=1e-3, watch=(b, y_ml[held]))
show_table(pd.DataFrame(net_history), floatfmt=".3f")
net_rows = []
for hidden, epochs, l2 in ((32, 50, 1e-3), (32, 200, 1e-3), (32, 200, 1e-1)):
    mean_auc, sd_auc = cross_validated_auc(net_fit_predict(hidden, epochs, l2), desc_ml)
    net_rows.append({"network": f"{hidden} hidden units, {epochs} epochs, L2 penalty {l2:g}", "mean AUC (5 folds)": mean_auc, "standard deviation": sd_auc})
show_table(pd.DataFrame(net_rows), floatfmt=".3f")
net_best = max(net_rows, key=lambda r: r["mean AUC (5 folds)"])
net_history_last = net_history[-1]
'''

MASKED = '''
# The masked loss of a multitask network on three compounds and two heads (two enzymes), with some labels missing. Masking sums the loss only over the labels that exist; filling the missing ones with
# "negative" is a different loss, and gives a different number.
logits = np.array([[0.8, -0.2], [-1.0, 0.5], [2.0, 0.1]])
targets = np.array([[1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]])
present = ~np.isnan(targets)
per_entry = np.where(present, log_loss(np.nan_to_num(targets), logits), 0.0)
masked_mean = per_entry.sum() / present.sum()
imputed_mean = log_loss(np.nan_to_num(targets, nan=0.0), logits).mean()
print(f"labels present: {int(present.sum())} of {present.size}; masked mean loss {masked_mean:.4f}; loss if the missing labels were filled in as negative {imputed_mean:.4f}")
gradient_masked = np.where(present, (sigmoid(logits) - np.nan_to_num(targets)) / present.sum(), 0.0)
print("gradient of the masked loss with respect to the logits (rows: compounds; zero where the label is missing):")
print(np.round(gradient_masked, 4))
masked_values = (float(masked_mean), float(imputed_mean))
'''

CELLS = [
    md("""
## 4.5 Nearest neighbours and support vector machines with the Tanimoto kernel

> **Field:** Machine learning · Mathematics (kernels)

**Nearest neighbours.** The simplest model of all has no training at all. To score a compound, find the $k$ most similar compounds in the training set and let them vote. The project weights the votes by similarity: with $N_k(x)$ the set of the $k$ training compounds most similar to $x$, and $s(x, x_i)$ the Tanimoto similarity of $x$ to $x_i$, and $\\hat p(x)$ the resulting estimate of the probability that $x$ is positive,

$$\\hat p(x) \\;=\\; \\frac{\\sum_{i \\in N_k(x)} s(x, x_i)\\, y_i}{\\sum_{i \\in N_k(x)} s(x, x_i)},$$

the similarity-weighted share of positive neighbours. It encodes the idea that similar compounds have similar labels (section 3.2 warned how far that holds). The number $k$ sets the trade-off of section 4.1: with $k = 1$ the score follows the single closest compound (high variance), with a large $k$ it approaches the overall rate (high bias).

**Support vector machines.** A different way to use similarity is to look for a rule that is a weighted sum of similarities to *some* training compounds, the **support vectors**. Write the label as $\\tilde y_i = \\pm 1$ and the score as a linear function in an (imagined) feature space, $f(x) = w \\cdot \\phi(x) + b$, where $\\phi$ maps a compound to that space. The linear function that separates the classes with the widest possible margin
is the one preferred: the distance from the separating surface $f = 0$ to the nearest compound on either side is $1/\\lVert w \\rVert$ when the compounds closest to the surface have $|f| = 1$, so the widest margin means the smallest $\\lVert w \\rVert$. Real data cannot be separated perfectly, so every compound is allowed to violate the margin by an amount $\\xi_i \\ge 0$ at a price, and the problem is

$$\\min_{w, b, \\xi}\\;\\; \\tfrac12\\lVert w\\rVert^2 + C\\sum_{i=1}^{n}\\xi_i \\qquad\\text{subject to}\\qquad \\tilde y_i\\,(w\\cdot\\phi(x_i) + b) \\ge 1 - \\xi_i,\\;\\; \\xi_i \\ge 0 ,$$

where $C > 0$ balances a wide margin against violations; at the optimum $\\xi_i = \\max(0, 1 - \\tilde y_i f(x_i))$, the **hinge loss**, which is zero for compounds beyond the margin. Introducing a multiplier $\\alpha_i$ for each constraint and eliminating $w$, $b$ and $\\xi$ turns this into a problem in the $\\alpha_i$ alone:

$$\\max_{\\alpha}\\;\\; \\sum_i \\alpha_i - \\tfrac12\\sum_{i,j}\\alpha_i\\alpha_j\\,\\tilde y_i \\tilde y_j\\,K(x_i, x_j) \\qquad\\text{subject to}\\qquad 0 \\le \\alpha_i \\le C,\\;\\; \\sum_i \\alpha_i \\tilde y_i = 0,\\qquad f(x) = \\sum_i \\alpha_i \\tilde y_i\\,K(x_i, x) + b .$$

The map $\\phi$ appears only through the **kernel** $K(x, x') = \\phi(x)\\cdot\\phi(x')$, a similarity function: one never needs to compute $\\phi$, only $K$ (the **kernel trick**). Any function that is symmetric and positive semi-definite, in the sense that every matrix $[K(x_i, x_j)]$ has no negative eigenvalue, is a kernel of some $\\phi$ (Mercer's condition). The Tanimoto similarity of two binary fingerprints is such a function (Ralaivola and co-workers, 2005; Swamidass and co-workers, 2005), which is why a similarity that
chemists already trust can be used directly: $K = T$. Only the compounds with $\\alpha_i > 0$, the support vectors, appear in $f$; those on or inside the margin. The project's SVMs use `class_weight='balanced'`, which multiplies $C$ for each class by $n / (2 n_{\\text{class}})$ so that the rarer class (the positives) is not outvoted. The cell builds the kernel on the 2334 compounds, checks its positive semi-definiteness numerically, and compares the two models
with several values of $k$ and $C$, including the score of each on the compounds it was fitted on.
"""),
    code(KERNEL),
    md("""
The kernel has no negative eigenvalue (the smallest on this block is {{=smallest_eigenvalue:.2f}}, so the block is even positive definite), which is what a kernel must satisfy. The nearest-neighbour model with $k = 1$ reaches an AUC of {{=knn_one['AUC on the compounds it was fitted on']:.2f}} on the compounds it was fitted on, each one being its own nearest neighbour, and {{=knn_one['mean AUC on held-out compounds']:.3f}} on held-out ones: the memorising student again. The best of the neighbour models is {{=knn_best['model']}} ({{=knn_best['mean AUC on held-out compounds']:.3f}}),
the best SVM reaches {{=svm_best['mean AUC on held-out compounds']:.3f}}. Both work from the similarity alone and know nothing of descriptors, which is why they differ in kind from the tree models and are worth keeping in a collection of models that is blended at the end (section 8.2): when different models make different mistakes, their blend can be better than any of them. The project's candidates are `knn_tanimoto_k5`, `k15`, `k40` and `svm_tanimoto_C0.3`, `C1`, `C3`.

"""),
    figure("ml_tanimoto_kernel"),
    md("""
## 4.6 Regression for the two assay arms, and floor-clipping

> **Field:** Machine learning · Pharmacology

The label is a threshold on a continuous quantity (section 2.3), and a classifier is fitted to the label alone, which throws away how far a compound is from the threshold. A **regression** model predicts a number instead. The label depends on two measured numbers, so there are three natural ways to use them: predict the **shift** $s = t - \\max(d, 4)$ directly and call a compound positive when the prediction is above the threshold; predict the two arms $t$ and $d$ with separate models
and combine them as $\\hat t - \\max(\\hat d, 4)$; or predict the arms after **clipping** them at the floor. The squared-error loss of section 4.1 is used, whose best prediction is the conditional mean.

**Why clipping leaves the label alone.** The label is positive when $t - \\max(d, 4) > \\log_{10} 2$. Replace $d$ by $d' = \\max(d, 4)$: then $\\max(d', 4) = \\max(d, 4)$ and the label does not change. Replace $t$ by $t' = \\max(t, 4)$: if $t > 4$ nothing changes; if $t \\le 4$ then $t - \\max(d, 4) \\le 4 - 4 = 0 < \\log_{10} 2$, so the label is negative, and with $t' = 4$ the difference is $t' - \\max(d, 4) \\le 0$, which is again negative. So both arms can be clipped at 4 without changing any label, while the regression no longer has to explain differences between values below the floor, which are mostly measurement noise
(section 1.7). The cell verifies the claim on all compounds and compares the four ways of scoring.
"""),
    code(ARMS),
    md("""
Clipping both arms at 4 changes {{=n_label_changes}} labels, as the proof says. The classifier on the label reaches {{=arm_auc['classifier on the label']:.3f}}; the regressor of the shift {{=arm_auc['regressor of the shift s']:.3f}}; the two arms scored together {{=arm_auc['two arms, t_hat - max(d_hat, 4)']:.3f}}, and with clipping {{=arm_auc['two arms clipped at 4, t_hat - max(d_hat, 4)']:.3f}}. The arms themselves are predicted with $R^2$ values of {{=r2_t:.2f}} ($t$) and {{=r2_d:.2f}} ($d$), where $R^2 = 1 - \\sum (y - \\hat y)^2 / \\sum (y - \\bar y)^2$ is the fraction of the variance of the
quantity that the model explains. These are single-split numbers on one task; whether the arm models add anything to a blend of classifiers is the question that section 9.6 answers with the full evaluation, and the project keeps them as members of the stack (`lgbm_reg_shift_*`, `lgbm_reg_arms_*`).

## 4.7 Neural networks: the perceptron, backpropagation and shared-trunk networks for several tasks

> **Field:** Machine learning · Mathematics (calculus)

**A neuron and a layer.** An artificial neuron computes a weighted sum of its inputs plus a constant, $a = \\sum_k w_k x_k + b$, and passes it through a non-linear **activation function** $\\varphi$ to give its output $\\varphi(a)$. The earliest form, the **perceptron** (Rosenblatt, 1958), used a step as its activation. A **layer** is many neurons reading the same inputs: with a weight matrix $W$ (one row per neuron) and a vector $b$, it computes $\\varphi(Wx + b)$. A **network** stacks layers, the outputs of one being the inputs of
the next. Without the non-linearity a stack of layers would collapse into a single linear map, $W_2(W_1 x) = (W_2 W_1)x$; with it, a network with one hidden layer can approximate any continuous function on a bounded region if the layer is wide enough (Cybenko, 1989; Hornik, 1991). The common activations are $\\tanh(a)$, which squashes to the interval $(-1, 1)$, and the **rectified linear unit** $\\mathrm{ReLU}(a) = \\max(0, a)$.

**Training by backpropagation.** For a network with one hidden layer of $H$ units, $h = \\tanh(W_1 x + b_1)$, logit $z = w_2\\cdot h + b_2$ and probability $p = \\sigma(z)$, the log-loss of section 4.1 is to be minimised by gradient descent, which needs its gradient with respect to every weight. **Backpropagation** (Rumelhart, Hinton and Williams, 1986) is the chain rule applied from the output backwards. Write $\\delta_2 = \\partial L/\\partial z = p - y$ (section 4.1). Then
$\\partial L/\\partial w_2 = \\delta_2\\,h$ and $\\partial L/\\partial b_2 = \\delta_2$, since $z$ depends on $w_2$ through $h$ only linearly. The hidden activations receive $\\partial L/\\partial h = \\delta_2\\,w_2$; the pre-activation $a = W_1 x + b_1$ of the hidden layer is passed through the tanh, whose derivative is $1 - \\tanh^2(a) = 1 - h^2$, so $\\delta_1 = \\partial L/\\partial a = \\delta_2\\,w_2 \\odot (1 - h^2)$ (the symbol $\\odot$ is multiplication element by element), and
$\\partial L/\\partial W_1 = \\delta_1\\,x^{\\top}$, $\\partial L/\\partial b_1 = \\delta_1$. Averaged over the compounds and with an **L2 penalty** $\\tfrac12 \\ell\\,\\lVert W\\rVert^2$ added to keep the weights small (the gradient of the penalty is $\\ell\\,W$), these are the quantities the code below computes. The optimiser used is **Adam** (Kingma and Ba, 2015), which keeps running averages of the gradient $g$ and of its square, $m \\leftarrow \\beta_1 m + (1 - \\beta_1)g$ and
$v \\leftarrow \\beta_2 v + (1 - \\beta_2)g^2$, corrects their start-up bias ($\\hat m = m/(1 - \\beta_1^t)$, $\\hat v = v/(1 - \\beta_2^t)$ after $t$ steps) and moves each weight by $\\eta\\,\\hat m/(\\sqrt{\\hat v} + \\varepsilon)$: every weight gets a step of its own size. The cell compares the analytic gradient with a finite difference for every weight and then trains the network on the descriptors.
"""),
    code(TINY_NET),
    md("""
The analytic gradient agrees with the finite difference over all {{=n_weights_checked}} weights to within {{=gradient_gap:.1e}}, so the backpropagation formulas are right. On the first fold the AUC of the network on the compounds it was fitted on climbs with the number of epochs (an *epoch* is one pass over the training data) to {{=net_history_last['AUC on the fitted compounds']:.2f}}, while on the held-out compounds it ends at {{=net_history_last['AUC on the held-out compounds']:.2f}}; the table of the five-fold results shows that a stronger L2 penalty or fewer epochs
changes the balance. The best of the three settings reaches {{=net_best['mean AUC (5 folds)']:.3f}} ({{=net_best['network']}}). A network of this size on a table of descriptors is not the best model for these data (section 8.1 gives the full comparison); its importance is as the basis of the networks that follow.

**The shared-trunk multitask network.** The training file carries measurements for four enzymes and two arms, but a given compound has only some of them (section 2.2). A **multitask** network has one **trunk**, a stack of layers shared by all tasks, which turns a compound into a representation, and one **head** per task, a small network that reads the representation and produces that task's output. The hope is that a representation that must serve several tasks captures what the tasks have in common. Compounds lack some labels, and
the missing ones must not be filled in with a guess. The loss is therefore **masked**: it is summed over the labels that exist only, $L = \\bigl(\\sum_{i,k} m_{ik}\\,\\ell_{ik}\\bigr)/\\sum_{i,k} m_{ik}$, with $m_{ik} = 1$ if compound $i$ has a label for task $k$ and 0 otherwise, so that a missing label contributes no loss and no gradient. The cell shows the difference between masking and filling the missing labels with "negative".
"""),
    code(MASKED),
    md("""
Masked, the loss over the {{=int(present.sum())}} labels that exist is {{=masked_values[0]:.3f}}; with the missing labels filled in as negative it would be {{=masked_values[1]:.3f}}, a different number that rewards the model for predicting inhibition of the unknown, and the gradient of the masked loss is exactly zero where the label is missing. The project has two networks of this kind. The first, `cypmode/challenge/multitask.py`, has a trunk of two layers of 64 units with dropout, and a head of 32 units for each of the two enzymes, trained with the masked binary
cross-entropy and Adam. The second, in `cypmode/challenge/nested_runners_torch.py`, is the one that enters the nested evaluation: the input is the logarithm of one plus the Morgan counts together with quantile-normalised descriptors, the trunk has two layers of 1024 and 512 units, each followed by **batch normalisation** (every unit's activation is shifted and scaled to mean 0 and variance 1 over a mini-batch of 128 compounds, $\\hat a = (a - \\mu_B)/\\sqrt{\\sigma_B^2 + \\varepsilon}$, followed by a learned scale and shift, which makes training faster and steadier), a ReLU
and **dropout** of 0.35 (during training every unit is switched off with probability 0.35, so that no unit can be relied on and the network cannot memorise; at prediction time all units are on, scaled to match). One head regresses the eight standardised potency endpoints (direct and TDI-arm pIC50 of four enzymes, masked where unmeasured) and one head classifies the label of the task; the loss is the sum of the masked squared error and the masked cross-entropy; the optimiser is AdamW (Adam with the weight decay applied to the weights directly and not through the gradient; Loshchilov and Hutter, 2019) with a one-cycle schedule of the learning rate that rises to $2\\times10^{-3}$ and falls again over 60 epochs; the scores of three seeds (a **seed** is the number that starts the random number generator, so that a run can be repeated exactly; a different seed is a different random start of the same network) are averaged, and the candidate's score is either the label head or the predicted shift $\\hat t - \\max(\\hat d, 4)$. The network may learn from the compounds outside the plan's pool as well, whose arm values give it far more supervision than the label of the pool alone; it never sees an outer-validation row (Part VI). Torch and LightGBM cannot be imported into one process on the machine of this project (a recorded segmentation fault), so the networks run in a separate environment and write their predictions to the cache that the rest reads.
"""),
    figure("ml_backprop"),
]
