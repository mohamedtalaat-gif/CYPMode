# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part V (second part): the panel data and the odds ratio (5.3), the test against potency (5.4)."""

from notebooks.notebook_builder import code, md

PANEL = '''
# The five tables of the panel, annotated by the project's loader (cypmode/data/panel.py): one row per compound and enzyme, with the label (1 = inhibitor), the motifs found in the compound and the mechanism sentence.
# The loader prints a line per file it reads; the output is discarded here.
import contextlib
import io

import numpy as np
import pandas as pd

from cypmode.data.panel import load_panel

with contextlib.redirect_stdout(io.StringIO()):
    panel = load_panel()
panel["has_motif"] = panel["motifs"].apply(lambda m: bool(m) if m is not None else False)
enzyme_rows = []
for enzyme, group in panel.groupby("isoform"):
    inhibitors = group[group["inhibitor"] == 1]
    enzyme_rows.append({"enzyme": enzyme, "compounds": len(group), "inhibitors": len(inhibitors), "share that are inhibitors": len(inhibitors) / len(group),
                        "inhibitors with a motif": int(inhibitors["has_motif"].sum()), "share of inhibitors with a motif": float(inhibitors["has_motif"].mean())})
show_table(pd.DataFrame(enzyme_rows))
n_panel_rows, n_unparseable = len(panel), int((panel["mechanism"] == "unparseable SMILES").sum())
lowest_share = min(r["share of inhibitors with a motif"] for r in enzyme_rows)
highest_share = max(r["share of inhibitors with a motif"] for r in enzyme_rows)
'''

ODDS = '''
# The comparison for one enzyme (CYP3A4), step by step: the four counts, the odds ratio, its standard error and interval, and Fisher's exact test computed from the hypergeometric distribution and by the library.
from scipy.stats import fisher_exact, hypergeom, norm

def table_counts(group):
    a = int(((group["inhibitor"] == 1) & group["has_motif"]).sum())      # motif, inhibitor
    b = int(((group["inhibitor"] == 1) & ~group["has_motif"]).sum())     # no motif, inhibitor
    c = int(((group["inhibitor"] == 0) & group["has_motif"]).sum())      # motif, not an inhibitor
    d = int(((group["inhibitor"] == 0) & ~group["has_motif"]).sum())     # no motif, not an inhibitor
    return a, b, c, d

a, b, c, d = table_counts(panel[panel["isoform"] == "CYP3A4"])
odds_inhibitors, odds_others = a / b, c / d
odds_ratio = odds_inhibitors / odds_others
log_or_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
z_975 = norm.ppf(0.975)
ci_low, ci_high = np.exp(np.log(odds_ratio) - z_975 * log_or_se), np.exp(np.log(odds_ratio) + z_975 * log_or_se)

N_all, K_inhibitors, n_motif = a + b + c + d, a + b, a + c
possible = np.arange(max(0, n_motif - (N_all - K_inhibitors)), min(K_inhibitors, n_motif) + 1)
pmf = hypergeom.pmf(possible, N_all, K_inhibitors, n_motif)                 # the probability of every table with the same margins, if the flag and the label were independent
p_observed = hypergeom.pmf(a, N_all, K_inhibitors, n_motif)
p_by_hand = float(pmf[pmf <= p_observed * (1 + 1e-7)].sum())                 # the tables that are as unlikely as the observed one or less
p_library = fisher_exact([[a, b], [c, d]])[1]
expected_a = K_inhibitors * n_motif / N_all
print(f"CYP3A4 counts: a={a}, b={b}, c={c}, d={d}; compounds {N_all}, inhibitors {K_inhibitors}, with a motif {n_motif}")
print(f"odds of a motif among inhibitors {odds_inhibitors:.3f}, among non-inhibitors {odds_others:.3f}; odds ratio {odds_ratio:.3f}, standard error of its logarithm {log_or_se:.4f}, interval {ci_low:.2f} to {ci_high:.2f}")
print(f"under independence a would be about {expected_a:.0f}; Fisher's exact p by the hypergeometric sum {p_by_hand:.3e}, by the library {p_library:.3e}")

# the same for all five enzymes
odds_rows = []
for enzyme, group in panel.groupby("isoform"):
    a_, b_, c_, d_ = table_counts(group)
    value = (a_ * d_) / (b_ * c_)
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    odds_rows.append({"enzyme": enzyme, "motif share, inhibitors": a_ / (a_ + b_), "motif share, non-inhibitors": c_ / (c_ + d_), "odds ratio": value,
                      "interval low": float(np.exp(np.log(value) - z_975 * se)), "interval high": float(np.exp(np.log(value) + z_975 * se)), "Fisher p": float(fisher_exact([[a_, b_], [c_, d_]])[1])})
odds_table = pd.DataFrame(odds_rows)
show_table(odds_table.assign(**{"Fisher p": odds_table["Fisher p"].map(lambda p: f"{p:.1e}")}))

fig, axis = plt.subplots(figsize=(6.4, 3.4))
positions = np.arange(len(odds_table))[::-1]
axis.hlines(positions, odds_table["interval low"], odds_table["interval high"], color=theme["blue"], lw=2.4)
axis.plot(odds_table["odds ratio"], positions, "o", color=theme["orange"], ms=7)
axis.axvline(1.0, color=theme["muted"], ls=":", lw=1)
axis.set_yticks(positions, odds_table["enzyme"])
axis.set(xlabel="odds ratio of carrying a motif, inhibitors against non-inhibitors (95 % interval)", xlim=(0.9, 1.9))
tidy(axis, theme)
fig.tight_layout()
cyp3a4_row = odds_table[odds_table["enzyme"] == "CYP3A4"].iloc[0]
cyp2d6_row = odds_table[odds_table["enzyme"] == "CYP2D6"].iloc[0]
largest_row = odds_table.loc[odds_table["odds ratio"].idxmax()]
'''

OCTANT = '''
# The Octant screen (cypmode/data/octant_benchmark.py): load the release, drop the one compound that is also in the challenge's blinded test set, annotate the motifs, keep the curves that passed quality
# control, and compare the potency of the two groups with the project's own function; the result is checked against the file the module wrote when it was run (data/tdi_track/inputs/octant_cache/).
import json

from cypmode.challenge.official_extra import exclude_test_compounds
from cypmode.data.octant_benchmark import annotate_motifs, compare_potency_by_motif, load_inhibition_data

released = load_inhibition_data()
guarded, dropped_for_leak = exclude_test_compounds(released, "ocnt_batch")
octant = annotate_motifs(guarded)
module_result = compare_potency_by_motif(octant)
stored_result = json.loads((ROOT / "data/tdi_track/inputs/octant_cache/motif_potency_validation.json").read_text())
agree_with_stored = all(abs(module_result[k] - stored_result[k]) < 1e-9 for k in module_result)
scored = octant[(octant["drc_qc_status"] == "PASS")].dropna(subset=["CYP3A4_pIC50"])
with_motif = scored.loc[scored["has_motif"], "CYP3A4_pIC50"].to_numpy()
without_motif = scored.loc[~scored["has_motif"], "CYP3A4_pIC50"].to_numpy()
n1, n2 = len(with_motif), len(without_motif)
print(f"{len(released)} compounds in the release; excluded because they are in the blinded test set: {dropped_for_leak}; curves that passed quality control and have a pIC50: {len(scored)} ({n1} with a motif, {n2} without)")
print(f"median pIC50: {np.median(with_motif):.3f} with a motif, {np.median(without_motif):.3f} without; mean {with_motif.mean():.3f} and {without_motif.mean():.3f}")
print(f"the module's result equals the stored file: {agree_with_stored}")
n_released, n_failed_qc = len(released), int((guarded["drc_qc_status"] != "PASS").sum())
'''

RANKS = '''
# The rank test written out. First a tiny example (four values against five) so that the two ways of computing U can be compared; then the real groups: ranks of the pooled values (ties get the average rank),
# U from the rank sum, the mean and variance of U if the group label were irrelevant, the z value with the usual continuity correction, and the p-value; all checked against the library.
from scipy.stats import mannwhitneyu, rankdata

def u_by_pairs(x, y):
    greater = (x[:, None] > y[None, :]).sum()
    equal = (x[:, None] == y[None, :]).sum()
    return float(greater + 0.5 * equal)

def u_by_ranks(x, y):
    ranks = rankdata(np.concatenate([x, y]))
    return float(ranks[: len(x)].sum() - len(x) * (len(x) + 1) / 2)

tiny_x, tiny_y = np.array([6.1, 5.4, 7.0, 5.4]), np.array([5.0, 5.4, 4.8, 6.0, 5.2])
tiny_pairs, tiny_ranks = u_by_pairs(tiny_x, tiny_y), u_by_ranks(tiny_x, tiny_y)
print(f"tiny example: U by counting the pairs {tiny_pairs}, by ranks {tiny_ranks}; ranks of the nine pooled values {rankdata(np.concatenate([tiny_x, tiny_y]))}")

pooled = np.concatenate([with_motif, without_motif])
N_pooled = len(pooled)
U_pairs_check = u_by_pairs(with_motif[:300], without_motif[:300])                 # counting pairs on the whole data is a 450 x 633 comparison; a 300 x 300 piece shows the two methods agree
U_ranks_check = u_by_ranks(with_motif[:300], without_motif[:300])
U_statistic = u_by_ranks(with_motif, without_motif)
_, tie_counts = np.unique(pooled, return_counts=True)
mu_U = n1 * n2 / 2
var_U = n1 * n2 / 12 * ((N_pooled + 1) - (tie_counts**3 - tie_counts).sum() / (N_pooled * (N_pooled - 1)))
z_U = (U_statistic - mu_U - 0.5) / np.sqrt(var_U)                                 # the 0.5 is the continuity correction: U moves in steps of 0.5
p_one_sided = float(norm.sf(z_U))
library = mannwhitneyu(with_motif, without_motif, alternative="greater")
superiority = U_statistic / (n1 * n2)
print(f"U by ranks {U_statistic:.1f} (the library: {library.statistic:.1f}); mean of U under independence {mu_U:.1f}, standard deviation {np.sqrt(var_U):.1f}; z = {z_U:.3f}")
print(f"one-sided p by hand {p_one_sided:.5f}, by the library {library.pvalue:.5f}, in the stored file {stored_result['p_value_one_sided_greater']:.5f}; two-sided p {2 * p_one_sided:.5f}")
print(f"probability that a compound with a motif is more potent than one without (ties counted half): {superiority:.4f}")

# Does the formula for the spread of U hold? Shuffle the group labels many times and measure it.
rng = np.random.default_rng(0)
simulated = []
for _ in range(4000):
    shuffled = rng.permutation(pooled)
    simulated.append(rankdata(shuffled)[:n1].sum() - n1 * (n1 + 1) / 2)
simulated = np.array(simulated)
print(f"over 4000 shuffles of the labels: mean of U {simulated.mean():.1f} (formula {mu_U:.1f}), standard deviation {simulated.std():.1f} (formula {np.sqrt(var_U):.1f})")
n_shuffles_u = len(simulated)
'''

RESAMPLING = '''
# Two more ways to say the same thing, with no formula at all. A permutation test: shuffle the group labels, compute the difference of medians each time, and count how often a shuffle gives a difference as large
# as the observed one. A bootstrap interval: resample each group with replacement, many times, and read the middle 95 percent of the differences of medians.
observed_difference = float(np.median(with_motif) - np.median(without_motif))
rng = np.random.default_rng(1)
n_permutations, n_bootstrap = 20000, 10000
permuted = np.empty(n_permutations)
for i in range(n_permutations):
    shuffled = rng.permutation(pooled)
    permuted[i] = np.median(shuffled[:n1]) - np.median(shuffled[n1:])
p_permutation = float((permuted >= observed_difference).mean())
boot = np.empty(n_bootstrap)
for i in range(n_bootstrap):
    boot[i] = np.median(rng.choice(with_motif, n1)) - np.median(rng.choice(without_motif, n2))
boot_low, boot_high = np.percentile(boot, [2.5, 97.5])
fold_in_ic50 = 10 ** observed_difference
print(f"difference of medians {observed_difference:.3f} pIC50 units (a factor {fold_in_ic50:.2f} in IC50); permutation p {p_permutation:.4f}; bootstrap 95 % interval {boot_low:.3f} to {boot_high:.3f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.7))
for values, label, colour in ((with_motif, "with a motif", theme["orange"]), (without_motif, "without a motif", theme["blue"])):
    xs = np.sort(values)
    left.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=colour, lw=2.2, label=f"{label} (n = {len(xs)})")
left.set(xlabel="CYP3A4 pIC50", ylabel="share of compounds at or below")
left.legend(frameon=False, loc="lower right")
right.hist(permuted, bins=60, color=theme["muted"], alpha=0.85)
right.axvline(observed_difference, color=theme["orange"], lw=2.2)
right.set(xlabel="difference of medians after shuffling the labels", ylabel="shuffles", title="the observed difference (orange) among 20000 shuffles")
for axis in (left, right):
    tidy(axis, theme)
fig.tight_layout()
'''

CELLS = [
    md("""
## 5.3 The panel data, and how a yes-or-no flag is compared with a yes-or-no label

> **Field:** Cheminformatics · Statistics

*Where this section sits:* sections 5.1 and 5.2 produced two rules. The first data on which the project tried the motif screen was a public panel of inhibition labels, which is also the simplest data one can imagine for the purpose: a yes-or-no label per compound and enzyme. Section 5.4 will use a finer yardstick, the potency of the inhibition. This section uses the coarse one and, with it, introduces the
statistics that the rest of the notebook needs again and again: how to say, with a number and a statement of its uncertainty, that a yes-or-no flag goes with a yes-or-no label.

**The panel.** The Therapeutics Data Commons (TDC) distributes standard tables for drug-discovery research. Five of them are the CYP inhibition panel of Veith and co-workers (2009): between twelve and thirteen thousand compounds each, tested against one of five cytochrome P450 enzymes (CYP1A2, CYP2C9, CYP2C19, CYP2D6 and CYP3A4) in a **quantitative high-throughput screen**
(qHTS): every compound is tested at several concentrations, a dose-response curve (section 1.7) is fitted to the readings, and the curve is reduced to a label, inhibitor or non-inhibitor. The label is binary. It says whether a compound inhibits and nothing about how: a Type II binder, a Type I binder and a time-dependent inactivator are all labelled 1, which is the reason the project looked
at the panel with the motif screen in the first place (section 0.2). The module `cypmode/data/panel.py` loads the five tables from a local copy (`data/tdi_track/inputs/tdc_cache/`) and annotates every compound with the names of the motifs that the screen finds in it and with the mechanism sentence of section 5.1. The first cell does that and counts, for each enzyme, the compounds, the inhibitors and the inhibitors that carry a motif.
"""),
    code(PANEL),
    md("""
The five tables hold {{=n_panel_rows:,}} rows in all (a compound that was tested on several enzymes has a row in each table), and every structure could be read ({{=n_unparseable}} unreadable). Between {{=lowest_share:.0%}} and {{=highest_share:.0%}} of the inhibitors of an enzyme carry a motif. Taken alone this number says little: if a quarter of *all* compounds carry a motif, then a quarter of the inhibitors carrying one means that the motif goes with nothing.
The two shares must be compared, the share of motif carriers among inhibitors and among non-inhibitors, and the comparison needs a way of putting a number on the difference and saying how far it can be trusted.

**The two-by-two table.** A **contingency table** counts the compounds according to two yes-or-no properties at once. Here the properties are *the compound carries a motif* and *the compound is an inhibitor*, and the four counts are $a$ (motif and inhibitor), $b$ (inhibitor, no motif), $c$ (motif, not an inhibitor) and $d$ (neither). The **odds** of an event are the number of times it happens divided
by the number of times it does not: among the inhibitors the odds of carrying a motif are $a/b$, and among the non-inhibitors they are $c/d$. The **odds ratio** compares them,

$$\\mathrm{OR} \\;=\\; \\frac{a/b}{c/d} \\;=\\; \\frac{a\\,d}{b\\,c}.$$

It is 1 when the flag and the label are unrelated, above 1 when inhibitors carry the flag more often than non-inhibitors do, and below 1 when they carry it less often. It is preferred here to the plain difference of the two shares for a reason that matters for a panel: it is symmetric. The odds ratio of "motif, given inhibitor" equals that of "inhibitor, given motif" (swap the roles of $b$ and $c$ in the formula and the product $a\\,d$ over $b\\,c$ is unchanged), so it does not depend on
how many inhibitors the panel happens to contain, and the panel was assembled to contain many.

**How sure is the number?** The four counts come from a finite collection, and another collection would give slightly different ones. A count that is the result of many independent chance events has a spread equal to the square root of its size, so the relative uncertainty of a count $a$ is about $1/\\sqrt{a}$, and the uncertainty of its logarithm is about the same, $1/\\sqrt{a}$. The logarithm of the odds ratio is
$\\log a - \\log b - \\log c + \\log d$, a combination of four independent counts, and uncertainties of independent quantities add as squares, so the standard error of $\\log \\mathrm{OR}$ is (Woolf, 1955)

$$\\mathrm{SE}_{\\log} \\;=\\; \\sqrt{\\frac{1}{a}+\\frac{1}{b}+\\frac{1}{c}+\\frac{1}{d}}.$$

The distribution of $\\log \\mathrm{OR}$ over repeated collections is close to the normal distribution (the bell curve, with mean at the observed value and spread $\\mathrm{SE}_{\\log}$), and for it 95 percent of the probability lies within $1.96$ standard errors of the centre, the value $z_{0.975}$ at which the normal distribution leaves 2.5 percent in each tail. The 95 percent **confidence interval** (section 2.5) of the odds ratio is therefore $\\exp(\\log\\mathrm{OR} \\pm 1.96\\,\\mathrm{SE}_{\\log})$: a range that, in repeated collections of this kind, contains the true odds ratio 95 times in 100.

**Could it be chance?** A second kind of statement answers a different question. The **null hypothesis** is the assumption that the flag and the label are unrelated ($\\mathrm{OR}=1$ in the population). If it were true, how often would a collection show an association as strong as the observed one, or stronger? That probability is the **p-value**. Fisher (1935) showed how to compute it exactly. Fix the margins of the table: the total number of compounds $N$, the number of inhibitors $K$ and the number $n$ of compounds
that carry a motif. If the flag is unrelated to the label, the motif carriers are in effect a random draw of $n$ compounds from the $N$, of which $K$ are inhibitors, and the probability that exactly $a$ of the drawn compounds are inhibitors is the **hypergeometric distribution**,

$$P(a) \\;=\\; \\frac{\\binom{K}{a}\\binom{N-K}{n-a}}{\\binom{N}{n}},$$

where $\\binom{m}{k}$, the binomial coefficient, is the number of ways of choosing $k$ things out of $m$. The p-value of **Fisher's exact test** adds the probabilities $P(a')$ of all the tables that are as unlikely as the observed one, or less (a two-sided test: it counts surprises in both directions). A result with a p-value below 0.05 is called **statistically significant**, by a convention, and the word must be read with care: it says that chance alone is an unlikely explanation, and nothing about the *size* of the effect. With many thousands of compounds, a very small association is
significant; the odds ratio and its interval are what describe the size. The next cell does all of this for CYP3A4 step by step, checks the exact test against the library, and repeats the odds ratio for the five enzymes.
"""),
    code(ODDS),
    md("""
For CYP3A4 the four counts are {{=a:,}}, {{=b:,}}, {{=c:,}} and {{=d:,}}. Among the inhibitors the odds of a motif are {{=odds_inhibitors:.3f}} and among the non-inhibitors {{=odds_others:.3f}}, an odds ratio of {{=odds_ratio:.2f}} with a 95 percent interval from {{=ci_low:.2f}} to {{=ci_high:.2f}}. Under independence the table would show about {{=expected_a:.0f}}
compounds that are both motif carriers and inhibitors, and it shows {{=a:,}}. The exact p-value, computed by summing the hypergeometric probabilities and by the library, is {{=p_by_hand:.1e}} and {{=p_library:.1e}}: the two agree, as they must, since one is the other's definition. Over the five enzymes (the figure) the odds ratios run from {{=odds_table['odds ratio'].min():.2f}} to {{=odds_table['odds ratio'].max():.2f}}, the highest for {{=largest_row['enzyme']}}, and every
interval lies above 1.

**Reading the result.** A motif goes with inhibition, in all five enzymes, and the association is far beyond chance in four of them and modest in the fifth; its size is a different matter. For CYP3A4, {{=cyp3a4_row['motif share, inhibitors']:.1%}} of the inhibitors carry a motif against {{=cyp3a4_row['motif share, non-inhibitors']:.1%}} of the non-inhibitors, so a compound with a motif has about {{=odds_ratio:.1f}} times the odds of being an inhibitor, and
more than two thirds of the inhibitors have no motif at all. The screen is therefore not a predictor of inhibition, and was never meant to be: it flags a mechanism, and most inhibition goes by another. CYP2D6 has the weakest association ({{=cyp2d6_row['odds ratio']:.2f}}, interval {{=cyp2d6_row['interval low']:.2f}} to {{=cyp2d6_row['interval high']:.2f}}). A textbook reason, not tested here, is that the pocket of CYP2D6 is built to bind a positively charged amine rather than a nitrogen
that coordinates the iron, so the nitrogen that matters for that enzyme is of the kind that section 5.2 looked for and that this screen does not. Five tests were run here. Multiplying each p-value by the number of tests (the simplest correction for several tests, due to Bonferroni) leaves four verdicts untouched and takes CYP2D6 from {{=cyp2d6_row['Fisher p']:.3f}} to {{=min(1, 5 * cyp2d6_row['Fisher p']):.2f}}, so its association is the one to hold most loosely; the notebook returns to the matter in section 6.5, where a family of comparisons is the normal case.

*On what actually ran.* `load_panel` calls the TDC reader for each enzyme and reads the cached files; `annotate_mechanism` runs `detect_motifs` once per compound and records the mechanism sentence; `summarize_panel` produces the first table. The odds ratios and the exact tests of this section are computed in the cells of the notebook and are not part of the repository's modules. The panel's labels are the ones TDC distributes and come from a threshold applied to the screen's curves, which this project did not
re-derive.
"""),
    md("""
## 5.4 Testing the screen against real potency: the Octant screen and the rank test

> **Field:** Statistics · Pharmacology

*Where this section sits:* section 5.3 compared a flag with a label. A label throws away how strong the inhibition is; the potency of section 1.7 keeps it. This section asks whether the compounds that carry a motif are more potent, on a collection that the rule never saw. In doing so it builds the tool on which most comparisons of two groups of numbers in this
notebook rest, the **rank test** (a test that uses only the order of the values and not their size), whose meaning as an area under a curve was met in section 4.10.

**The data.** OpenADMET and Octant released the results of a CYP3A4 dose-response screen of a diversity library together with the post "Building the OpenADMET Data Engine" (CC BY 4.0; section 2.8 introduced the release). Each compound was incubated with the enzyme for 30 minutes before the substrate was added (the pre-incubation of section 1.8), so the potency that was measured is that of
reversible and time-dependent inhibition together. A diversity library is made of commercial screening compounds and not of marketed drugs; when the release was compared with the project's reference compounds by canonical SMILES, only one of them, vardenafil, appeared in it, and the motif screen had never been tuned on it. That makes it an *independent* test: a result on it cannot have been built into the rule.

**A guard against leakage.** One compound of the release, OCNT-0493952, is also in the blinded test set of the challenge. The release contains its measured potency, and a test compound whose measurement can be read from a public file is a leak in the making. The project's function `exclude_test_compounds` removes such compounds before anything is computed, so that the value is never looked at (the same rule is applied to every public source, section 6.8). The first cell
applies it, annotates the rest with the motif screen, keeps the dose-response curves that passed the quality control of the fit and compares the potency of the two groups with the project's own function, `compare_potency_by_motif`, and then checks that the result equals the file that the module wrote when it was run.
"""),
    code(OCTANT),
    md("""
The release holds {{=n_released:,}} compounds, {{=len(dropped_for_leak)}} of which ({{=', '.join(dropped_for_leak)}}) was excluded; {{=n_failed_qc}} further compounds failed the quality control of the curve, and {{=len(scored):,}} remain with a pIC50, {{=n1}} with a motif and {{=n2}} without. The motif carriers are more potent on average, {{=np.median(with_motif):.2f}} against {{=np.median(without_motif):.2f}} in median pIC50. The question is how
far a difference of that size, between groups of that size, can be trusted, and the answer has to avoid one assumption. The usual test for the difference of two averages, Student's t-test, assumes that the values of each group are spread like a bell curve, and potencies are not: they pile up against the floor of the assay and have long tails. A **rank test** assumes nothing of the kind, because it uses only the order of the values.

**Ranks and the statistic U.** Put the two groups together and sort all $N = n_1 + n_2$ values from the smallest to the largest. The smallest has **rank** 1, the next rank 2, and so on; values that are equal (a **tie**) all receive the average of the ranks they occupy. Let $x_i$, for $i = 1, \\dots, n_1$, be the potency of the $i$-th compound with a motif and $y_j$, for $j = 1, \\dots, n_2$, that of the $j$-th compound without one. The Mann-Whitney statistic $U$ counts, over every pair made of one $x$ and one $y$,
how often the $x$ is the larger, and counts a tie as one half:

$$U \\;=\\; \\sum_{i=1}^{n_1}\\sum_{j=1}^{n_2} \\Bigl( \\mathbb{1}[x_i > y_j] + \\tfrac12\\,\\mathbb{1}[x_i = y_j] \\Bigr).$$

Counting pairs is slow for large groups, and there is a shortcut through the ranks. If $R_1$ is the sum of the ranks of the $x$ values in the pooled sort, then the $x$ of rank $r$ is larger than the other $r-1$ values below it, some of which are $x$ values; the $x$ values among themselves account for $n_1(n_1+1)/2$ of these comparisons, and the remainder are comparisons against $y$ values, so

$$U \\;=\\; R_1 - \\frac{n_1\\,(n_1+1)}{2}.$$

The first part of the cell checks the two ways on a tiny example, and on a piece of the real data, so that nothing is taken on trust. $U$ divided by the number of pairs, $A = U/(n_1 n_2)$, is the probability that a compound drawn at random from the first group is more potent than one drawn at random from the second (ties counted half). That is the same quantity as the AUC of section 4.10, here with the potency as the score and the group as the class, and it is the natural *size* of the effect: 0.5 means no difference and 1 means that every compound of the first group beats every compound of the second.

**The null distribution of U.** If the group a compound belongs to were irrelevant to its potency, each pair would be equally likely to go either way, so the expected value of $U$ is half the number of pairs, $\\mu_U = n_1 n_2/2$. Its spread follows from the same argument, with a correction for ties (Mann and Whitney, 1947; the tie term is due to later authors):

$$\\sigma_U^2 \\;=\\; \\frac{n_1 n_2}{12}\\left( N + 1 - \\frac{\\sum_k (t_k^3 - t_k)}{N\\,(N-1)} \\right),$$

where $t_k$ is the number of values that share the $k$-th tied value. For groups of this size the distribution of $U$ is very close to the normal distribution, so the **z value** $z = (U - \\mu_U - \\tfrac12)/\\sigma_U$ (the half is a correction because $U$ moves in steps of one half) is looked up in the normal distribution and gives the one-sided p-value: the probability, if potency and group were unrelated, of a $U$ at least as large. The test is one-sided
because the claim of the screen is directional (a nitrogen that coordinates the iron tightens binding), and the direction was fixed before the data were read; a two-sided p-value, twice as large, is printed as well. The cell computes all of this by hand, checks it against the library and the stored file, and then checks the spread formula the direct way, by shuffling the group labels thousands of times and measuring the spread of $U$.
"""),
    code(RANKS),
    md("""
The two ways of computing $U$ agree, on the tiny example ({{=tiny_pairs}} and {{=tiny_ranks}}) and on the piece of real data ({{=U_pairs_check}} and {{=U_ranks_check}}). On the real groups $U$ is {{=U_statistic:,.1f}}, against {{=mu_U:,.1f}} expected if the group did not matter, and the standard deviation of $U$ is {{=np.sqrt(var_U):.1f}}, so the observed value is {{=z_U:.2f}} standard deviations above its expectation. The one-sided p-value is {{=p_one_sided:.4f}} by
hand, {{=library.pvalue:.4f}} by the library and {{=stored_result['p_value_one_sided_greater']:.4f}} in the file the module wrote: the three agree. The shuffles confirm the formulas for the centre and the spread of $U$ ({{=simulated.mean():.1f}} and {{=simulated.std():.1f}} over {{=n_shuffles_u:,}} shuffles, against {{=mu_U:.1f}} and {{=np.sqrt(var_U):.1f}}). The effect size is $A = {{=superiority:.3f}}$: a compound with a motif is more potent than one without in {{=superiority:.1%}} of the pairs, against 50 percent for no effect.

Two more checks say the same thing without any formula. A **permutation test** is the shuffling just used, turned into a test: shuffle the group labels many times, compute the statistic of interest each time (here the difference of the medians), and take as the p-value the share of shuffles that give a result as large as the observed one; it relies on nothing but the assumption that, if the group were irrelevant, any shuffle of the labels would have been as likely as the true one.
A **bootstrap** interval resamples each group with replacement (a bootstrap sample, section 4.4) many times, computes the difference of medians for each pair of resamples, and reads the middle 95 percent of those differences as the interval; it shows how much the difference would vary if the screen were run again on other compounds of the same library. The next cell does both and draws the two distributions of potency, as cumulative curves, next to the shuffled differences.
"""),
    code(RESAMPLING),
    md("""
The difference of medians is {{=observed_difference:.3f}} pIC50 units, which in units of the IC50 is a factor of {{=fold_in_ic50:.2f}} (section 1.7: one pIC50 unit is a factor of ten). The permutation test of that difference gives p = {{=p_permutation:.4f}}, and the bootstrap interval of the difference runs from {{=boot_low:.3f}} to {{=boot_high:.3f}}: it reaches just below zero. The figure shows what the numbers mean: the two cumulative curves are close and cross the same range, the curve of the
compounds with a motif lying a little to the right, and the observed difference sits in the tail of the shuffled differences, but not far out in it.

**Reading the result.** The motif carriers of this independent collection are more potent, and the evidence that this is not chance is real but not strong. The rank test, which looks at the whole distributions, gives p = {{=p_one_sided:.4f}}. The permutation test of the difference of medians, a cruder summary of the same data, gives {{=p_permutation:.3f}}, and the bootstrap interval for that difference includes zero by {{=abs(boot_low):.3f}}. The three results are not in conflict: a median
looks at one point of each distribution and throws away the rest, so it resolves less than the rank statistic does. What they say together is that the effect is at the edge of what a collection of this size can resolve. Its size is small on any reading: a factor of about {{=fold_in_ic50:.1f}} in IC50 for the medians, and a probability of {{=superiority:.0%}} that a motif carrier beats a non-carrier, where 50 percent would mean no effect. The two distributions overlap almost entirely.
This is independent evidence, on data the rule never saw, that the feature the screen looks for goes with a real pharmacological property. It is not evidence that the screen could rank compounds by potency, which it was never built to do, and a single test, on one enzyme, in one library, with a potency that mixes two mechanisms, cannot be stretched into more. Two cautions belong with it. The test is a single, directional one fixed beforehand,
so no correction for several tests applies; the same sentence would be false of a test chosen after looking at the data (section 6.7, on gates, is about exactly that). And the weaker cases of section 5.1 are in the data too: a compound flagged only through a nitrogen without a free lone pair counts here as a motif carrier, which, if anything, dilutes the contrast.

*On what actually ran.* `load_inhibition_data` reads the cached release (`inhibition.parquet`, downloaded once from the Hugging Face dataset), `annotate_motifs` applies the screen to the standardised SMILES, `compare_potency_by_motif` keeps the curves with `drc_qc_status == 'PASS'` and a fitted pIC50 and runs a one-sided Mann-Whitney test (`scipy.stats.mannwhitneyu`), and `main` writes `motif_potency_validation.json`; the call of `exclude_test_compounds` in `main` is the leak guard. The by-hand computation, the shuffles
and the bootstrap of this section are computed in the cells of the notebook and are not part of the module.
"""),
]
