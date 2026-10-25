# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures drawn from the data files and the logs of the experiments: the single-concentration screen against the direct potency, the screen as the gate of the CYP2D6 hits, the compounds without a direct curve, the coverage of the
external panels (PXR and Tox21), the negative control with permuted labels, the pretrained models as one forest plot, the stage-1 screen of the fine-tuned networks against its rule, and the table of the pretrained models."""

import re
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from style import apply, tidy

GROUP = "data"
ML = "ml"  # the figures about models and gates are filed with the other figures of machine learning
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
EXTRA = "data/tdi_track/inputs/official_extra/cyp_challenge"
SCREEN = f"{EXTRA}/cyp-challenge-single-concentration-TRAIN.csv"
INHIBITION = f"{EXTRA}/cyp-challenge-TRAIN_inhibition.csv"
PXR = "data/tdi_track/inputs/official_extra/pxr_challenge/pxr-challenge_TRAIN.csv"
TOX21 = "data/tdi_track/inputs/tox21_cache/tox21_cyp_compact.csv.gz"
LOGS = "data/structure_track/outputs/cyp3a4/logs"
ISOFORMS = ("CYP3A4", "CYP2D6")
LOG2 = np.log10(2)


@cache
def screen_table() -> pd.DataFrame:
    """One row per compound and enzyme: the mean log2 fold change of the single-concentration screen (50 micromolar), the direct pIC50 when the compound went to a dose-response, and whether it reached the TDI assay."""
    screen = pd.read_csv(ROOT / SCREEN)
    direct = pd.read_csv(ROOT / INHIBITION).set_index("Molecule_Name")
    labelled = pd.read_csv(ROOT / TRAIN).set_index("Molecule_Name")
    rows = []
    for iso in ISOFORMS:
        reading = screen[screen["enzyme"] == iso].groupby("Molecule_Name")["log2fc_estimate"].mean()
        frame = pd.DataFrame({"screen": reading})
        frame["pic50"] = direct[f"{iso}_pIC50_direct_inhibition"].reindex(frame.index)
        frame["tdi_assayed"] = labelled[f"{iso}_is_TDI"].notna().reindex(frame.index)
        frame["enzyme"] = iso
        frame = frame.dropna(subset=["tdi_assayed"])
        frame["tdi_assayed"] = frame["tdi_assayed"].astype(bool)
        rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True)


# -- the single-concentration screen against the direct potency ----------------------------------------------------------------------------------

def screen_caption() -> str:
    table = screen_table()
    parts = []
    for iso in ISOFORMS:
        part = table[(table["enzyme"] == iso)].dropna(subset=["pic50"])
        parts.append(f"{iso}: {len(part)} compounds with both, Spearman {spearmanr(part['screen'], part['pic50'])[0]:+.2f}")
    return ("The organisers' single-concentration screen (every compound at 50 micromolar, the log2 fold change of the signal against the control: more negative is more inhibition) against the direct-inhibition pIC50 of the same compound from its dose-response curve, "
            "for the compounds that have both. Every point is a compound; the colour counts the compounds in each cell. The dashed line is the floor of the assay at pIC50 = 4, below which a potency is not measured reliably. "
            + "; ".join(parts) + ". The screen reads the same biology as the curve, so a compound that inhibits strongly in the screen has a high pIC50; but the relation is loose below the floor, and the screen is available only for the training compounds, "
            "so it could be used to train a block of the model and never as an input at prediction time (section 9.10): predicted from structure, it did not pass its gate.")


@figure("data_screen_vs_pic50", GROUP, "The single-concentration screen against the direct pIC50", screen_caption, data=(SCREEN, INHIBITION), section="2.8")
def screen_vs_pic50(theme_name):
    th = apply(theme_name)
    table = screen_table()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, iso in zip(axes, ISOFORMS, strict=True):
        part = table[table["enzyme"] == iso].dropna(subset=["pic50"])
        image = ax.hexbin(part["screen"], part["pic50"], gridsize=34, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(4, color=th["orange"], ls="--", lw=1.2)
        ax.set(xlabel="screen: log2 fold change at 50 uM", title=f"{iso}, Spearman {spearmanr(part['screen'], part['pic50'])[0]:+.2f}")
        fig.colorbar(image, ax=ax, label="compounds (log scale)")
        tidy(ax, th)
    axes[0].set_ylabel("direct-inhibition pIC50")
    fig.tight_layout()
    return fig


# -- the screen as the gate of the hits ---------------------------------------------------------------------------------------------------------------

def gate_study() -> dict:
    table = screen_table()
    out = {}
    for iso in ISOFORMS:
        part = table[table["enzyme"] == iso]
        out[iso] = {"table": part, "share": float(part["tdi_assayed"].mean()), "rho": float(spearmanr(part["screen"], part["tdi_assayed"].astype(int))[0]),
                    "max_assayed": float(part.loc[part["tdi_assayed"], "screen"].max()), "min_not": float(part.loc[~part["tdi_assayed"], "screen"].min())}
    return out


def gate_caption() -> str:
    study = gate_study()
    d, a = study["CYP2D6"], study["CYP3A4"]
    return (f"Which compounds reached the time-dependent-inhibition assay on an enzyme, against what the single-concentration screen showed on that enzyme, for the {len(d['table'])} compounds of the all-isoform panel. "
            f"Left (CYP2D6): the histogram of the screen reading of the compounds that were taken to the TDI assay (red) and of those that were not (grey). The two groups hardly overlap: every compound taken on has a reading of {d['max_assayed']:.2f} or less, "
            f"and the compounds above {d['min_not']:.2f} were never taken on ({d['share']:.1%} of the panel reached the assay, rank correlation {d['rho']:+.2f} with the reading). So for CYP2D6 the hit is almost a threshold on a number that the test compounds do not come with, "
            f"and the selection-aware model learns that threshold from structure (its gate). Right (CYP3A4): the same share of compounds taken on in each tenth of the reading ({a['share']:.1%} overall, rank correlation {a['rho']:+.2f}): for CYP3A4 the choice did not follow the CYP3A4 reading, "
            "because the hit expansion that built the test set prioritised the hits of CYP3A4, CYP1A2 and CYP2C9, which is the reason that the selection model matters for CYP2D6 and not for CYP3A4.")


@figure("ml_hit_gate_vs_screen", ML, "The CYP2D6 hit gate against the real screen", gate_caption, data=(SCREEN, TRAIN), section="9.1")
def hit_gate(theme_name):
    th = apply(theme_name)
    study = gate_study()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax = axes[0]
    part = study["CYP2D6"]["table"]
    bins = np.linspace(part["screen"].min(), part["screen"].max(), 50)
    ax.hist(part.loc[~part["tdi_assayed"], "screen"], bins=bins, color=th["muted"], alpha=0.8, label="not taken to the TDI assay")
    ax.hist(part.loc[part["tdi_assayed"], "screen"], bins=bins, color=th["red"], alpha=0.9, label="taken to the TDI assay")
    ax.set(xlabel="CYP2D6 screen: log2 fold change at 50 uM", ylabel="compounds", title="CYP2D6: the reading decides")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax, th)
    ax = axes[1]
    for iso, color in (("CYP2D6", th["red"]), ("CYP3A4", th["blue"])):
        part = study[iso]["table"]
        decile = pd.qcut(part["screen"], 10, labels=False)
        share = part.groupby(decile)["tdi_assayed"].mean()
        ax.plot(range(1, 11), share.to_numpy(), "o-", color=color, label=iso)
    ax.set(xlabel="tenth of the screen reading (1 = the strongest inhibition)", ylabel="share taken to the TDI assay", title="by tenth of the reading", ylim=(-0.02, 1.05))
    ax.legend(frameon=False)
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the compounds without a direct curve ---------------------------------------------------------------------------------------------------------------

@cache
def nodirect_study() -> dict:
    train = pd.read_csv(ROOT / TRAIN)
    rows = {}
    for iso in ISOFORMS:
        labelled = train[train[f"{iso}_is_TDI"].notna()]
        direct = labelled[f"{iso}_pIC50_direct_inhibition"].notna()
        rows[iso] = {"both": int(direct.sum()), "none": int((~direct).sum()), "rate_both": float(labelled.loc[direct, f"{iso}_is_TDI"].astype(float).mean()), "rate_none": float(labelled.loc[~direct, f"{iso}_is_TDI"].astype(float).mean()),
                     "tdi_both": labelled.loc[direct, f"{iso}_pIC50_TDI_condition"].dropna().to_numpy(), "tdi_none": labelled.loc[~direct, f"{iso}_pIC50_TDI_condition"].dropna().to_numpy()}
    return rows


def nodirect_caption() -> str:
    study = nodirect_study()
    a, d = study["CYP3A4"], study["CYP2D6"]
    return (f"The labelled compounds that have no direct-inhibition curve. For CYP3A4, {a['none']} of {a['both'] + a['none']} labelled compounds were measured only in the TDI arm, and all of them are labelled negative ({a['rate_none']:.0%} positive, against {a['rate_both']:.1%} of the {a['both']} compounds that have both arms), "
            f"although their TDI-arm potency is high (median pIC50 {np.median(a['tdi_none']):.1f}, against {np.median(a['tdi_both']):.1f} for the compounds with both arms): the label rule needs the direct value to compute a shift, and without it the convention assigns a negative. "
            f"The blinded test set was measured in both arms, so it has no such compounds; training on them and scoring on them mixes two populations, which is why they are excluded from the pool of CYP3A4 (section 2.4). For CYP2D6 only {d['none']} labelled compounds are in this group. "
            "Left: the counts and the share of positives in each group; right: the TDI-arm potency of the two groups of CYP3A4; the dashed line is 4.301, the potency above which the organisers infer a positive for a compound with a direct pIC50 of 4 or less.")


@figure("data_nodirect_coverage", GROUP, "The labelled compounds without a direct curve", nodirect_caption, data=(TRAIN,), section="2.4")
def nodirect_coverage(theme_name):
    th = apply(theme_name)
    study = nodirect_study()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    labels = [f"{iso}\nboth arms" for iso in ISOFORMS] + [f"{iso}\nno direct curve" for iso in ISOFORMS]
    counts = [study[iso]["both"] for iso in ISOFORMS] + [study[iso]["none"] for iso in ISOFORMS]
    rates = [study[iso]["rate_both"] for iso in ISOFORMS] + [study[iso]["rate_none"] for iso in ISOFORMS]
    positions = [0, 3, 1, 4]
    ax.bar(positions, counts, color=[th["blue"], th["blue"], th["orange"], th["orange"]])
    for x, count, rate in zip(positions, counts, rates, strict=True):
        ax.text(x, count, f"{count}\n{rate:.0%} positive", ha="center", va="bottom", fontsize=8, color=th["text"])
    ax.set_xticks(positions, labels, fontsize=7.5)
    ax.set(ylabel="labelled compounds", title="how many, and how many are positive", ylim=(0, max(counts) * 1.25))
    tidy(ax, th, grid=False)
    ax = axes[1]
    a = study["CYP3A4"]
    bins = np.linspace(2, 9, 40)
    ax.hist(a["tdi_both"], bins=bins, color=th["blue"], alpha=0.75, label=f"both arms (n = {a['both']})")
    ax.hist(a["tdi_none"], bins=bins, color=th["orange"], alpha=0.75, label=f"no direct curve (n = {a['none']})")
    ax.axvline(4 + LOG2, color=th["text"], ls="--", lw=1.2)
    ax.set(xlabel="CYP3A4 pIC50 in the TDI arm", ylabel="compounds", title="the same potency, a different label")
    ax.legend(frameon=False, fontsize=8)
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the coverage of the external panels --------------------------------------------------------------------------------------------------------------

def fingerprints(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        out.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol is not None else None)
    return out


@cache
def coverage_study() -> dict:
    """Nearest-neighbour Tanimoto similarity of the test compounds and of 1,500 training compounds to the PXR training panel and to the Tox21 CYP screen."""
    from rdkit import DataStructs

    train = pd.read_csv(ROOT / TRAIN)["SMILES"].sample(1500, random_state=0).tolist()
    test = pd.read_csv(ROOT / TEST)["SMILES"].tolist()
    panels = {"PXR challenge (training file)": pd.read_csv(ROOT / PXR)["SMILES"].tolist(), "Tox21 CYP screen": pd.read_csv(ROOT / TOX21)["smiles"].tolist()}
    result = {"panels": {}}
    queries = {"test": [f for f in fingerprints(test) if f is not None], "training (sample of 1,500)": [f for f in fingerprints(train) if f is not None]}
    for name, smiles in panels.items():
        reference = [f for f in fingerprints(smiles) if f is not None]
        result["panels"][name] = {"size": len(reference), "similarity": {group: np.array([max(DataStructs.BulkTanimotoSimilarity(q, reference)) for q in fps]) for group, fps in queries.items()}}
    return result


def coverage_caption() -> str:
    study = coverage_study()
    parts = []
    for name, panel in study["panels"].items():
        test = panel["similarity"]["test"]
        training = panel["similarity"]["training (sample of 1,500)"]
        parts.append(f"{name} ({panel['size']} compounds): median nearest-neighbour similarity of a test compound {np.median(test):.2f}, {np.mean(test >= 0.5):.1%} of the test compounds with a neighbour at 0.5 or more, {int((test >= 0.999).sum())} identical; "
                     f"{np.mean(training >= 0.999):.1%} of the training sample are identical to a compound of the panel")
    return ("How much of the chemistry of the challenge the two external panels cover: for each test compound (and for a random 1,500 of the training compounds), the Tanimoto similarity of its Morgan fingerprint to the most similar compound of the panel. "
            + "; ".join(parts) + ". The spike at 1 is the compounds that the training set and the panel share (the same library was tested in both); the test set shares almost none. A block that is trained on a panel can only help where the panel has neighbours, and both panels cover the test chemistry thinly, which is consistent with the verdicts of these blocks (NOT MET alone and inside the stack, section 9.10). "
            "Only identifiers and structures of the panels are read here, never a measurement of a test compound.")


@figure("data_external_coverage", GROUP, "How far the PXR and Tox21 panels cover the chemistry of the challenge", coverage_caption, data=(PXR, TOX21, TRAIN, TEST), section="2.8")
def external_coverage(theme_name):
    th = apply(theme_name)
    study = coverage_study()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (name, panel) in zip(axes, study["panels"].items(), strict=True):
        for group, color in (("test", th["red"]), ("training (sample of 1,500)", th["blue"])):
            ax.hist(panel["similarity"][group], bins=bins, histtype="stepfilled", alpha=0.45, color=color, label=group, density=True)
        ax.axvline(0.5, color=th["text"], ls="--", lw=1)
        ax.set(xlabel="similarity to the nearest compound of the panel", title=f"{name}: {panel['size']} compounds")
        tidy(ax, th)
    axes[0].set_ylabel("density of compounds")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# -- the negative control -----------------------------------------------------------------------------------------------------------------------------

CONTROL = re.compile(r"z0=([\d.]+)\s+real labels: AUC ([\d.]+), MCC ([+-][\d.]+) \| permuted: AUC ([\d.]+) \(range ([\d.]+) to ([\d.]+)\), MCC ([+-][\d.]+) \(range ([+-][\d.]+) to ([+-][\d.]+)\)")


@cache
def control_rows() -> pd.DataFrame:
    rows = []
    for iso in ISOFORMS:
        text = (ROOT / LOGS / f"permutation_control_{iso}.log").read_text(errors="ignore")
        n = re.search(r"(\d+) permutations", text)
        for z0, real, _mcc, mean, low, high, *_ in CONTROL.findall(text):
            rows.append({"isoform": iso, "z0": float(z0), "real": float(real), "permuted": float(mean), "low": float(low), "high": float(high), "permutations": int(n.group(1)) if n else 0})
    return pd.DataFrame(rows)


def control_caption() -> str:
    table = control_rows()
    worst = table.loc[(table["permuted"] - 0.5).abs().idxmax()]
    return (f"The negative control: the labels of the pool are permuted at random and the whole nested procedure, with every selection step, is run again; {int(table['permutations'].iloc[0])} permutations per enzyme. A pipeline that leaks information from the labels into its selection would still score above chance on shuffled labels. "
            f"Here the AUC on permuted labels is {table['permuted'].min():.3f} to {table['permuted'].max():.3f} (the bars show the range over the permutations) against {table['real'].min():.3f} to {table['real'].max():.3f} on the real labels, "
            f"and the largest distance from 0.5 is {abs(worst['permuted'] - 0.5):.3f} ({worst['isoform']}, z0 = {worst['z0']}). The control passes: nothing in the procedure scores above chance without a signal in the labels.")


@figure("ml_permuted_control", ML, "The negative control with permuted labels", control_caption, data=(f"{LOGS}/permutation_control_<isoform>.log",), section="9.10")
def permuted_control(theme_name):
    th = apply(theme_name)
    table = control_rows()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = []
    for position, row in enumerate(table.itertuples()):
        ax.vlines(position - 0.12, row.low, row.high, color=th["muted"], lw=6, alpha=0.6)
        ax.scatter(position - 0.12, row.permuted, s=50, color=th["muted"], zorder=3, label="permuted labels" if position == 0 else None)
        ax.scatter(position + 0.12, row.real, s=60, color=th["blue"], zorder=3, label="real labels" if position == 0 else None)
        labels.append(f"{row.isoform}\nz0 = {row.z0}")
    ax.axhline(0.5, color=th["red"], ls="--", lw=1.2, label="chance")
    ax.set_xticks(range(len(table)), labels)
    ax.set(ylabel="outer AUC of the nested procedure", ylim=(0.4, 0.9), title="the procedure scores chance on shuffled labels")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- pretrained models: one forest plot, the staged screen, the table -------------------------------------------------------------------------------

PRETRAINED = ("aux_tabicl", "aux_tabpfn", "aux_tabicl_monroe", "aux_tabicl_chemeleon", "aux_tabpfn_chemeleon", "aux_atomjepa", "aux_atomjepa_head", "monroe_embeddings", "embeddings", "aux_chemprop", "aux_chemprop_foundation", "aux_mlp", "aux_neural")


@cache
def pretrained_rows() -> pd.DataFrame:
    from cypmode.audit.docs_state import TITLES
    from cypmode.audit.ledger_kinds import stem_of
    from cypmode.explorer.data import paired_rows

    paired = paired_rows(ROOT / LOGS)
    paired = paired[paired["comparison"].str.startswith("stacked base+added")].copy()
    paired["stem"] = paired["log"].map(stem_of)
    paired = paired[paired["stem"].isin(PRETRAINED) & ~paired["log"].str.contains("_f10")]
    paired["idea"] = paired["stem"].map(lambda s: TITLES.get(s, s))
    return paired


def forest_caption() -> str:
    table = pretrained_rows()
    both, one = [], []
    for (idea, iso), part in table.groupby(["idea", "enzyme"]):
        passes = int(((part["p"] < 0.05) & (part["auc_gain"] > 0)).sum())
        if passes == len(part) and passes == 2:
            both.append(f"{idea} ({iso})")
        elif passes:
            one.append(f"{idea} ({iso})")
    return (f"The pretrained and learned-representation models that were added to the stack, as one forest plot: for each, the change in the outer AUC of the stack when it is added (positive is better), at the two confidence cuts of the labels, on each enzyme, with the corrected p-value beside the point; "
            f"a filled green point has p < 0.05 and a gain, an open grey point does not. The gate asks for both cuts, alone and inside the stack. Passing at both cuts: {', '.join(both) or 'none'}. Passing at one cut only, which the gate does not accept: {', '.join(one) or 'none'}. "
            "The tabular foundation models (TabICL, TabPFN), the frozen embeddings (CheMeleon, Monroe, Atom-JEPA) and the readouts on them gave gains that the design cannot tell from zero; the smallest gain that the design certifies is about 0.01 (section 6.7).")


@figure("ml_pretrained_forest", ML, "The pretrained models as one forest plot", forest_caption, data=(f"{LOGS}/aux_*.log",), section="9.4")
def pretrained_forest(theme_name):
    th = apply(theme_name)
    table = pretrained_rows()
    ideas = list(dict.fromkeys(table["idea"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.55 * len(ideas) + 2.0), sharey=True)
    for ax, iso in zip(axes, ISOFORMS, strict=True):
        for i, idea in enumerate(ideas):
            for z0, dy, marker in ((1.0, -0.16, "o"), (1.96, 0.16, "s")):
                row = table[(table["idea"] == idea) & (table["enzyme"] == iso) & (table["z0"] == z0)]
                if row.empty:
                    continue
                value, p = float(row["auc_gain"].iloc[0]), float(row["p"].iloc[0])
                passed = p < 0.05 and value > 0
                color = th["green"] if passed else th["muted"]
                ax.plot(value, i + dy, marker, color=color, mfc=color if passed else "none", ms=6)
                ax.text(value + 0.0008, i + dy, f"{p:.2f}", va="center", fontsize=6.5, color=th["muted"])
        ax.axvline(0, color=th["text"], lw=1)
        ax.set_yticks(range(len(ideas)), ideas, fontsize=8)
        ax.set_ylim(len(ideas) - 0.4, -0.6)
        ax.set(xlabel="change in outer AUC when added to the default stack", title=iso)
        tidy(ax, th)
    axes[0].plot([], [], "o", color=th["muted"], mfc="none", label="z0 = 1.0")
    axes[0].plot([], [], "s", color=th["muted"], mfc="none", label="z0 = 1.96")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    return fig


STAGE = re.compile(r"z0=1\.0\s+stacked base\+added vs base: AUC ([+-][\d.]+) \(p=([\d.]+), better (\d+)/(\d+)\)")


@cache
def stage_rows() -> list[dict]:
    from cypmode.challenge.chemeleon_stage import MIN_FOLDS, MIN_GAIN

    rows = []
    for stem, label in (("chemeleon", "CheMeleon, fine-tuned"), ("chemprop_foundation", "Chemprop from the two OpenADMET pretraining weights")):
        path = ROOT / LOGS / f"stage1_{stem}_CYP3A4.txt"
        found = STAGE.search(path.read_text(errors="ignore")) if path.exists() else None
        if found:
            gain, p, better, folds = float(found.group(1)), float(found.group(2)), int(found.group(3)), int(found.group(4))
            rows.append({"label": label, "gain": gain, "p": p, "better": better, "folds": folds, "continue": gain >= MIN_GAIN and better >= MIN_FOLDS, "min_gain": MIN_GAIN, "min_folds": MIN_FOLDS})
    return rows


def stage_caption() -> str:
    rows = stage_rows()
    parts = [f"{r['label']}: a gain of {r['gain']:+.4f} AUC, better in {r['better']} of {r['folds']} folds, so {'CONTINUE to the full design' if r['continue'] else 'STOP'}" for r in rows]
    return (f"The first stage of the staged design of the fine-tuned networks (CYP3A4, repeat 0, 5 outer folds, confidence cut z0 = 1.0). The rule was written before any result, for a reason of cost (about 38 minutes per task): the second stage runs only if the mean gain in AUC reaches {rows[0]['min_gain']} "
            f"and the stack is better in at least {rows[0]['min_folds']} of the 5 folds. " + "; ".join(parts) + ". A screen is not a gate and is not in the ledger: the network that stopped has no verdict, and the one that continued went on to the full 15-fold gate, which it met. "
            "The smallest gain that a 5-fold screen can certify is about one and a half times that of the 15-fold design, so a stop at this stage is an economy and not a negative result.")


@figure("ml_chemeleon_stage1", ML, "The stage-1 screen of the fine-tuned networks against its rule", stage_caption, data=(f"{LOGS}/stage1_chemeleon_CYP3A4.txt", f"{LOGS}/stage1_chemprop_foundation_CYP3A4.txt"), section="9.4")
def chemeleon_stage1(theme_name):
    th = apply(theme_name)
    rows = stage_rows()
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for position, row in enumerate(rows):
        ax.barh(position, row["gain"], color=th["green"] if row["continue"] else th["muted"], height=0.5)
        ax.text(max(row["gain"], row["min_gain"]) + 0.0006, position, f"{row['gain']:+.4f}, better in {row['better']} of {row['folds']} folds: {'CONTINUE' if row['continue'] else 'STOP'}", va="center", fontsize=8.5, color=th["text"])
    ax.axvline(rows[0]["min_gain"], color=th["orange"], ls="--", lw=1.4, label=f"the bar of the rule: +{rows[0]['min_gain']}")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=8.5)
    ax.set(xlabel="change in outer AUC when added to the stack (stage 1)", xlim=(0, 0.034))
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    tidy(ax, th)
    fig.tight_layout()
    return fig


@cache
def foundry_rows() -> list[dict]:
    from docs.video import facts

    return facts._foundry()


VERDICT_COLORS = {"MET": "green", "NOT MET": "muted", "screened": "orange", "not run": "grid", "pending": "gold", "RUNNING": "gold"}


def foundry_caption() -> str:
    rows = foundry_rows()
    counts = {}
    for row in rows:
        for cell in row["enzymes"].values():
            counts[cell["verdict"]] = counts.get(cell["verdict"], 0) + 1
    return (f"The pretrained models of the project and what each one brought, the table of the film's Foundry scene as a figure: one row per model, one column per enzyme, each cell the change in the AUC of the stack at z0 = 1.0 with the corrected p-value and the verdict of the registered gate. "
            f"Of the {sum(counts.values())} cells, {counts.get('MET', 0)} are MET, {counts.get('NOT MET', 0)} NOT MET, {counts.get('screened', 0)} only screened in the first stage of a staged design and {counts.get('not run', 0)} not run (the fine-tuned networks were not run on CYP2D6, "
            "where plain Chemprop is at chance, and a model dropped for its cost has no verdict). The rows are read from the same logs as the ledger.")


@figure("ml_foundry_table", ML, "The pretrained models and what each brought", foundry_caption, data=(f"{LOGS}/aux_*.log", f"{LOGS}/stage1_*.txt"), section="9.4")
def foundry_table(theme_name):
    th = apply(theme_name)
    rows = foundry_rows()
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.6))
    ax.axis("off")
    ax.set_xlim(0, 3)
    ax.set_ylim(len(rows), -1)
    for column, iso in enumerate(ISOFORMS):
        ax.text(1.35 + column * 0.85, -0.65, iso, ha="center", va="center", fontsize=10, fontweight="bold", color=th["text"])
    for i, row in enumerate(rows):
        ax.text(0.0, i, row["label"], ha="left", va="center", fontsize=8.5, color=th["text"])
        for column, iso in enumerate(ISOFORMS):
            cell = row["enzymes"][iso]
            color = th[VERDICT_COLORS.get(cell["verdict"], "muted")]
            ax.add_patch(plt.Rectangle((0.97 + column * 0.85, i - 0.4), 0.78, 0.8, color=color, alpha=0.35, linewidth=0))
            text = cell["verdict"] if "delta_auc" not in cell else f"{cell['delta_auc']:+.4f} (p {cell['p']:.2f})\n{cell['verdict']}"
            ax.text(1.36 + column * 0.85, i, text, ha="center", va="center", fontsize=7.5, color=th["text"])
    ax.legend(handles=[Patch(color=th[c], alpha=0.5, label=v) for v, c in VERDICT_COLORS.items() if v not in ("RUNNING",)], frameon=False, fontsize=7.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig
