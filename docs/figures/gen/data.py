"""Figures about the training data: the label rule, the distribution of the shift, and what was measured on which isoform."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from style import apply, tidy

GROUP = "data"
ROOT = Path(__file__).resolve().parents[3]
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
FLOOR = 4.0  # pIC50 floor of the direct arm in the label rule
LOG2 = np.log10(2.0)
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def train() -> pd.DataFrame:
    return pd.read_csv(ROOT / TRAIN)


def labelled_arms(isoform: str) -> pd.DataFrame:
    """Rows of one isoform with both arms and a label: t, d, the shift s = t - max(d, 4) and the challenge label."""
    frame = train()
    t, d, label = (frame[f"{isoform}_pIC50_TDI_condition"], frame[f"{isoform}_pIC50_direct_inhibition"], frame[f"{isoform}_is_TDI"])
    keep = t.notna() & d.notna() & label.notna()
    out = pd.DataFrame({"t": t[keep], "d": d[keep], "label": label[keep].astype(bool)})
    out["floored"] = np.maximum(out["d"], FLOOR)
    out["shift"] = out["t"] - out["floored"]
    return out


@figure("data_label_rule", GROUP, "The label rule drawn on the real CYP3A4 data",
        "Each point is one compound: the pIC50 under the time-dependent condition (t, vertical) against the direct pIC50 floored at 4 (horizontal). "
        "The diagonal band is the rule t - max(d, 4) = log10 2: points above it are labelled time-dependent inhibitors (red), points below are not (blue). "
        "The rule reproduces the label of every plotted compound.",
        data=(TRAIN,), section="2.2")
def label_rule(theme_name):
    th = apply(theme_name)
    arms = labelled_arms("CYP3A4")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for flag, color, name in ((False, th["blue"], "not time-dependent"), (True, th["red"], "time-dependent")):
        part = arms[arms["label"] == flag]
        ax.scatter(part["floored"], part["t"], s=9, alpha=0.5, color=color, label=f"{name} (n = {len(part)})", linewidths=0)
    x = np.array([3.9, 8.0])
    ax.plot(x, x + LOG2, color=th["text"], lw=1.6, label=r"$t = \max(d,4) + \log_{10}2$")
    ax.set(xlabel="direct pIC50, floored at 4", ylabel="time-dependent pIC50 (t)", xlim=(3.9, 7.6), ylim=(3.9, 7.6))
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    tidy(ax, th)
    fig.tight_layout()
    return fig


@figure("data_shift_distribution", GROUP, "The shift between the two arms",
        "Histogram of s = t - max(d, 4) for every CYP3A4 and CYP2D6 compound with both arms. The dashed line is the threshold log10 2 = 0.301; "
        "the area to its right is the positive class. Most compounds sit near zero because most compounds are not time-dependent inhibitors.",
        data=(TRAIN,), section="2.3")
def shift_distribution(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=False)
    bins = np.linspace(-1.5, 3.0, 91)
    for ax, isoform in zip(axes, ("CYP3A4", "CYP2D6"), strict=True):
        arms = labelled_arms(isoform)
        ax.hist(arms.loc[~arms["label"], "shift"], bins=bins, color=th["blue"], alpha=0.85, label="negative")
        ax.hist(arms.loc[arms["label"], "shift"], bins=bins, color=th["red"], alpha=0.85, label="positive")
        ax.axvline(LOG2, color=th["text"], ls="--", lw=1.4)
        ax.set(xlabel=r"shift $s = t - \max(d, 4)$", ylabel="compounds", title=f"{isoform}: {arms['label'].mean():.1%} positive (n = {len(arms)})")
        tidy(ax, th)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def coverage_caption() -> str:
    frame = train()
    parts = []
    for iso in ("CYP3A4", "CYP2D6"):
        labelled = frame[f"{iso}_is_TDI"].notna()
        both = labelled & frame[f"{iso}_pIC50_TDI_condition"].notna() & frame[f"{iso}_pIC50_direct_inhibition"].notna()
        no_direct = labelled & frame[f"{iso}_pIC50_direct_inhibition"].isna()
        positives = int(frame.loc[no_direct, f"{iso}_is_TDI"].astype(bool).sum())
        parts.append(f"{iso}: {int(labelled.sum())} labelled, {int(both.sum())} with both arms, {int(no_direct.sum())} without a direct measurement ({positives} of them positive)")
    return ("Number of training compounds with each assay arm measured, per isoform, and the number carrying a challenge label (CYP3A4 and CYP2D6 only). "
            "The labelled compounds split into those with both arms measured and those with no direct-arm measurement. " + "; ".join(parts) + ". "
            "The label rule of the previous figures can therefore only be checked on the compounds with both arms.")


@figure("data_coverage", GROUP, "What was measured on which isoform", coverage_caption, data=(TRAIN,), section="2.1")
def coverage(theme_name):
    th = apply(theme_name)
    frame = train()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.24
    x = np.arange(len(ISOFORMS))
    for offset, name, suffix, color in ((-1, "time-dependent arm measured", "_pIC50_TDI_condition", th["red"]), (0, "direct arm measured", "_pIC50_direct_inhibition", th["blue"])):
        counts = [int(frame[f"{iso}{suffix}"].notna().sum()) for iso in ISOFORMS]
        ax.bar_label(ax.bar(x + offset * width, counts, width, color=color, label=name), fontsize=8, padding=2)
    labelled = [iso for iso in ISOFORMS if f"{iso}_is_TDI" in frame]
    both = {iso: int((frame[f"{iso}_is_TDI"].notna() & frame[f"{iso}_pIC50_TDI_condition"].notna() & frame[f"{iso}_pIC50_direct_inhibition"].notna()).sum()) for iso in labelled}
    total = {iso: int(frame[f"{iso}_is_TDI"].notna().sum()) for iso in labelled}
    positions = np.array([ISOFORMS.index(iso) + width for iso in labelled])
    ax.bar(positions, [both[i] for i in labelled], width, color=th["green"], label="labelled, both arms measured")
    ax.bar(positions, [total[i] - both[i] for i in labelled], width, bottom=[both[i] for i in labelled], color=th["muted"], label="labelled, no direct arm (all negative)")
    for pos, iso in zip(positions, labelled, strict=True):
        ax.text(pos, total[iso] + 30, str(total[iso]), ha="center", fontsize=8)
    ax.set_xticks(x, [i if i in labelled else f"{i}\n(no label)" for i in ISOFORMS])
    ax.set(ylabel="compounds")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax, th, grid=False)
    fig.tight_layout()
    return fig
