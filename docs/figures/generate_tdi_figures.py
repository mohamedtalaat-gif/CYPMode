# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Generates docs/figures/tdi_track/*.svg from the checked-in TDI-track
result CSVs (data/tdi_track/outputs/).

The figures draw the first three stages of the model (the baseline, M3 without the amine flag, M3 with a tuned threshold) and the first reliability check, as they were
measured on the population as first built; the final results are in the README and in METHOD_REPORT.md.

Not part of the cypmode package or its runtime dependencies -- a one-off
script, run manually, matching docs/figures/generate_validation_figure.py's
publication style (light/dark SVG pairs, GitHub's own #gh-light-mode-only
/ #gh-dark-mode-only convention). Needs matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TDI_DATA = ROOT / "data" / "tdi_track" / "outputs"
COMPARISONS = TDI_DATA / "comparisons"
OUT_DIR = Path(__file__).resolve().parent / "tdi_track"

THEMES = {
    "light": {"background": "#FFFFFF", "text": "#1A1A1A", "muted": "#4D4D4D", "blue": "#0F4D92", "red": "#B64342", "green": "#1B7F3A", "edge": "#000000"},
    "dark": {"background": "#0D1117", "text": "#E6EDF3", "muted": "#9198A1", "blue": "#58A6FF", "red": "#F85149", "green": "#3FB950", "edge": "#E6EDF3"},
}
RCPARAMS = {"font.family": "sans-serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11}


def _style(theme: dict):
    plt.rcParams.update(RCPARAMS)
    plt.rcParams["text.color"] = theme["text"]
    plt.rcParams["axes.edgecolor"] = theme["muted"]
    plt.rcParams["axes.labelcolor"] = theme["text"]
    plt.rcParams["xtick.color"] = theme["text"]
    plt.rcParams["ytick.color"] = theme["text"]


def generate_mcc_progression(theme_name: str):
    theme = THEMES[theme_name]
    _style(theme)

    baseline = pd.read_csv(COMPARISONS / "baseline_fixed_seed.csv")
    # Stages 2 and 3 come from the runs of M3 without the amine flag (the feature set of the settled decision); the runs of 2026-09-25 that kept the flag by mistake are in comparisons/ and are not drawn.
    without_flag = {cyp: pd.read_csv(TDI_DATA / cyp.lower() / f"threshold_tuning_{cyp.lower()}_no_amine.csv") for cyp in ("CYP3A4", "CYP2D6")}

    isoforms = ["CYP3A4", "CYP2D6"]
    stages = ["Baseline\n(RDKit)", "M3, no amine flag\n(+motif+Morgan)", "M3 + tuned\nthreshold"]
    values = {
        cyp: [
            baseline[baseline["isoform"] == cyp]["mcc"].mean(),
            without_flag[cyp]["mcc_naive_0.5"].mean(),
            without_flag[cyp]["mcc_tuned"].mean(),
        ]
        for cyp in isoforms
    }

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=theme["background"])
    ax.set_facecolor(theme["background"])

    x = range(len(stages))
    width = 0.35
    colors = {"CYP3A4": theme["blue"], "CYP2D6": theme["red"]}
    for i, cyp in enumerate(isoforms):
        offsets = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offsets, values[cyp], width=width, label=cyp, color=colors[cyp], edgecolor=theme["edge"], linewidth=0.6)
        for bar, v in zip(bars, values[cyp]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=theme["text"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(stages)
    ax.set_ylabel("MCC (5-fold CV mean)")
    ax.set_title("TDI classification, first three stages: MCC by stage")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=theme["muted"], linewidth=0.6)

    fig.tight_layout()
    fig.text(0.5, -0.01, "Population as first built (every labelled row); superseded by the corrected population of the specification.", ha="center", va="top", fontsize=8, style="italic", color=theme["muted"])
    fig.savefig(OUT_DIR / f"tdi_mcc_progression-{theme_name}.svg", transparent=False, bbox_inches="tight")
    plt.close(fig)


def generate_reliability_correlation(theme_name: str):
    theme = THEMES[theme_name]
    _style(theme)

    df = pd.read_csv(COMPARISONS / "reliability_correlation.csv")

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=theme["background"])
    ax.set_facecolor(theme["background"])

    colors = [theme["blue"] if r >= 0 else theme["red"] for r in df["point_biserial_r"]]
    bars = ax.barh(df["component"].str.replace("_", " "), df["point_biserial_r"], color=colors, edgecolor=theme["edge"], linewidth=0.6)
    for bar, r in zip(bars, df["point_biserial_r"]):
        ax.text(r + (0.01 if r >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f"{r:.3f}", va="center",
                 ha="left" if r >= 0 else "right", fontsize=9, color=theme["text"])

    ax.axvline(0, color=theme["muted"], linewidth=0.6)
    ax.set_xlabel("Point-biserial correlation with real held-out prediction error")
    ax.set_title(f"Reliability score components vs. actual error\n(CYP3A4, descriptors, first fold, n={int(df['n'].iloc[0])})")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"tdi_reliability_correlation-{theme_name}.svg", transparent=False)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme_name in THEMES:
        generate_mcc_progression(theme_name)
        generate_reliability_correlation(theme_name)
        print(f"wrote {theme_name} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
