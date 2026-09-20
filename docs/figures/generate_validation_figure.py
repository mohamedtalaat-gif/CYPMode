"""Generates docs/figures/fe_n_validation.svg from the checked-in validation summary.

Not part of the cypmode package or its runtime dependencies -- a one-off
script, run manually, in the publication-figure style described in
https://github.com/ChenLiu-1996/figures4papers (PUBLICATION_RCPARAMS/PALETTE
from its scientific-figure-making skill). Needs matplotlib, which the rest
of this project doesn't depend on: `pip install matplotlib`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "data" / "boltz_test" / "validation_summary.json"
OUT_PATH = Path(__file__).resolve().parent / "fe_n_validation.svg"

PALETTE = {
    "blue_main": "#0F4D92",
    "red_strong": "#B64342",
    "neutral": "#4D4D4D",
}

COORDINATION_CUTOFF_A = 2.6

LABELS = {
    "ketoconazole": "Ketoconazole",
    "ritonavir": "Ritonavir",
    "azamulin": "Azamulin",
    "vardenafil": "Vardenafil",
}


def main() -> None:
    results = json.loads(SUMMARY_PATH.read_text())
    names = list(LABELS)
    distances = [results[n]["fe_n_distance_angstrom"] for n in names]
    coordinated = [results[n]["coordinated"] for n in names]
    colors = [PALETTE["blue_main"] if c else PALETTE["red_strong"] for c in coordinated]

    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 16,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    x = range(len(names))
    bars = ax.bar(x, distances, color=colors, edgecolor="black", linewidth=1.5, width=0.6)

    ax.axhline(COORDINATION_CUTOFF_A, color=PALETTE["neutral"], linestyle="--", linewidth=2, zorder=0)
    ax.text(
        len(names) - 0.35,
        COORDINATION_CUTOFF_A + 0.35,
        f"dative coordination range (≤3.0 Å, cutoff {COORDINATION_CUTOFF_A} Å)",
        ha="right",
        va="bottom",
        fontsize=11,
        color=PALETTE["neutral"],
    )

    for bar, dist in zip(bars, distances, strict=True):
        # Bars near the cutoff line need extra clearance so the label doesn't
        # sit on top of the dashed line itself.
        pad = max(0.3, COORDINATION_CUTOFF_A - dist + 0.3) if dist < COORDINATION_CUTOFF_A else 0.3
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            f"{dist:.2f} Å",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[n] for n in names])
    ax.set_ylabel("Heme Fe – closest ligand N distance (Å)")
    ax.set_ylim(0, max(distances) + 2)
    ax.set_title(
        "Boltz-2 structural check: heme explicitly modeled,\ncovalently bonded to Cys442, ligand position unconstrained",
        fontsize=13,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["blue_main"], edgecolor="black", label="Coordinated"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["red_strong"], edgecolor="black", label="Not coordinated"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.text(
        0.5,
        -0.02,
        "All four calls independently confirmed by Walters, cyp-heatmap (ProLIF, github.com/PatWalters/cyp-heatmap)",
        ha="center",
        fontsize=10,
        style="italic",
        color=PALETTE["neutral"],
    )

    fig.tight_layout(pad=2)
    fig.savefig(OUT_PATH, format="svg", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
