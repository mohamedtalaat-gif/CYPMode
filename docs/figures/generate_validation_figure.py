"""Generates docs/figures/fe_n_validation-{light,dark}.svg from the checked-in
validation summary.

Not part of the cypmode package or its runtime dependencies -- a one-off
script, run manually, in the publication-figure style described in
https://github.com/ChenLiu-1996/figures4papers (PUBLICATION_RCPARAMS/PALETTE
from its scientific-figure-making skill). Needs matplotlib, which the rest
of this project doesn't depend on: `pip install matplotlib`.

Two theme variants, not one figure with a CSS media query -- GitHub renders
Markdown images as plain <img> tags, so a single SVG can't switch itself.
README.md embeds both with the `#gh-light-mode-only` / `#gh-dark-mode-only`
URL-fragment convention GitHub's Markdown renderer recognizes, so the right
one shows automatically.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "data" / "boltz_test" / "validation_summary.json"
OUT_DIR = Path(__file__).resolve().parent

COORDINATION_CUTOFF_A = 2.6

LABELS = {
    "ketoconazole": "Ketoconazole",
    "ritonavir": "Ritonavir",
    "azamulin": "Azamulin",
    "vardenafil": "Vardenafil",
}

# Light matches a plain white page; dark matches GitHub's own dark-theme
# colors (#0d1117 background, #e6edf3 text, #58a6ff/#f85149 accent blue/red)
# so the chart reads as native to the surrounding page in either theme.
THEMES = {
    "light": {
        "background": "#FFFFFF",
        "text": "#1A1A1A",
        "muted": "#4D4D4D",
        "blue": "#0F4D92",
        "red": "#B64342",
        "edge": "#000000",
    },
    "dark": {
        "background": "#0D1117",
        "text": "#E6EDF3",
        "muted": "#8B949E",
        "blue": "#58A6FF",
        "red": "#F85149",
        "edge": "#E6EDF3",
    },
}


def render(theme_name: str, colors: dict, names: list, distances: list, coordinated: list) -> Path:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 16,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "text.color": colors["text"],
            "axes.labelcolor": colors["text"],
            "axes.edgecolor": colors["text"],
            "xtick.color": colors["text"],
            "ytick.color": colors["text"],
        }
    )

    bar_colors = [colors["blue"] if c else colors["red"] for c in coordinated]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor(colors["background"])

    x = range(len(names))
    bars = ax.bar(x, distances, color=bar_colors, edgecolor=colors["edge"], linewidth=1.5, width=0.6)

    ax.axhline(COORDINATION_CUTOFF_A, color=colors["muted"], linestyle="--", linewidth=2, zorder=0)
    ax.text(
        len(names) - 0.35,
        COORDINATION_CUTOFF_A + 0.35,
        f"dative coordination range (≤3.0 Å, cutoff {COORDINATION_CUTOFF_A} Å)",
        ha="right",
        va="bottom",
        fontsize=11,
        color=colors["muted"],
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
            color=colors["text"],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[n] for n in names])
    ax.set_ylabel("Heme Fe – closest ligand N distance (Å)")
    ax.set_ylim(0, max(distances) + 2)
    ax.set_title(
        "Boltz-2 structural check: heme explicitly modeled,\ncovalently bonded to Cys442, ligand position unconstrained",
        fontsize=13,
        color=colors["text"],
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["blue"], edgecolor=colors["edge"], label="Coordinated"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["red"], edgecolor=colors["edge"], label="Not coordinated"),
    ]
    legend = ax.legend(handles=legend_handles, loc="upper left")
    for text in legend.get_texts():
        text.set_color(colors["text"])

    fig.text(
        0.5,
        -0.02,
        "All four calls independently confirmed by Walters, cyp-heatmap (ProLIF, github.com/PatWalters/cyp-heatmap)",
        ha="center",
        fontsize=10,
        style="italic",
        color=colors["muted"],
    )

    out_path = OUT_DIR / f"fe_n_validation-{theme_name}.svg"
    fig.tight_layout(pad=2)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.3, facecolor=colors["background"])
    plt.close(fig)
    return out_path


def main() -> None:
    results = json.loads(SUMMARY_PATH.read_text())
    names = list(LABELS)
    distances = [results[n]["fe_n_distance_angstrom"] for n in names]
    coordinated = [results[n]["coordinated"] for n in names]

    for theme_name, colors in THEMES.items():
        out_path = render(theme_name, colors, names, distances, coordinated)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
