# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Shared style of the project's figures: palette, fonts, and the writer that saves a figure as light and dark SVG plus a 2x PNG.

The palette extends the one of the earlier figures (generate_tdi_figures.py): blue, red and green keep their roles, and a few more
colours are added for diagrams. Every figure is a function of the theme name, so the light and the dark version are drawn, not inverted.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {"background": "#FFFFFF", "text": "#1A1A1A", "muted": "#4D4D4D", "grid": "#D9D9D9", "blue": "#0F4D92", "red": "#B64342", "green": "#1B7F3A",
              "orange": "#C26A00", "purple": "#6F42C1", "teal": "#0E7C86", "gold": "#B08900", "panel": "#F3F5F8", "edge": "#000000"},
    "dark": {"background": "#0D1117", "text": "#E6EDF3", "muted": "#9198A1", "grid": "#30363D", "blue": "#58A6FF", "red": "#F85149", "green": "#3FB950",
             "orange": "#F0883E", "purple": "#BC8CFF", "teal": "#39C5CF", "gold": "#D29922", "panel": "#161B22", "edge": "#E6EDF3"},
}
RCPARAMS = {"font.family": "sans-serif", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "svg.fonttype": "none", "svg.hashsalt": "cypmode-figures"}  # a fixed salt gives the same ids to the clip paths and the glyphs in every build
ROOT = Path(__file__).resolve().parent


def apply(theme_name: str) -> dict:
    theme = THEMES[theme_name]
    plt.rcParams.update(RCPARAMS)
    plt.rcParams.update({"text.color": theme["text"], "axes.edgecolor": theme["muted"], "axes.labelcolor": theme["text"], "xtick.color": theme["text"],
                         "ytick.color": theme["text"], "figure.facecolor": theme["background"], "axes.facecolor": theme["background"],
                         "savefig.facecolor": theme["background"], "grid.color": theme["grid"], "axes.titlecolor": theme["text"]})
    return theme


def tidy(ax, theme: dict, grid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def save(make, group: str, figure_id: str) -> list[Path]:
    """``make(theme_name) -> Figure``. Writes ``<group>/<id>-light.svg``, ``-dark.svg`` and the 2x PNGs ``<id>-light.png`` and ``<id>-dark.png``; returns the files."""
    out_dir = ROOT / group
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("light", "dark"):
        fig = make(name)
        path = out_dir / f"{figure_id}-{name}.svg"
        fig.savefig(path, format="svg", metadata={"Date": None})  # no build date in the file, so that the same figure is the same bytes
        written.append(path)
        png = out_dir / f"{figure_id}-{name}.png"
        fig.savefig(png, format="png", dpi=200)
        written.append(png)
        plt.close(fig)
    return written
