# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Small helpers to draw block diagrams with matplotlib: rounded boxes with wrapped text, arrows between them, curved arrows."""

import textwrap

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, fill, edge, text_color, size=9, wrap=22, title=None):
    """A rounded box centred at (x, y): an optional bold title above the wrapped text."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.6, edgecolor=edge, facecolor=fill))
    body = "\n".join(textwrap.wrap(text, wrap))
    if title is None:
        ax.text(x, y, body, ha="center", va="center", fontsize=size, color=text_color)
        return
    ax.text(x, y + h * 0.3, title, ha="center", va="center", fontsize=size + 1.4, color=text_color, fontweight="bold")
    ax.text(x, y - h * 0.08, body, ha="center", va="center", fontsize=size, color=text_color)


def arrow(ax, start, end, color, style="-|>", rad=0.0, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, linewidth=lw, color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def canvas(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
