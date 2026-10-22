# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Generates docs/figures/structure_track/posebusters_summary-{light,dark}.svg
from the real, checked-in PoseBusters validity results.

Same one-off, manually-run pattern as generate_validation_figure.py and
generate_tdi_figures.py (not part of the cypmode package/runtime deps):
needs matplotlib, `pip install matplotlib`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POSEBUSTERS_PATH = ROOT / "data" / "structure_track" / "outputs" / "posebusters_validity.csv"
OUT_DIR = Path(__file__).resolve().parent / "structure_track"

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


def render(theme_name: str, colors: dict, n_pass: int, n_total: int, check_failures: list) -> Path:
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

    fig, (ax_summary, ax_checks) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1, 2]})
    fig.patch.set_facecolor(colors["background"])
    for ax in (ax_summary, ax_checks):
        ax.set_facecolor(colors["background"])

    # Left panel: overall pass/fail
    n_fail = n_total - n_pass
    bars = ax_summary.bar(
        ["Pass", "Fail"], [n_pass, n_fail], color=[colors["blue"], colors["red"]], edgecolor=colors["edge"], linewidth=1.5, width=0.55
    )
    for bar, count in zip(bars, [n_pass, n_fail], strict=True):
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", va="bottom", fontsize=15, fontweight="bold", color=colors["text"]
        )
    ax_summary.set_ylabel("Structures")
    ax_summary.set_ylim(0, n_total + 4)
    ax_summary.set_title(f"All checks\n{n_pass}/{n_total} pass ({n_pass / n_total * 100:.1f}%)", fontsize=13, color=colors["text"])

    # Right panel: which individual checks fail, and how often -- the real
    # point of this figure, per data/structure_track/outputs/README.md's
    # own stated plan.
    labels = [name for name, _ in check_failures]
    counts = [count for _, count in check_failures]
    y = range(len(labels))
    ax_checks.barh(y, counts, color=colors["red"], edgecolor=colors["edge"], linewidth=1.5, height=0.55)
    for yi, count in zip(y, counts, strict=True):
        ax_checks.text(count + 0.15, yi, str(count), va="center", fontsize=13, fontweight="bold", color=colors["text"])
    ax_checks.set_yticks(list(y))
    ax_checks.set_yticklabels(labels, fontsize=12)
    ax_checks.invert_yaxis()
    ax_checks.set_xlabel("Structures failing this check")
    ax_checks.set_xlim(0, max(counts) + 2 if counts else 1)
    ax_checks.set_title("Which real checks fail, and how often", fontsize=13, color=colors["text"])

    fig.suptitle("PoseBusters physical/chemical validity — every completed structure this project has", fontsize=14, color=colors["text"])
    fig.text(
        0.5,
        -0.02,
        "Real steric clashes cluster entirely on one check (minimum_distance_to_protein) — not scattered failures across many checks.",
        ha="center",
        fontsize=10,
        style="italic",
        color=colors["muted"],
    )

    out_path = OUT_DIR / f"posebusters_summary-{theme_name}.svg"
    fig.tight_layout(pad=2)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.3, facecolor=colors["background"])
    plt.close(fig)
    return out_path


def main() -> None:
    if not POSEBUSTERS_PATH.exists():
        raise SystemExit(
            f"{POSEBUSTERS_PATH} not found -- run `python -m cypmode.validation.pose_validity` first."
        )
    df = pd.read_csv(POSEBUSTERS_PATH)
    n_total = len(df)
    n_pass = int(df["all_passed"].sum())

    check_cols = [c for c in df.columns if c not in ("compound", "all_passed", "error")]
    failed = df[~df["all_passed"]]
    check_failures = []
    for c in check_cols:
        n_fail = int((failed[c] == False).sum())
        if n_fail > 0:
            check_failures.append((c, n_fail))
    n_parse_errors = int(failed["error"].notna().sum()) if "error" in failed.columns else 0
    if n_parse_errors:
        check_failures.append(("parse error (structure reconstruction)", n_parse_errors))
    check_failures.sort(key=lambda x: -x[1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme_name, colors in THEMES.items():
        out_path = render(theme_name, colors, n_pass, n_total, check_failures)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
