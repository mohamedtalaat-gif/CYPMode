# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures of the ligand dynamics in the frozen pocket (``cypmode/validation/ligand_dynamics.py``): the gate per ligand for the static and the dynamic scores, the drift of the poses against their error,
and the traces of the pose of one ligand over the 20 ps. All of them are drawn from the table of the experiment and from its trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from scipy.stats import spearmanr
from style import apply, tidy

GROUP = "structure"
ROOT = Path(__file__).resolve().parents[3]
OUT = "data/structure_track/outputs/cyp3a4"
TABLE = f"{OUT}/ligand_dynamics_poses.csv"
GATED = "novel"
NAMES = {"static_pharmacophore_model": "static model (one frame)", "dynamic_pharmacophore": "dynamic pharmacophore", "stability": "stability", "dynamic_combined": "dynamic combined"}


def poses(regime: str = GATED) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / TABLE, float_precision="round_trip")
    return frame[frame["regime"] == regime].reset_index(drop=True)


def per_ligand_rho(frame: pd.DataFrame) -> pd.DataFrame:
    """Per ligand, the Spearman rho of each score with minus the true RMSD (the quantity of the gate, computed by the function that the gate uses)."""
    from cypmode.validation.ligand_dynamics import pool_rho

    rows = []
    for compound, pool in frame.groupby("compound", sort=True):
        row = {"compound": compound.removeprefix("bench-"), "poses": len(pool)}
        for name in NAMES:
            row[name] = pool_rho(pool[name], pool["true_rmsd"])
        rows.append(row)
    return pd.DataFrame(rows)


# -- the gate per ligand -------------------------------------------------------------------------------------------------------------------------------------

def gate_caption() -> str:
    rho = per_ligand_rho(poses())
    means = {NAMES[n]: rho[n].mean() for n in NAMES}
    return ("Each row is a benchmark ligand with at least 3 predicted poses, each dot the Spearman correlation, within the pool of the ligand, between a score and minus the distance of the pose to the crystal pose "
            f"(right of zero: the score ranks the better poses first). The line at 0.3 is the bar of the gate. Means over {len(rho)} ligands: " + ", ".join(f"{k} {v:+.2f}" for k, v in means.items()) + ".")


@figure("struct_dyn_gate", GROUP, "The static and the dynamic scores of a pose, ligand by ligand", gate_caption, data=(TABLE,), section="11.13")
def gate_figure(theme_name):
    th = apply(theme_name)
    rho = per_ligand_rho(poses())
    colours = {"static_pharmacophore_model": th["muted"], "dynamic_pharmacophore": th["blue"], "stability": th["orange"], "dynamic_combined": th["purple"]}
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rho) + 2.4))
    for k, name in enumerate(NAMES):
        ax.scatter(rho[name], np.arange(len(rho)) + (k - 1.5) * 0.14, s=34, color=colours[name], label=NAMES[name], zorder=3)
    ax.axvline(0.3, color=th["red"], lw=1.2, ls="--")
    ax.axvline(0, color=th["muted"], lw=0.8)
    ax.set_yticks(range(len(rho)), [f"{c} ({n})" for c, n in zip(rho["compound"], rho["poses"], strict=True)])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho of the score with minus the true RMSD")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncols=2)
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- drift against error -----------------------------------------------------------------------------------------------------------------------------------

def drift_caption() -> str:
    frame = poses()
    centred_drift = frame["drift"] - frame.groupby("compound")["drift"].transform("mean")
    centred_error = frame["true_rmsd"] - frame.groupby("compound")["true_rmsd"].transform("mean")
    rho = float(spearmanr(centred_drift, centred_error).statistic)
    return (f"Each dot is one of the {len(frame)} predicted poses of the benchmark ligands. Horizontal: the mean heavy-atom RMSD of the ligand to its start pose over the 18 scored frames of the 20 ps "
            f"(the drift). Vertical: the RMSD of the start pose to the crystal pose. Both are centred on the mean of their ligand, so that only the differences between the poses of one ligand are shown "
            f"(Spearman rho {rho:+.2f}). A stable pose is not a pose that is close to the crystal.")


@figure("struct_dyn_drift", GROUP, "The drift of a pose under dynamics against its error", drift_caption, data=(TABLE,), section="11.13")
def drift_figure(theme_name):
    th = apply(theme_name)
    frame = poses()
    x = frame["drift"] - frame.groupby("compound")["drift"].transform("mean")
    y = frame["true_rmsd"] - frame.groupby("compound")["true_rmsd"].transform("mean")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(x, y, s=22, color=th["blue"], alpha=0.8)
    ax.axhline(0, color=th["muted"], lw=0.8)
    ax.axvline(0, color=th["muted"], lw=0.8)
    ax.set_xlabel("drift in 20 ps, minus the mean of the ligand (A)")
    ax.set_ylabel("RMSD to the crystal pose, minus the mean of the ligand (A)")
    tidy(ax, th)
    fig.tight_layout()
    return fig


# -- the traces of one ligand --------------------------------------------------------------------------------------------------------------------------------

def example_ligand(frame: pd.DataFrame) -> str:
    """The ligand with the largest spread of the true RMSD among its poses (the pool in which a ranking matters most)."""
    return str(frame.groupby("compound")["true_rmsd"].agg(lambda v: v.max() - v.min()).idxmax())


def trace_caption() -> str:
    ligand = example_ligand(poses())
    return (f"The 20 ps of the poses of {ligand.removeprefix('bench-')}, the benchmark ligand whose poses differ most in their distance to the crystal pose. Left: the heavy-atom RMSD of the ligand to its start pose; "
            "right: the field score of the pose in the pocket model, frame by frame, which is the quantity that the dynamic pharmacophore averages. The best and the worst pose against the crystal are drawn thick.")


@figure("struct_dyn_trace", GROUP, "The traces of the poses of one ligand over 20 ps", trace_caption, data=(TABLE, f"{OUT}/ligand_dynamics/trajectories"), section="11.13")
def trace_figure(theme_name):
    from cypmode.validation import ligand_dynamics as ld
    from cypmode.validation.pose_pharmacophore import (
        benchmark_pools,
        benchmark_smiles,
        fields_and_sites,
        kept_indices,
        pose_in_frame,
    )
    from cypmode.validation.structures import load_structure

    th = apply(theme_name)
    ligand = example_ligand(poses())
    smiles = benchmark_smiles()[ligand]
    fields, sites = fields_and_sites(kept_indices(smiles, GATED))
    rows = []
    for run, sample, rmsd, cif, _ in benchmark_pools(ROOT / OUT)[ligand]:
        data = np.load(ROOT / OUT / "ligand_dynamics" / "trajectories" / f"{ligand}_{run}_{sample}.npz")
        in_frame = pose_in_frame(load_structure(cif), smiles)
        if in_frame is None:
            continue
        per_frame = ld.frame_scores(in_frame, data["frames"], fields, sites)
        drift = np.sqrt(((data["frames"] - data["start"][None]) ** 2).sum(axis=-1).mean(axis=-1))
        rows.append((rmsd, drift, per_frame["field"].to_numpy()))
    best, worst = min(r[0] for r in rows), max(r[0] for r in rows)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    for rmsd, drift, field in rows:
        thick = rmsd in (best, worst)
        colour = th["green"] if rmsd == best else th["red"] if rmsd == worst else th["muted"]
        ax.plot(np.arange(1, len(drift) + 1), drift, color=colour, lw=2.4 if thick else 1.0, alpha=1.0 if thick else 0.7)
        bx.plot(np.arange(1, len(field) + 1), field, color=colour, lw=2.4 if thick else 1.0, alpha=1.0 if thick else 0.7)
    for axis, label in ((ax, "RMSD to the start pose (A)"), (bx, "field score of the pose")):
        axis.axvspan(0.5, ld.EQUILIBRATION_FRAMES + 0.5, color=th["panel"], zorder=0)
        axis.set_xlabel("time (ps)")
        axis.set_ylabel(label)
        tidy(axis, th)
    fig.tight_layout()
    return fig
