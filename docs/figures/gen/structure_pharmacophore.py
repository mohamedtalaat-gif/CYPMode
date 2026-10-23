# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures of S8, the pharmacophore model of the predicted poses of the structure track: the check of the conformational bias, the check of the binding-site representation, and the expected
pharmacophore of the 20 official ligands.

The first two use no blinded compound and are shown as they are. The third is drawn from ``pharmacophore_official20/`` and shows predicted poses of the blinded ligands: its files are in the HELD BACK
block of ``.gitignore`` and the notebook embeds it with ``held_back=True``.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from matplotlib.patches import Circle
from style import apply, tidy

GROUP = "structure"
ROOT = Path(__file__).resolve().parents[3]
OUT = "data/structure_track/outputs/cyp3a4"
OFFICIAL = f"{OUT}/pharmacophore_official20"
FIELDS = "data/tdi_track/outputs/caches/pocket_fields_cyp3a4.npz"
FAMILY_COLOUR = {"Donor": "blue", "Acceptor": "red", "Aromatic": "green", "Hydrophobe": "orange", "PosIonizable": "purple", "NegIonizable": "teal"}
FAMILY_MARK = {"Donor": "D", "Acceptor": "A", "Aromatic": "R", "Hydrophobe": "H", "PosIonizable": "+", "NegIonizable": "−"}


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


# -- the conformational bias -------------------------------------------------------------------------------------------------------------------------------

def window_caption() -> str:
    table = _table(f"{OUT}/conformer_bias.csv")
    done = table[table.nearest_rmsd.notna()]
    covered = done[done.nearest_rmsd <= 1.0]
    block = float((done.ensemble_rmsd <= 1.0).mean())
    needed = covered.needed_energy.dropna()
    return (f"Does a low-energy ensemble contain the bound conformation? For {len(done)} crystal ligand copies of the CYP3A4 reference set ({done.comp_id.nunique()} distinct ligands) the heavy-atom RMSD of the nearest conformer to "
            f"the crystal pose: of 300 conformers and of the 12 conformers within 5 kcal/mol that the pocket block uses (left, cumulative; the dashed line is 1.0 Å). {len(covered) / len(done):.0%} of the copies have a conformer within "
            f"1.0 Å among the 300, {block:.0%} among the 12. For the copies that have one, the energy of the lowest such conformer above the global minimum (right): {(needed <= 5).mean():.0%} lie within 5 kcal/mol, "
            f"{(needed <= 10).mean():.0%} within 10 and {(needed <= 20).mean():.0%} within 20 (MMFF, in vacuum, so the energies are indicative).")


@figure("struct_conformer_window", GROUP, "Does a low-energy ensemble contain the bound conformation?", window_caption, data=(f"{OUT}/conformer_bias.csv",), section="11.11")
def conformer_window(theme_name):
    th = apply(theme_name)
    table = _table(f"{OUT}/conformer_bias.csv")
    done = table[table.nearest_rmsd.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax = axes[0]
    for label, column, colour in (("300 conformers", "nearest_rmsd", th["blue"]), ("12 conformers within 5 kcal/mol (the block)", "ensemble_rmsd", th["orange"])):
        values = np.sort(done[column].dropna().to_numpy())
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=colour, label=label)
    ax.axvline(1.0, color=th["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("RMSD of the nearest conformer to the crystal pose (Å)", fontsize=9)
    ax.set_ylabel("share of crystal copies", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(ax, th)
    ax = axes[1]
    needed = done[done.nearest_rmsd <= 1.0].needed_energy.dropna()
    ax.hist(needed.clip(upper=30), bins=np.arange(0, 31, 1.5), color=th["teal"])
    for window in (5, 10, 20):
        ax.axvline(window, color=th["muted"], linestyle="--", linewidth=1)
        ax.text(window + 0.3, ax.get_ylim()[1] * 0.92, f"{window}", fontsize=8, color=th["muted"])
    ax.set_xlabel("energy of the lowest conformer within 1.0 Å of the crystal pose (kcal/mol above the minimum)", fontsize=9)
    ax.set_ylabel("crystal copies", fontsize=9)
    tidy(ax, th)
    return fig


# -- the binding-site representation -----------------------------------------------------------------------------------------------------------------------

def site_caption() -> str:
    poses = _table(f"{OUT}/site_representation_poses.csv")
    flags = json.loads((ROOT / OUT / "site_representation_flags.json").read_text())
    return (f"How far is a pocket, and a pose, from what the crystal ligands have shown? Left: the pocket C-alpha RMSD (the {flags['core_residues']} residues most entries share) of each crystal copy to the nearest copy with another "
            f"PDB code and ligand, and of each of the {len(poses)} predicted complexes of the benchmark pools to the nearest crystal pocket (its own compound left out); the dashed line is the 95th percentile of the crystal "
            f"distribution ({flags['pocket_rmsd_flag']:.2f} Å), beyond which a pocket would be flagged as unseen. Right: the share of heavy atoms outside the support of the fields built without the compound, for the crystal poses "
            f"(at 0 Å on the horizontal axis) and for the predicted poses against their true RMSD; the dashed line is the 95th percentile of the crystal poses ({flags['support_cut']:.2f}). Both measures are coarse: "
            f"a backbone that agrees with the crystal says little about the side chains and loops that shape a cryptic pocket.")


@figure("struct_site_check", GROUP, "How far is a pocket, and a pose, from what the crystal ligands have shown?", site_caption,
        data=(f"{OUT}/site_representation_crystal.csv", f"{OUT}/site_representation_poses.csv", f"{OUT}/site_representation_flags.json"), section="11.11")
def site_check(theme_name):
    th = apply(theme_name)
    crystal, poses = _table(f"{OUT}/site_representation_crystal.csv"), _table(f"{OUT}/site_representation_poses.csv")
    flags = json.loads((ROOT / OUT / "site_representation_flags.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax = axes[0]
    upper = max(crystal.pocket_nearest_rmsd.max(), poses.pocket_nearest_rmsd.max(), flags["pocket_rmsd_flag"] * 1.5)
    bins = np.linspace(0, upper, 30)
    ax.hist(crystal.pocket_nearest_rmsd.dropna(), bins=bins, color=th["blue"], alpha=0.7, label="crystal copies, nearest other entry")
    ax.hist(poses.pocket_nearest_rmsd.dropna(), bins=bins, color=th["orange"], alpha=0.7, label="predicted complexes, nearest crystal pocket")
    ax.axvline(flags["pocket_rmsd_flag"], color=th["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("pocket C-alpha RMSD (Å)", fontsize=9)
    ax.set_ylabel("complexes", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax, th)
    ax = axes[1]
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, 0.12, len(crystal)), crystal.outside, s=14, color=th["blue"], alpha=0.7, label="crystal poses")
    ax.scatter(poses.true_rmsd, poses.outside, s=14, color=th["orange"], alpha=0.8, label="predicted poses")
    ax.axhline(flags["support_cut"], color=th["muted"], linestyle="--", linewidth=1)
    ax.set_xlabel("true RMSD of the pose (Å)", fontsize=9)
    ax.set_ylabel("share of heavy atoms outside the support", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    tidy(ax, th)
    return fig


# -- the expected pharmacophore of the 20 official ligands (held back) -----------------------------------------------------------------------------------

def official_caption() -> str:
    summary, common = _table(f"{OFFICIAL}/summary.csv"), _table(f"{OFFICIAL}/common_sites.csv")
    return (f"The expected pharmacophore of the {len(summary)} official ligands, from their final submitted poses in the frame of the heme (the iron is the orange dot). Top: the typed features of all final poses (small letters: "
            f"D donor, A acceptor, R aromatic ring, H hydrophobe, + and − ionisable) over the heavy-atom occupancy of the crystal ligands (grey), seen from above the heme and from the side; rings are the sites of the crystal model "
            f"and stars the {len(common)} sites at which at least 14 of the 20 poses place a feature of the same family. Bottom left: for each ligand the share of the site weight that its final pose matches (dot) and the mean over the "
            f"pool of its valid poses (bar, one standard deviation); a cross marks a pose beyond a flag of the check of the binding-site representation. Bottom right: the similarity of the 20 feature fields, ordered by clustering.")


@figure("struct_pharm_official20", GROUP, "The expected pharmacophore of the 20 official ligands", official_caption,
        data=(f"{OFFICIAL}/summary.csv", f"{OFFICIAL}/features.csv", f"{OFFICIAL}/common_sites.csv", f"{OFFICIAL}/similarity.csv", FIELDS), section="11.11")
def official(theme_name):
    from cypmode.challenge import pocket_pharmacophore as pp

    th = apply(theme_name)
    features, common, summary = _table(f"{OFFICIAL}/features.csv"), _table(f"{OFFICIAL}/common_sites.csv"), _table(f"{OFFICIAL}/summary.csv")
    similarity = pd.read_csv(ROOT / OFFICIAL / "similarity.csv", index_col=0)
    with np.load(ROOT / FIELDS) as archive:
        fields = {name: archive[name] for name in archive.files}
    sites = pp.sites_of(fields)
    fig = plt.figure(figsize=(11.5, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    views = (("top view (x, y)", fields["occupancy"].max(axis=2), ("x", "y"), (-14.5, 14.5, -14.5, 14.5)), ("side view (x, z)", fields["occupancy"].max(axis=1), ("x", "z"), (-14.5, 14.5, -3.5, 16.5)))
    for column, (title, image, (a, b), extent) in enumerate(views):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(image.T, origin="lower", extent=extent, cmap="Greys" if theme_name == "light" else "Greys_r", vmin=0, vmax=float(fields["occupancy"].max()), alpha=0.7)
        ax.plot([0], [0], marker="o", color=th["orange"], markersize=7, markeredgecolor=th["edge"], zorder=6)
        index = {"x": 0, "y": 1, "z": 2}
        for family, rows in sites.items():
            for site in rows:
                ax.add_patch(Circle((site[index[a]], site[index[b]]), pp.SITE_RADIUS, fill=False, edgecolor=th[FAMILY_COLOUR[family]], linewidth=1.2, alpha=0.8, zorder=3))
        for family, group in features.groupby("family"):
            for _, row in group.iterrows():
                ax.text(row[a], row[b], FAMILY_MARK[family], ha="center", va="center", fontsize=7, color=th[FAMILY_COLOUR[family]], alpha=0.9, zorder=5)
        ax.scatter(common[a], common[b], marker="*", s=150, color=[th[FAMILY_COLOUR[f]] for f in common.family], edgecolor=th["edge"], linewidth=0.6, zorder=7)
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(f"{a} (Å)", fontsize=9)
        ax.set_ylabel(f"{b} (Å)", fontsize=9)
        ax.set_title(title, fontsize=10)
        tidy(ax, th, grid=False)
    ax = fig.add_subplot(grid[1, 0])
    ordered = summary.set_index("ligand").loc[[name for name in similarity.index if name in set(summary.ligand)]].reset_index()
    y = np.arange(len(ordered))
    ax.barh(y, ordered.pool_recall_mean, xerr=ordered.pool_recall_sd, color=th["blue"], alpha=0.5, error_kw={"ecolor": th["muted"], "capsize": 2})
    ax.scatter(ordered.recall, y, color=th["orange"], zorder=3, s=22)
    flagged = ordered[(ordered.beyond_pocket_flag.astype(bool)) | (ordered.beyond_support_cut.astype(bool))]
    ax.scatter(flagged.recall + 0.02, [list(ordered.ligand).index(n) for n in flagged.ligand], marker="x", color=th["red"], zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered.ligand, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("share of the site weight matched (dot: final pose; bar: pool)", fontsize=9)
    tidy(ax, th)
    ax = fig.add_subplot(grid[1, 1])
    image = ax.imshow(similarity.to_numpy(), cmap="viridis" if theme_name == "dark" else "YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(similarity)))
    ax.set_xticklabels([n.replace("OCNT-", "") for n in similarity.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(similarity)))
    ax.set_yticklabels([n.replace("OCNT-", "") for n in similarity.index], fontsize=6)
    fig.colorbar(image, ax=ax, shrink=0.8, label="cosine similarity")
    return fig
