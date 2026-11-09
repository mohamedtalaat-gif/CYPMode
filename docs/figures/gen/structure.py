# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Figures of the structure track: the effect of the iron-nitrogen restraint ligand by ligand, the pose-selection rules, the OpenFold3 memory arithmetic and the PoseBusters checks."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catalog import figure
from style import apply, tidy

from gen.pymol_scene import cif, pose_comparison

GROUP = "structure"
ROOT = Path(__file__).resolve().parents[3]
OUT = "data/structure_track/outputs/cyp3a4"
INPUTS = "data/structure_track/inputs"
UNGUIDED = f"{OUT}/benchmark_engine_pose_validation_unguided_boltz.csv"
RESTRAINTS = {"unguided": UNGUIDED, "contact restraint": f"{OUT}/benchmark_engine_pose_validation_fecontact.csv", "bond restraint": f"{OUT}/benchmark_engine_pose_validation_febond.csv"}


def first_sample_rmsd() -> pd.DataFrame:
    """Ligand RMSD to the crystal pose of sample 0, for the ligands that were run under all three conditions (rows: ligand, columns: condition)."""
    columns = {}
    for name, path in RESTRAINTS.items():
        frame = pd.read_csv(ROOT / path)
        frame["ligand"] = frame["compound"].str.replace("-febond", "").str.replace("-fecontact", "")
        columns[name] = frame[frame["sample"] == 0].set_index("ligand")["true_rmsd"]
    table = pd.DataFrame(columns).dropna()
    return table


PICTURE_LIGAND = "bench-A1A4S"  # one of the two ligands where the restraint gained most; its crystal entry is 9COV
PICTURE_CRYSTAL = ("9COV", "A1A4S")


def restraint_picture(theme_name: str):
    """The first sample of the ligand without and with the bond restraint, aligned on the crystal structure (white ligand)."""
    from cypmode.validation.pose_consensus import boltz_cif_paths

    unguided, restrained = boltz_cif_paths(PICTURE_LIGAND, "cyp3a4")[0], boltz_cif_paths(PICTURE_LIGAND + "-febond", "cyp3a4")[0]
    return pose_comparison(restrained, [(unguided, "0xff6b6b" if theme_name == "dark" else "0xc43a3a")], theme=theme_name, size=(1800, 1100), key=f"restraint_{PICTURE_LIGAND}", crystal=(cif(PICTURE_CRYSTAL[0]), PICTURE_CRYSTAL[1]))


def restraint_caption() -> str:
    table = first_sample_rmsd()
    gain = (table["unguided"] - table["bond restraint"]).sort_values(ascending=False)
    best = ", ".join(f"{name.removeprefix('bench-')} {value:+.2f} A" for name, value in gain.head(2).items())
    return (f"First panel: one ligand ({PICTURE_LIGAND.removeprefix('bench-')}, crystal entry {PICTURE_CRYSTAL[0]}) in the pocket, the predicted poses aligned on the crystal structure and rendered from their coordinates. Then: ligand RMSD to the crystal pose of the first sample, for the {len(table)} crystal ligands that were folded without a restraint, with a soft contact between the iron and a donor "
            f"nitrogen, and with a bond-length restraint. Means: unguided {table['unguided'].mean():.2f} A, contact {table['contact restraint'].mean():.2f} A, bond {table['bond restraint'].mean():.2f} A. "
            f"The gain of the bond restraint over no restraint is concentrated in two ligands ({best}); the other {len(table) - 2} change by less than {gain.iloc[2:].abs().max():.2f} A. "
            f"With six ligands the evidence is real but thin.")


@figure("struct_restraint_effect", GROUP, "The iron-nitrogen restraint, ligand by ligand", restraint_caption,
        data=tuple(RESTRAINTS.values()), section="11.6")
def restraint_effect(theme_name):
    th = apply(theme_name)
    table = first_sample_rmsd()
    gain = (table["unguided"] - table["bond restraint"]).sort_values(ascending=False)
    fig, (pic, a, b) = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1.1, 0.95]})
    scene = restraint_picture(theme_name)
    pic.imshow(plt.imread(scene.png))
    pic.set_xlim(330, 1470)
    pic.set_ylim(930, 190)
    pic.axis("off")
    pic.set_title(f"{PICTURE_LIGAND.removeprefix('bench-')} in the pocket: crystal pose (white), without the restraint (red), with it (teal)", fontsize=9, loc="left", color=th["text"])
    names = list(RESTRAINTS)
    label_y = {}
    last = -np.inf
    for ligand in table["bond restraint"].sort_values().index:
        label_y[ligand] = max(table.loc[ligand, "bond restraint"], last + 0.2)
        last = label_y[ligand]
    for ligand, row in table.iterrows():
        better = row["bond restraint"] < row["unguided"] - 0.5
        a.plot(range(3), row[names].to_numpy(), "-o", color=th["green"] if better else th["muted"], lw=1.8 if better else 1.1, alpha=0.9, ms=5)
        a.text(2.06, label_y[ligand], ligand.removeprefix("bench-"), va="center", fontsize=8, color=th["text"])
    a.plot(range(3), table[names].mean().to_numpy(), "s-", color=th["red"], lw=2.6, ms=8, label="mean")
    a.axhline(2.0, color=th["muted"], ls=":", lw=1)
    a.set_xticks(range(3), names)
    a.set(ylabel="ligand RMSD to the crystal pose (Å)", xlim=(-0.2, 2.5))
    a.legend(frameon=False, loc="upper right")
    tidy(a, th)
    b.barh([n.removeprefix("bench-") for n in gain.index][::-1], gain.to_numpy()[::-1], color=[th["green"] if v > 0 else th["red"] for v in gain.to_numpy()[::-1]])
    b.axvline(0, color=th["text"], lw=1)
    b.set(xlabel="RMSD gain of the bond restraint (Å, positive = better)")
    tidy(b, th)
    fig.tight_layout()
    return fig


def selection_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / OUT / "selection_validation_outcomes.csv").set_index("compound")


RULES = {"first_boltz": "first Boltz pick", "mbr_boltz": "minimum Bayes risk\n(Boltz pool)", "mean_pool_boltz": "pool average", "oracle_boltz": "best in pool\n(oracle)", "first_openfold3": "first OpenFold3\nsample"}


def selection_caption() -> str:
    table = selection_table()
    first, mbr, oracle = table["first_boltz"], table["mbr_boltz"], table["oracle_boltz"]
    gain = first - mbr
    without = gain.drop(index=[gain.idxmax()])
    return (f"Ligand RMSD to the crystal pose of the pose each rule picks, for the {len(table)} benchmark ligands. Means: first Boltz sample {first.mean():.2f} A, "
            f"minimum-Bayes-risk pick {mbr.mean():.2f} A, best sample in the pool {oracle.mean():.2f} A, first OpenFold3 sample {table['first_openfold3'].mean():.2f} A. "
            f"The minimum-Bayes-risk rule closes {(first.mean() - mbr.mean()) / (first.mean() - oracle.mean()):.0%} of the distance to the oracle, but the mean gain of {gain.mean():.2f} A "
            f"is driven by one ligand ({gain.idxmax().removeprefix('bench-')}, {gain.max():.2f} A); without it the mean gain is {without.mean():.2f} A. The rule is therefore reported as exploratory.")


@figure("struct_selection_rules", GROUP, "Which pose to hand in: the selection rules on the benchmark", selection_caption,
        data=(f"{OUT}/selection_validation_outcomes.csv",), section="11.8")
def selection_rules(theme_name):
    th = apply(theme_name)
    table = selection_table()
    keys = list(RULES)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    gain = table["first_boltz"] - table["mbr_boltz"]
    driver = gain.idxmax()
    for ligand, row in table.iterrows():
        highlight = ligand == driver
        ax.plot(range(len(keys)), row[keys].to_numpy(), "-o", color=th["red"] if highlight else th["muted"], lw=2.2 if highlight else 1, alpha=0.95 if highlight else 0.55, ms=4)
        if highlight:
            ax.text(len(keys) - 0.9, row[keys[-1]], ligand.removeprefix("bench-"), color=th["red"], fontsize=8, va="center")
    ax.plot(range(len(keys)), table[keys].mean().to_numpy(), "s-", color=th["blue"], lw=2.8, ms=8, label="mean over ligands")
    ax.axhline(2.0, color=th["muted"], ls=":", lw=1)
    ax.set_xticks(range(len(keys)), [RULES[k] for k in keys], fontsize=8.5)
    ax.set(ylabel="ligand RMSD to the crystal pose (Å)")
    ax.legend(frameon=False, loc="upper left")
    tidy(ax, th)
    fig.tight_layout()
    return fig


HEME_ATOMS = 43  # heavy atoms of the CCD component HEM
CAP_GIB = 15.39  # the MPS working-set limit reported by the runtime on this machine
HEADS = 4  # no_heads_pair of the OpenFold3 pairformer (model_config.py)
CHUNK = 128  # chunk_size of the runner configuration


def allocation_gib(tokens, chunk=None):
    """Bytes of the attention-logit tensors of one triangle-attention layer: 3 tensors x heads x rows x N x N x 4 bytes (rows = N, or the chunk)."""
    rows = tokens if chunk is None else np.minimum(tokens, chunk)
    return 3 * HEADS * rows * tokens**2 * 4 / 2**30


VARIANT = re.compile(r"-(u|rs|ex|cf|ss|seed\d+|em\d*)$")


def token_counts() -> pd.DataFrame:
    from rdkit import Chem

    rows = []
    for path in sorted((ROOT / INPUTS).glob("cyp3a4_*.yaml")):
        if VARIANT.search(path.stem) or "series-" in path.stem:
            continue  # the generated variants of an input (restraint, seed, template levers) have the same token count as their base
        spec = yaml.safe_load(path.read_text())
        residues = sum(len(item["protein"]["sequence"]) for item in spec["sequences"] if "protein" in item)
        ligand = sum(Chem.MolFromSmiles(item["ligand"]["smiles"]).GetNumHeavyAtoms() for item in spec["sequences"] if "smiles" in item.get("ligand", {}))
        rows.append({"input": path.stem.removeprefix("cyp3a4_"), "tokens": residues + ligand + HEME_ATOMS})
    return pd.DataFrame(rows)


def memory_caption() -> str:
    frame = token_counts()
    ynv = int(frame.loc[frame["input"] == "bench-YNV", "tokens"].iloc[0])
    return (f"Memory of one triangle-attention layer of the OpenFold3 pairformer against the number of tokens N (residues, ligand heavy atoms and the {HEME_ATOMS} atoms of the heme): "
            f"3 tensors x {HEADS} heads x N rows x N x N x 4 bytes = {3 * HEADS * 4} N^3 bytes. For bench-YNV, N = {ynv} and the formula gives {allocation_gib(ynv):.2f} GiB, "
            f"the size of the buffer the runtime refused in the first probe (log: Invalid buffer size 9.27 GiB). The runtime's working-set limit on this machine is {CAP_GIB} GiB and "
            f"the weights and activations already occupy about half of it, so the layer cannot be allocated whole. Cutting the rows into chunks of {CHUNK} makes the layer need "
            f"{allocation_gib(ynv, CHUNK):.2f} GiB at the same N. Dots mark the {len(frame)} base inputs of this project (one per ligand), N from {frame['tokens'].min()} to {frame['tokens'].max()}.")


@figure("struct_openfold3_memory", GROUP, "OpenFold3 on a 16 GB machine: the memory arithmetic and the chunking that fixes it", memory_caption,
        data=(f"{INPUTS}/cyp3a4_*.yaml", f"{OUT}/logs/openfold3_probe_config.log"), section="11.9")
def openfold3_memory(theme_name):
    th = apply(theme_name)
    frame = token_counts()
    n = np.linspace(100, 900, 400)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.semilogy(n, allocation_gib(n), color=th["red"], lw=2.6, label=f"whole layer: {3 * HEADS * 4} N^3 bytes")
    ax.semilogy(n, allocation_gib(n, CHUNK), color=th["green"], lw=2.6, label=f"rows in chunks of {CHUNK}")
    ax.axhline(CAP_GIB, color=th["text"], ls="--", lw=1.4)
    ax.text(110, CAP_GIB * 1.12, f"working-set limit {CAP_GIB} GiB", fontsize=9)
    ax.axhline(CAP_GIB / 2, color=th["muted"], ls=":", lw=1)
    ax.text(110, CAP_GIB / 2 * 1.12, "about half is already in use", fontsize=8, color=th["muted"])
    ax.semilogy(frame["tokens"], allocation_gib(frame["tokens"].to_numpy()), "o", color=th["red"], ms=5, alpha=0.8)
    ax.semilogy(frame["tokens"], allocation_gib(frame["tokens"].to_numpy(), CHUNK), "o", color=th["green"], ms=5, alpha=0.8)
    ax.set(xlabel="tokens N", ylabel="one layer's attention tensors (GiB, log scale)", xlim=(100, 900), ylim=(0.01, 200))
    ax.legend(frameon=False, loc="lower right")
    tidy(ax, th)
    fig.tight_layout()
    return fig


def posebusters_table() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / OUT / "posebusters_validity_official20.csv")
    checks = [c for c in frame.columns if c not in ("compound", "all_passed", "model_index")]
    return frame.set_index("compound")[checks].astype(bool)


def posebusters_caption() -> str:
    table = posebusters_table()
    failed = int((~table.all(axis=1)).sum())
    return (f"The {table.shape[1]} PoseBusters checks (columns) on the {table.shape[0]} ligand poses (rows) of the official CYP3A4 set in the final structure zip, as written by the code that builds the zip. "
            f"Green is a pass: {failed} of {table.shape[0]} poses fail any check. The audit repeats the check.")


@figure("struct_posebusters", GROUP, "PoseBusters: every check on every submitted pose", posebusters_caption,
        data=(f"{OUT}/posebusters_validity_official20.csv",), section="11.12")
def posebusters(theme_name):
    th = apply(theme_name)
    table = posebusters_table()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.imshow(table.to_numpy().astype(float), aspect="auto", cmap=plt.matplotlib.colors.ListedColormap([th["red"], th["green"]]), vmin=0, vmax=1)
    ax.set_xticks(range(table.shape[1]), [c.replace("_", " ") for c in table.columns], rotation=75, ha="right", fontsize=7.5)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0]), minor=True)
    ax.grid(which="minor", color=th["background"], lw=1.2)
    ax.tick_params(which="minor", length=0)
    ax.grid(False, which="major")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig
