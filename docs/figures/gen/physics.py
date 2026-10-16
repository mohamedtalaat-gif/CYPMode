"""Figures of the quantum-chemistry features: the self-consistent field iterations of GFN2-xTB, Koopmans' relation on the real compounds, Boltzmann weights of conformers,
hydrogen-abstraction energies by site type, and the signal each physics column carries on its own."""

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from sklearn.metrics import roc_auc_score

from catalog import figure
from style import apply, tidy

GROUP = "physics"
ROOT = Path(__file__).resolve().parents[3]
CACHES = "data/tdi_track/outputs/caches"
TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
EXAMPLE = "Cc1ccccc1"  # toluene, the benzylic hydrogen-abstraction probe of the physics block
CYCLE = re.compile(r"^\s+(\d+)\s+(-?\d+\.\d+)\s+(-?[\d.]+E[+-]\d+)\s+([\d.]+E[+-]\d+)\s*$", re.M)


SINGLE_POINT = """
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tblite.interface import Calculator

mol = Chem.AddHs(Chem.MolFromSmiles(sys.argv[1]))
AllChem.EmbedMolecule(mol, randomSeed=7)
AllChem.MMFFOptimizeMolecule(mol)
xyz = mol.GetConformer().GetPositions() / 0.52917721067
numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
calc = Calculator("GFN2-xTB", numbers, xyz)
calc.set("verbosity", 2)
calc.singlepoint()
"""


@lru_cache(maxsize=None)
def scf_trace(smiles: str = EXAMPLE) -> pd.DataFrame:
    """The cycle table GFN2-xTB prints for one single point (total energy in Eh, energy error, density error). The library prints it from compiled code, so the calculation
    runs in a child process whose output is captured whole."""
    text = subprocess.run([sys.executable, "-c", SINGLE_POINT, smiles], capture_output=True, text=True, check=True).stdout
    rows = [(int(a), float(b), float(c), float(d)) for a, b, c, d in CYCLE.findall(text)]
    if not rows:
        raise RuntimeError("no SCF cycle table in the tblite output")
    return pd.DataFrame(rows, columns=["cycle", "energy_eh", "energy_error", "density_error"])


def scf_caption() -> str:
    trace = scf_trace()
    return (f"The self-consistent field iterations of GFN2-xTB for toluene (one conformer, MMFF geometry), taken from the library's own printout: {len(trace)} cycles. "
            f"Left: the distance of each cycle's total energy from the converged energy ({trace['energy_eh'].iloc[-1]:.6f} Eh), log scale. Right: the density error that decides convergence, "
            f"from {trace['density_error'].iloc[0]:.1e} in the first cycle to {trace['density_error'].iloc[-1]:.1e} in the last. Each cycle builds the Fock matrix from the current electron "
            f"density, solves for new orbitals, and mixes the new density into the old one until the density stops changing.")


def scf_animation(theme_name: str) -> Path:
    th = apply(theme_name)
    trace = scf_trace()
    final = trace["energy_eh"].iloc[-1]
    out = ROOT / "docs/figures" / GROUP / f"phys_scf_convergence-anim-{theme_name}.gif"
    frames = []
    for upto in range(1, len(trace) + 1):
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=110)
        part = trace.iloc[:upto]
        a.semilogy(part["cycle"], np.maximum(np.abs(part["energy_eh"] - final), 1e-12), "o-", color=th["blue"], lw=2)
        b.semilogy(part["cycle"], part["density_error"], "o-", color=th["red"], lw=2)
        a.set(xlim=(0.5, len(trace) + 0.5), ylim=(1e-12, 1e-1), xlabel="SCF cycle", ylabel="|E - E converged| (Eh)")
        b.set(xlim=(0.5, len(trace) + 0.5), ylim=(1e-6, 1), xlabel="SCF cycle", ylabel="density error")
        for ax in (a, b):
            tidy(ax, th)
        fig.suptitle(f"GFN2-xTB, toluene: cycle {upto}", fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 6, duration=550, loop=0)
    return out


@figure("phys_scf_convergence", GROUP, "The self-consistent field loop converging", scf_caption, data=("GFN2-xTB single point of toluene (tblite)",), section="10.3", animation=scf_animation)
def scf_convergence(theme_name):
    th = apply(theme_name)
    trace = scf_trace()
    final = trace["energy_eh"].iloc[-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4))
    a.semilogy(trace["cycle"][:-1], np.abs(trace["energy_eh"][:-1] - final), "o-", color=th["blue"], lw=2)
    a.set(xlabel="SCF cycle", ylabel="|E - E converged| (Eh)")
    b.semilogy(trace["cycle"], trace["density_error"], "o-", color=th["red"], lw=2)
    b.set(xlabel="SCF cycle", ylabel="density error")
    for ax in (a, b):
        tidy(ax, th)
    fig.tight_layout()
    return fig


def physics_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / CACHES / "physics_cache.csv")


def koopmans_caption() -> str:
    table = physics_table().dropna(subset=["phys_ip_ev", "phys_homo_ev", "phys_ea_ev", "phys_lumo_ev"])
    r_ip = np.corrcoef(-table["phys_homo_ev"], table["phys_ip_ev"])[0, 1]
    r_ea = np.corrcoef(-table["phys_lumo_ev"], table["phys_ea_ev"])[0, 1]
    offset = (table["phys_ip_ev"] + table["phys_homo_ev"]).mean()
    return (f"Koopmans' approximation on the {len(table)} compounds of the physics cache: the ionisation potential computed from the energy of the cation (vertical, vertical axis) against "
            f"minus the energy of the highest occupied orbital (horizontal). Pearson r = {r_ip:.3f} for the ionisation potential and {r_ea:.3f} for the electron affinity against minus the "
            f"lowest unoccupied orbital; the cation-based value sits {offset:+.2f} eV from the orbital value on average. Koopmans' approximation ignores how the remaining orbitals relax when an electron is removed, so the two are expected to track each other without being equal. The dashed line is equality.")


@figure("phys_koopmans", GROUP, "Orbital energies and ionisation potentials: Koopmans' relation on real compounds", koopmans_caption, data=(f"{CACHES}/physics_cache.csv",), section="10.4")
def koopmans(theme_name):
    th = apply(theme_name)
    table = physics_table().dropna(subset=["phys_ip_ev", "phys_homo_ev", "phys_ea_ev", "phys_lumo_ev"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    a.scatter(-table["phys_homo_ev"], table["phys_ip_ev"], s=5, alpha=0.3, color=th["blue"], linewidths=0)
    b.scatter(-table["phys_lumo_ev"], table["phys_ea_ev"], s=5, alpha=0.3, color=th["red"], linewidths=0)
    for ax, xlabel, ylabel in ((a, "minus the HOMO energy (eV)", "ionisation potential from the cation (eV)"), (b, "minus the LUMO energy (eV)", "electron affinity from the anion (eV)")):
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "--", color=th["muted"], lw=1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        tidy(ax, th)
    fig.tight_layout()
    return fig


BOLTZMANN_KCAL = 0.0019872041  # kcal / (mol K)
LADDER = (0.0, 0.4, 0.9, 1.5, 2.5)  # relative conformer energies in kcal/mol, an illustrative ladder


def boltzmann(energies, temperature):
    w = np.exp(-np.asarray(energies) / (BOLTZMANN_KCAL * temperature))
    return w / w.sum()


@figure("phys_boltzmann", GROUP, "Boltzmann weights of conformers", "A ladder of five conformer energies (0, 0.4, 0.9, 1.5 and 2.5 kcal/mol above the lowest, chosen to illustrate the formula) and the weight each conformer "
        "gets in a thermal average, w_i = exp(-E_i / kT) / sum_j exp(-E_j / kT), at 298 K and at 600 K. At 298 K the lowest conformer carries about half of the weight; hotter, the weights even out. "
        "The tier-B physics block averages its properties with these weights over up to three conformers.", section="10.7")
def boltzmann_weights(theme_name):
    th = apply(theme_name)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    for i, e in enumerate(LADDER):
        a.hlines(e, 0.2, 0.8, color=th["blue"], lw=4)
        a.text(0.85, e, f"{e:.1f}", va="center", fontsize=9)
    a.set(ylabel="energy above the lowest conformer (kcal/mol)", xlim=(0, 1.2), xticks=[], ylim=(-0.2, 2.9))
    tidy(a, th, grid=False)
    x = np.arange(len(LADDER))
    for offset, temperature, color in ((-0.2, 298.0, th["blue"]), (0.2, 600.0, th["red"])):
        b.bar(x + offset, boltzmann(LADDER, temperature), 0.38, color=color, label=f"{temperature:.0f} K")
    b.set(xlabel="conformer", ylabel="weight in the thermal average", xticks=x, xticklabels=[f"{e:.1f}" for e in LADDER])
    b.legend(frameon=False)
    tidy(b, th, grid=False)
    fig.tight_layout()
    return fig


SITES = {"phys_b_site1_is_alpha_N": "next to N", "phys_b_site1_is_alpha_O": "next to O", "phys_b_site1_is_benzylic": "benzylic", "phys_b_site1_is_allylic": "allylic", "phys_b_site1_is_aliphatic": "aliphatic"}


def hat_table() -> pd.DataFrame:
    table = pd.read_csv(ROOT / CACHES / "physics_b_cache.csv").dropna(subset=["phys_b_hat_relaxed_1"])
    table["site"] = table[list(SITES)].idxmax(axis=1).map(SITES)
    return table[table[list(SITES)].sum(axis=1) > 0]


def hat_caption() -> str:
    table = hat_table()
    medians = table.groupby("site")["phys_b_hat_relaxed_1"].median().sort_values()
    parts = ", ".join(f"{site} {value:.0f}" for site, value in medians.items())
    return (f"Energy needed to pull a hydrogen off the easiest site of each compound (GFN2-xTB, atoms around the radical relaxed, kcal/mol), by the kind of site, for the {len(table)} compounds of the "
            f"tier-B cache. Median by site: {parts}. A lower energy means an easier abstraction, which is the first step of the oxidation that makes a time-dependent inhibitor.")


@figure("phys_hat_by_site", GROUP, "Hydrogen-abstraction energy by kind of site", hat_caption, data=(f"{CACHES}/physics_b_cache.csv",), section="10.6")
def hat_by_site(theme_name):
    th = apply(theme_name)
    table = hat_table()
    order = table.groupby("site")["phys_b_hat_relaxed_1"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    data = [table.loc[table["site"] == s, "phys_b_hat_relaxed_1"].to_numpy() for s in order]
    box = ax.boxplot(data, tick_labels=[f"{s}\n(n = {len(d)})" for s, d in zip(order, data, strict=True)], patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set(facecolor=th["blue"], alpha=0.55, edgecolor=th["text"])
    for line in box["medians"]:
        line.set(color=th["red"], lw=2)
    ax.set(ylabel="H-abstraction energy (kcal/mol)")
    tidy(ax, th)
    fig.tight_layout()
    return fig


def single_column_auc(isoform: str = "CYP3A4") -> pd.Series:
    train = pd.read_csv(ROOT / TRAIN)
    keep = train[f"{isoform}_is_TDI"].notna() & train[f"{isoform}_pIC50_TDI_condition"].notna() & train[f"{isoform}_pIC50_direct_inhibition"].notna()
    labelled = train.loc[keep, ["SMILES", f"{isoform}_is_TDI"]].rename(columns={f"{isoform}_is_TDI": "label"})
    labelled["SMILES"] = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in labelled["SMILES"]]  # the cache is keyed by canonical SMILES
    merged = labelled.merge(physics_table(), on="SMILES", how="inner")
    if len(merged) < 0.9 * len(labelled):
        raise RuntimeError(f"only {len(merged)} of {len(labelled)} labelled compounds found in the physics cache")
    scores = {}
    for column in merged.columns.difference(["SMILES", "label"]):
        values = merged[column]
        if values.notna().sum() < 300 or values.nunique() < 3:
            continue
        filled = values.fillna(values.median())
        scores[column] = roc_auc_score(merged["label"].astype(int), filled)
    return pd.Series(scores).sort_values(key=lambda s: -(s - 0.5).abs())


def final_stack_auc(isoform: str = "CYP3A4") -> float:
    return float(pd.read_csv(ROOT / "data/tdi_track/outputs/final" / f"{isoform}_stack_choice.csv").query("stack == 'default'")["auc_z1.0"].iloc[0])


def column_caption() -> str:
    scores = single_column_auc()
    return (f"For the labelled CYP3A4 compounds, the AUC with which each physics column separates the time-dependent inhibitors from the rest when used alone, as a rank score (0.5 means no signal; "
            f"values below 0.5 mean the label is more common at low values); the {min(15, len(scores))} columns furthest from 0.5 of {len(scores)} with enough data. The strongest single column "
            f"reaches {(scores - 0.5).abs().iloc[0] + 0.5:.3f} once its direction is taken into account, far below the {final_stack_auc():.3f} of the final stack, which is why the physics block is judged by what it adds to the stack, not alone.")


@figure("phys_single_column_signal", GROUP, "How much signal each physics column carries alone", column_caption, data=(f"{CACHES}/physics_cache.csv", TRAIN), section="10.9")
def single_column_signal(theme_name):
    th = apply(theme_name)
    scores = single_column_auc().head(15)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh([c.removeprefix("phys_") for c in scores.index][::-1], (scores - 0.5).to_numpy()[::-1], left=0.5, color=[th["blue"] if v > 0.5 else th["red"] for v in scores.to_numpy()[::-1]])
    ax.axvline(0.5, color=th["text"], lw=1)
    ax.set(xlabel="AUC of the column alone")
    ax.tick_params(axis="y", labelsize=8)
    tidy(ax, th)
    fig.tight_layout()
    return fig

