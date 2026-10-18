"""Concept figures of the structure track: how the minimum-Bayes-risk rule picks a pose from a pool (on a real pool), and how a diffusion model turns noise into a pose (a toy denoiser
that knows two poses, driven by the real forward and reverse equations)."""

from functools import cache
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from PIL import Image
from style import apply, tidy

from gen.pymol_scene import cif, pose_comparison

GROUP = "structure"
ROOT = Path(__file__).resolve().parents[3]
OUT = "data/structure_track/outputs/cyp3a4"
POOL_LIGAND = "bench-PK9"
CRYSTAL = ("4D75", "PK9")  # the crystal entry of the pool ligand with the iron-nitrogen contact
BENCHMARK = "data/structure_track/inputs/structure_benchmark.csv"


def pool_matrix():
    """(sample indices, LDDT-PLI matrix, RMSD to the crystal pose per sample, index picked by minimum Bayes risk) for the Boltz samples of the pool ligand."""
    from cypmode.validation.pose_pools import mbr_pick, pairwise_lddt, pool_paths

    benchmark = pd.read_csv(ROOT / BENCHMARK)
    smiles = dict(zip(benchmark["Molecule_Name"], benchmark["SMILES"], strict=True))
    pool = [(engine, sample, path) for engine, sample, path in pool_paths(POOL_LIGAND, ("boltz",)) if path.exists()]
    matrix = pairwise_lddt([path for _, _, path in pool], smiles=smiles.get(POOL_LIGAND))
    rmsd = pd.read_csv(ROOT / OUT / "benchmark_engine_pose_validation_unguided_boltz.csv")
    rmsd = rmsd[rmsd["compound"] == POOL_LIGAND].set_index("sample")["true_rmsd"]
    samples = [sample for _, sample, _ in pool]
    pick = mbr_pick(matrix)
    outcome = pd.read_csv(ROOT / OUT / "selection_validation_outcomes.csv").set_index("compound").loc[POOL_LIGAND, "mbr_boltz"]
    if abs(rmsd.loc[samples[pick]] - outcome) > 1e-6:
        raise AssertionError("the pick drawn here is not the pick the selection validation scored")
    return samples, matrix, rmsd.loc[samples].to_numpy(), pick


def pool_picture(pick: int, theme_name: str):
    """The Boltz samples of the pool ligand in the pocket: the pose picked by the rule is the reference (teal), the other samples are blue, the crystal ligand is white."""
    from cypmode.validation.pose_pools import pool_paths

    paths = [path for _, _, path in pool_paths(POOL_LIGAND, ("boltz",)) if path.exists()]
    others = [(path, "0x6f8fe0" if theme_name == "dark" else "0x2457c5") for i, path in enumerate(paths) if i != pick]
    return pose_comparison(paths[pick], others, theme=theme_name, size=(1500, 1100), key=f"mbr_pool_{POOL_LIGAND}", crystal=(cif(CRYSTAL[0]), CRYSTAL[1]), zoom=7.0)


def mbr_caption() -> str:
    samples, _matrix, rmsd, pick = pool_matrix()
    return (f"Minimum Bayes risk on a real pool: the {len(samples)} Boltz samples of {POOL_LIGAND.removeprefix('bench-')}, a crystal-structure ligand. First panel: the pool in the pocket (rendered from the predicted coordinates, aligned on the crystal structure {CRYSTAL[0]}). Then: how much each pair of poses agrees (LDDT-PLI: the share of "
            f"ligand-protein distances the two poses reproduce, 1 = identical). Middle: each pose's mean agreement with the others, its expected quality if the others were the truth; the highest is picked "
            f"(sample {samples[pick]}). Right: the distance of each pose to the crystal pose, which the rule never sees: the pick is at {rmsd[pick]:.2f} A against {rmsd[0]:.2f} A for the first sample "
            f"and a best possible {rmsd.min():.2f} A. This is one ligand; the gain over the benchmark is driven by it, which is why the rule is reported as exploratory.")


@figure("struct_mbr_pool", GROUP, "How minimum Bayes risk picks a pose from a pool", mbr_caption, data=(f"{OUT}/pose_pool_cache", f"{OUT}/selection_validation_outcomes.csv"), section="11.8")
def mbr_pool(theme_name):
    th = apply(theme_name)
    samples, matrix, rmsd, pick = pool_matrix()
    n = len(samples)
    expected = (matrix.sum(axis=1) - 1) / (n - 1)
    fig, (pool_ax, a, b, c) = plt.subplots(1, 4, figsize=(17.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1.05, 0.95, 0.95]})
    pool_ax.imshow(plt.imread(pool_picture(pick, theme_name).png))
    pool_ax.set_xlim(350, 1450)
    pool_ax.set_ylim(900, 200)
    pool_ax.axis("off")
    pool_ax.set_title("the pool in the pocket: crystal pose (white), picked pose (teal), other samples (blue)", fontsize=8.5, loc="left", color=th["text"])
    image = a.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    a.set_xticks(range(n), samples)
    a.set_yticks(range(n), samples)
    a.set(xlabel="sample", ylabel="sample", title="agreement between poses (LDDT-PLI)")
    for i in range(n):
        for j in range(n):
            a.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if matrix[i, j] < 0.6 else "black")
    fig.colorbar(image, ax=a, fraction=0.046)
    colors = [th["red"] if i == pick else th["muted"] for i in range(n)]
    b.bar([str(s) for s in samples], expected, color=colors)
    b.set(xlabel="sample", ylabel="mean agreement with the others", title="expected quality: pick the highest", ylim=(0, 1))
    tidy(b, th, grid=False)
    c.bar([str(s) for s in samples], rmsd, color=colors)
    c.axhline(2.0, color=th["muted"], ls=":", lw=1)
    c.set(xlabel="sample", ylabel="RMSD to the crystal pose (A)", title="the truth, which the rule never sees")
    tidy(c, th, grid=False)
    fig.tight_layout()
    return fig


REAL_LIGAND = "OCNT-2395712"  # a donor ligand of the official set: its restrained and its unguided Boltz-2 pose are the two poses the denoiser knows


@cache
def real_poses():
    """The two real poses of the denoiser: the first Boltz-2 sample of the ligand with the restraint (pose A) and without it (pose B), the second superposed on the first by the protein backbone, both
    projected on the plane of the ligand's two principal axes (angstrom, centred). Also the bonds of the ligand, found from the distances of pose A."""
    from Bio.PDB import MMCIFParser, Superimposer

    from cypmode.validation.pose_consensus import boltz_cif_paths

    parser = MMCIFParser(QUIET=True)
    a, b = (parser.get_structure(name, str(boltz_cif_paths(name, "cyp3a4")[0]))[0] for name in (REAL_LIGAND + "-rs", REAL_LIGAND))
    backbone = [[r["CA"] for r in model.get_residues() if "CA" in r] for model in (a, b)]
    n = min(len(backbone[0]), len(backbone[1]))
    superimposer = Superimposer()
    superimposer.set_atoms(backbone[0][:n], backbone[1][:n])
    superimposer.apply(list(b.get_atoms()))

    def ligand(model):
        return np.array([x.coord for x in model.get_atoms() if x.get_parent().get_resname().startswith("LIG") and x.element != "H"])

    xa, xb = ligand(a), ligand(b)
    centre = xa.mean(axis=0)
    xa, xb = xa - centre, xb - centre
    plane = np.linalg.svd(xa)[2][:2].T
    bonds = [(i, j) for i in range(len(xa)) for j in range(i + 1, len(xa)) if np.linalg.norm(xa[i] - xa[j]) < 1.75]
    return xa @ plane, xb @ plane, bonds


STEPS = 40


def alpha_bar(s):
    """Cosine noise schedule: the share of the signal left at diffusion time s in [0, 1] (1 = noise only)."""
    return np.cos((s + 0.008) / 1.008 * np.pi / 2) ** 2


def denoise(seed: int, steps: int = STEPS):
    """Deterministic reverse diffusion (DDIM) with a denoiser that knows the two poses: its estimate of the clean pose is the posterior mean over them given the noisy points."""
    rng = np.random.default_rng(seed)
    pose_a, pose_b, _ = real_poses()
    poses = np.stack([pose_a, pose_b])
    x = rng.standard_normal(pose_a.shape) * 4.0
    path = [(1.0, x.copy(), np.array([0.5, 0.5]))]
    times = np.linspace(1.0, 0.0, steps + 1)
    for s, s_next in pairwise(times):
        a, a_next = alpha_bar(s), alpha_bar(s_next)
        distance = ((x[None] - np.sqrt(a) * poses) ** 2).sum(axis=(1, 2)) / (2 * (1 - a + 1e-9))
        weights = np.exp(-(distance - distance.min()))
        weights /= weights.sum()
        x0 = (weights[:, None, None] * poses).sum(axis=0)
        eps = (x - np.sqrt(a) * x0) / np.sqrt(1 - a + 1e-9)
        x = np.sqrt(a_next) * x0 + np.sqrt(1 - a_next) * eps
        path.append((s_next, x.copy(), weights))
    return path


def seeds_by_pose() -> tuple[int, int]:
    """The first noise draw that ends in pose A and the first that ends in pose B."""
    ends = {}
    for seed in range(64):
        ends.setdefault(int(denoise(seed)[-1][2].argmax()), seed)
        if len(ends) == 2:
            break
    return ends[0], ends[1]


def diffusion_caption() -> str:
    pose_a, pose_b, _bonds = real_poses()
    moved = float(np.linalg.norm(pose_a - pose_b, axis=1).mean())
    return (f"Diffusion on a real ligand: the {len(pose_a)} heavy atoms of {REAL_LIGAND}, a donor ligand of the official set, projected on the plane of their two principal axes. The denoiser knows two ways it "
            f"can sit: Boltz-2's first sample with the iron-nitrogen restraint (pose A, blue) and without it (pose B, red), which differ by {moved:.1f} A per atom on average after superposing the proteins. Forward, "
            f"noise is mixed in: x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) e, with a_t falling from 1 to 0 on a cosine schedule. Reverse, from pure noise, each of {STEPS} steps estimates the clean pose as the "
            f"posterior mean over the two poses given the noisy atoms and removes part of the noise. Different noise draws commit to different poses (top row: a start that ends in pose A; bottom row: one "
            f"that ends in pose B), which is where the diversity of a pool of samples comes from. A real network replaces the hand-made denoiser with one learned from thousands of structures.")


def snapshot(ax, x, th, poses=True):
    pose_a, pose_b, bonds = real_poses()
    if poses:
        for pose, colour in ((pose_a, th["blue"]), (pose_b, th["red"])):
            for i, j in bonds:
                ax.plot(pose[[i, j], 0], pose[[i, j], 1], "-", color=colour, alpha=0.2, lw=5, solid_capstyle="round")
    for i, j in bonds:
        ax.plot(x[[i, j], 0], x[[i, j], 1], "-", color=th["text"], lw=1.4)
    ax.plot(x[:, 0], x[:, 1], "o", color=th["orange"], ms=3.6)
    extent = float(np.abs(np.concatenate([pose_a, pose_b])).max()) + 3
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xticks=[], yticks=[], aspect="equal")


def diffusion_animation(theme_name: str) -> Path:
    th = apply(theme_name)
    out = ROOT / "docs/figures" / GROUP / f"struct_diffusion_denoising-anim-{theme_name}.gif"
    path = denoise(seeds_by_pose()[1])
    frames = []
    for step, (s, x, weights) in enumerate(path):
        fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=110)
        snapshot(ax, x, th)
        ax.set_title(f"step {step} of {STEPS}:  weight of pose A {weights[0]:.2f}, pose B {weights[1]:.2f}", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 8, duration=140, loop=0)
    return out


@figure("struct_diffusion_denoising", GROUP, "Diffusion on a real ligand: from noise to a pose", diffusion_caption, data=(f"{OUT}/<ligand>/predictions", "two Boltz-2 samples of one donor ligand"), section="11.4", animation=diffusion_animation)
def diffusion_denoising(theme_name):
    th = apply(theme_name)
    fig, axes = plt.subplots(2, 5, figsize=(12.4, 5.4))
    for row, seed in enumerate(seeds_by_pose()):
        path = denoise(seed)
        for col, step in enumerate((0, 12, 20, 28, STEPS)):
            s, x, _weights = path[step]
            snapshot(axes[row][col], x, th)
            axes[row][col].set_title(f"step {step}  (a = {alpha_bar(s):.2f})", fontsize=8.5)
    fig.tight_layout()
    return fig
