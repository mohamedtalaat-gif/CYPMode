# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The pharmacophore atlas of the structure track (declared on 2026-10-23 before any code of it ran): the pharmacophore model of the poses drawn on real poses in 3D and on the
structures in 2D, with the functional groups and the structure-activity relationships of the series.

``struct_pharm_pose_3d`` and ``struct_pharm_series`` use benchmark ligands, whose crystal structures are public, and are shown as they are. ``struct_pharm_official20_2d`` and ``struct_pharm_official20_3d`` show the
final poses of the 20 official ligands and the numbers of job 925_official_pharmacophore: their files are in the HELD BACK block of ``.gitignore`` and the notebook embeds them with ``held_back=True``.
"""

import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import figure
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from style import apply

from cypmode.challenge import pocket_pharmacophore as pp
from cypmode.validation import official_pharmacophore as official
from cypmode.validation import pose_pharmacophore as pose
from gen.molecules import draw
from gen.pharmacophore_atlas import (
    blank,
    draw_groups,
    family_legend,
    feature_change,
    pharmacophore_scene,
)

GROUP = "structure"
ROOT = Path(__file__).resolve().parents[3]
OUT = "data/structure_track/outputs/cyp3a4"
OFFICIAL = f"{OUT}/pharmacophore_official20"
SERIES = "data/structure_track/inputs/structure_benchmark_series.csv"
SERIES_RESULTS = f"{OUT}/series_results.csv"
CHEMOTYPE_SIMILARITY = 0.45
SCENE_SIZE = (1200, 900)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


# -- poses of the benchmark in 3D with the sites of the model ------------------------------------------------------------------------------------------------

def chosen_ligands() -> list[str]:
    """The rule declared on 2026-10-23: the benchmark ligands with the smallest, the median and the largest first-sample RMSD of the unguided Boltz-2 runs."""
    table = _table(f"{OUT}/benchmark_engine_pose_validation_unguided_boltz.csv")
    first = table[table["sample"] == 0].sort_values(["true_rmsd", "compound"]).reset_index(drop=True)
    return [first.compound[0], first.compound[len(first) // 2], first.compound[len(first) - 1]]


def crystal_of(smiles: str) -> dict | None:
    """The first crystal copy of the compound in the reference set whose entry file is available."""
    from cypmode.validation.pose_rescoring import connectivity_key

    key = connectivity_key(smiles)
    for record in pose.references():
        if connectivity_key(record["smiles"]) == key and (ROOT / pp.PDB_CACHE / f"{record['pdb_id']}.cif").exists():
            return record
    return None


def sites_in_complex(in_frame: dict, sites: dict[str, np.ndarray]) -> tuple[list[tuple], pd.DataFrame]:
    """The sites of a model as spheres in the coordinates of a predicted complex (family, x, y, z, radius, transparency; the sites that a feature of the pose matches are drawn firmer than those that it does not),
    and the table of the sites with their matches."""
    table = official.matched_sites(in_frame["features"], sites)
    spheres = []
    for row in table.itertuples():
        x, y, z = in_frame["iron"] + in_frame["frame"].T @ np.array([row.x, row.y, row.z])
        spheres.append((row.family, float(x), float(y), float(z), pp.SITE_RADIUS, 0.4 if row.matched else 0.9))
    return spheres, table


def solid_features(in_frame: dict) -> list[tuple[str, float, float, float]]:
    return [(family, float(p[0]), float(p[1]), float(p[2])) for family, positions in in_frame["features_raw"].items() for p in positions]


def pose_scene_for(ligand: str, smiles: str, cif: Path, theme_name: str, key: str, crystal: dict | None, sites_of: tuple[int, ...] | None = None, size=SCENE_SIZE) -> tuple[object, dict, pd.DataFrame]:
    """(the scene, the score of the pose, the sites with their matches) of one predicted pose with its features, the sites rebuilt without the references of its ligand and, when there is one, the crystal pose."""
    from cypmode.validation.structures import load_structure

    structure = official.load_pdb(Path(cif).read_text()) if str(cif).endswith(".pdb") else load_structure(cif)
    in_frame = pose.pose_in_frame(structure, smiles)
    fields, sites = pose.fields_and_sites(sites_of if sites_of is not None else pose.kept_indices(smiles, "novel"))
    spheres, table = sites_in_complex(in_frame, sites)
    score = pose.score_pose(in_frame["heavy"], in_frame["features"], fields, sites)
    reference = None if crystal is None else (ROOT / pp.PDB_CACHE / f"{crystal['pdb_id']}.cif", crystal["comp_id"], crystal["fe"])
    scene = pharmacophore_scene(ROOT / cif if not Path(cif).is_absolute() else cif, None, None, theme_name, key, spheres, solid_features(in_frame), size, crystal=reference, zoom=6)
    return scene, score, table


def pose_3d_caption() -> str:
    ligands = chosen_ligands()
    return ("Why the pharmacophore model of a pose does not rank poses. Real structures rendered with PyMOL: the predicted complex (teal ligand, orange heme), the crystal pose of the same ligand (white, the crystal entry "
            "aligned on the protein) and the sites of the structure-based model (translucent spheres, one colour per feature family) built without the references of that ligand and its analogues; a site that a feature "
            "of the pose matches is drawn firmer, one that it does not is nearly transparent, and the small solid spheres are the features of the pose. For the benchmark ligands with the smallest, the median and the "
            f"largest first-sample RMSD of the unguided runs ({', '.join(ligands)}), the best (top row) and the worst (bottom row) pose of the pool. The model sees where features lie, not whether the pose is the crystal "
            "one: a pose a few ångströms away from the crystal pose can match as many sites as the true one, which is what the gate of the pharmacophore model measured.")


@figure("struct_pharm_pose_3d", GROUP, "The pharmacophore model on real poses: the best and the worst pose of three benchmark ligands", pose_3d_caption,
        data=(f"{OUT}/benchmark_engine_pose_validation_unguided_boltz.csv", "data/tdi_track/outputs/caches/pocket_fields_cyp3a4.npz"), section="9.15")
def pose_3d(theme_name):
    th = apply(theme_name)
    from gen.pymol_scene import autocrop

    smiles = pose.benchmark_smiles()
    pooled = pose.pooled_poses(ROOT / OUT)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.6), constrained_layout=True)
    for column, ligand in enumerate(chosen_ligands()):
        poses = sorted((p for p in pooled if p[0] == ligand), key=lambda p: p[3])
        crystal = crystal_of(smiles[ligand])
        for row, (label, chosen) in enumerate((("best", poses[0]), ("worst", poses[-1]))):
            _, run, sample, rmsd, cif = chosen
            scene, score, table = pose_scene_for(ligand, smiles[ligand], cif, theme_name, f"pose3d_{run}_{sample}", crystal)
            ax = axes[row][column]
            ax.imshow(autocrop(plt.imread(scene.png)))
            blank(ax)
            ax.set_title(f"{ligand}, {label} pose of the pool: RMSD {rmsd:.1f} Å\nrecall {score['recall']:.2f} ({int(table.matched.sum())} of {len(table)} sites), field score {score['field']:.2f}", fontsize=9)
    family_legend(axes[1][1], th, [f for f in pp.FAMILIES if f != "NegIonizable"], fontsize=8)
    return fig


# -- the SAR of a change of linker in the series benchmark ------------------------------------------------------------------------------------------------------

def series_pairs() -> list[tuple[pd.Series, pd.Series]]:
    table = _table(SERIES)
    return [(group.iloc[0], group.iloc[1]) for _, group in table.groupby("series", sort=True) if len(group) == 2]


def changed_atoms(mol_a, mol_b) -> tuple[list[int], list[int]]:
    """The atoms of each compound that the pair changes: the atoms outside their maximum common substructure (found with any atom allowed to match any atom, complete rings only, so that a thioether that
    becomes an amine is one atom that changes and not half of the molecule) and the matched atoms whose element differs."""
    result = rdFMCS.FindMCS([mol_a, mol_b], timeout=10, atomCompare=rdFMCS.AtomCompare.CompareAny, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, completeRingsOnly=True)
    query = Chem.MolFromSmarts(result.smartsString)
    match_a, match_b = mol_a.GetSubstructMatch(query), mol_b.GetSubstructMatch(query)
    different = {i for i, (a, b) in enumerate(zip(match_a, match_b, strict=False)) if mol_a.GetAtomWithIdx(a).GetAtomicNum() != mol_b.GetAtomWithIdx(b).GetAtomicNum()}
    keep_a = {a for i, a in enumerate(match_a) if i not in different}
    keep_b = {b for i, b in enumerate(match_b) if i not in different}
    return [a for a in range(mol_a.GetNumAtoms()) if a not in keep_a], [b for b in range(mol_b.GetNumAtoms()) if b not in keep_b]


def series_caption() -> str:
    return ("Structure-activity relationship in the structure track: the three analogue pairs of the series benchmark, each pair differing by one change of linker (a thioether, an amine, a sulfonamide in place of an amide or "
            "a carbamate). For each ligand the 2D structure with the atoms outside the maximum common substructure of the pair highlighted, and the first predicted pose in 3D (teal) against the crystal pose (white) "
            "with the sites of the structure-based model built without the references of both members of the pair (a site that a feature of the pose matches is drawn firmer). Each title gives the RMSD of the pose "
            "to the crystal pose, the distance of the pyridine nitrogen to the iron and the recall of the sites; between the two structures of a pair the label gives the change in the number of pharmacophore features. "
            "The pyridine nitrogen binds the iron in all six poses, so the series shares its Type II pharmacophore and differs in what the linker puts on the way out of the pocket.")


@figure("struct_pharm_series", GROUP, "Structure-activity relationship in the series benchmark: three changes of linker, in 2D and in 3D", series_caption,
        data=(SERIES, SERIES_RESULTS, f"{OUT}/series_pose_validation.csv"), section="9.15")
def series_sar(theme_name):
    th = apply(theme_name)
    from cypmode.validation.pose_pools import pool_paths
    from gen.pymol_scene import autocrop

    results = _table(SERIES_RESULTS).set_index("compound")
    truth = _table(f"{OUT}/series_pose_validation.csv")
    pairs = series_pairs()
    fig, axes = plt.subplots(len(pairs), 4, figsize=(16.0, 4.6 * len(pairs)), constrained_layout=True)
    for row, (first, second) in enumerate(pairs):
        mols = [Chem.MolFromSmiles(first.SMILES), Chem.MolFromSmiles(second.SMILES)]
        changed = changed_atoms(*mols)
        kept = tuple(sorted(set(pose.kept_indices(first.SMILES, "novel")) & set(pose.kept_indices(second.SMILES, "novel"))))
        for side, (member, mol) in enumerate(zip((first, second), mols, strict=True)):
            ax2d, ax3d = axes[row][2 * side], axes[row][2 * side + 1]
            ax2d.imshow(draw(mol, theme_name, atoms=changed[side], size=(520, 400), line_width=1.8))
            blank(ax2d)
            ax2d.set_title(f"{member.Molecule_Name} (series {member.series})", fontsize=9, loc="left")
            if side == 1:
                ax2d.set_xlabel(f"from {first.Molecule_Name}: {feature_change(mols[0], mol)}", fontsize=8)
            cif = next(path for engine, sample, path in pool_paths(member.Molecule_Name, ("boltz",)) if sample == 0)
            crystal = crystal_of(member.SMILES)
            scene, score, _ = pose_scene_for(member.Molecule_Name, member.SMILES, cif, theme_name, f"series3d_{member.Molecule_Name}", crystal, sites_of=kept)
            rmsd = float(truth[(truth.compound == member.Molecule_Name) & (truth["sample"] == 0)].true_rmsd.iloc[0])
            distance = results.loc[member.Molecule_Name, "closest_motif_nitrogen_distance_angstrom"]
            ax3d.imshow(autocrop(plt.imread(scene.png)))
            blank(ax3d)
            ax3d.set_title(f"RMSD {rmsd:.1f} Å, Fe-N {distance:.2f} Å, recall {score['recall']:.2f}", fontsize=9)
    family_legend(axes[-1][1], th, [f for f in pp.FAMILIES if f != "NegIonizable"], fontsize=8)
    return fig


# -- the 20 official ligands (held back) ---------------------------------------------------------------------------------------------------------------------

def chemotypes(smiles: dict[str, str]) -> list[list[str]]:
    """The ligands grouped into chemotypes: the connected groups of ligands whose Morgan Tanimoto similarity to a neighbour is at least ``CHEMOTYPE_SIMILARITY``, the largest group first, singletons last."""
    names = list(smiles)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles[n]), 2, nBits=2048) for n in names]
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]) >= CHEMOTYPE_SIMILARITY:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def common_core(smiles: list[str]):
    """The maximum common substructure of a group of compounds as a query molecule (None for one compound or when it has fewer than 6 atoms)."""
    if len(smiles) < 2:
        return None
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    result = rdFMCS.FindMCS(mols, timeout=10, ringMatchesRingOnly=True, completeRingsOnly=True)
    return Chem.MolFromSmarts(result.smartsString) if result.numAtoms >= 6 else None


def official_2d_caption() -> str:
    summary = _table(f"{OFFICIAL}/summary.csv")
    smiles = dict(_table("data/structure_track/inputs/structure_track_test_blinded.csv").values)
    groups = chemotypes(smiles)
    return (f"The {len(summary)} official ligands in 2D, grouped into {len([g for g in groups if len(g) > 1])} chemotypes and {len([g for g in groups if len(g) == 1])} singletons (Morgan Tanimoto similarity of at least "
            f"{CHEMOTYPE_SIMILARITY} to a neighbour; the largest group first). Each structure shows its functional groups (one colour per group of the fixed table, named under it) and, for a chemotype, the atoms of the "
            "maximum common substructure of the group are outlined in the colour of the group. Under each, the pharmacophore numbers of its final pose from job 925_official_pharmacophore: the iron-donor error of a Type II donor or the distance of the "
            "oxidation site outside the 3.5 to 6.0 Å window, the recall of the crystal sites and a mark when a flag of the check of the binding-site representation is raised. These are predictions for blinded ligands.")


@figure("struct_pharm_official20_2d", GROUP, "The 20 official ligands: chemotypes, functional groups and the pharmacophore of their final poses", official_2d_caption,
        data=(f"{OFFICIAL}/summary.csv", "data/structure_track/inputs/structure_track_test_blinded.csv"), section="9.15")
def official_2d(theme_name):
    th = apply(theme_name)
    summary = _table(f"{OFFICIAL}/summary.csv").set_index("ligand")
    smiles = dict(_table("data/structure_track/inputs/structure_track_test_blinded.csv").values)
    groups = chemotypes(smiles)
    ordered = [name for group in groups for name in group]
    columns = 5
    rows = -(-len(ordered) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.9 * columns, 4.2 * rows), constrained_layout=True)
    border = [th[k] for k in ("blue", "red", "green", "orange", "purple", "teal", "gold")]
    colour_of = {name: border[g % len(border)] if len(group) > 1 else th["muted"] for g, group in enumerate(groups) for name in group}
    for ax in np.ravel(axes):
        blank(ax)
    for ax, name in zip(np.ravel(axes), ordered, strict=False):
        mol = Chem.MolFromSmiles(smiles[name])
        image, legend = draw_groups(mol, theme_name, size=(460, 360))
        ax.imshow(image)
        group = next(g for g in groups if name in g)
        core = common_core([smiles[n] for n in group])
        if core is not None:
            match = mol.GetSubstructMatch(core)
            if match:
                ax.text(0.02, 0.02, f"core of {len(group)}: {core.GetNumAtoms()} atoms", transform=ax.transAxes, fontsize=7, color=colour_of[name])
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(colour_of[name])
            side.set_linewidth(1.6)
        row = summary.loc[name]
        donor = f"donor error {row.donor_error:.2f} Å" if pd.notna(row.donor_error) else f"site {row.site_error:.1f} Å outside the window"
        flag = "  [flag]" if (bool(row.beyond_pocket_flag) or bool(row.beyond_support_cut)) else ""
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_xlabel(f"{donor}; recall {row.recall:.2f}{flag}\n" + "; ".join(n for n, _ in legend), fontsize=6.5)
    return fig


def official_3d_caption() -> str:
    return ("The 20 official ligands in their pockets, with the pharmacophore model. For each ligand a PyMOL render of its final submitted pose (teal) in the pocket of the predicted complex (orange heme, the lining residues), "
            "with the sites of the structure-based model, built without the references of the ligand and of its analogues, as translucent spheres in the colour of the feature family (a site that a feature of the pose "
            "matches is drawn firmer) and the features of the pose as small solid spheres. The panels are in the order of the chemotypes of the previous figure. These are predictions for blinded ligands.")


def final_pdb_paths(directory: Path) -> dict[str, Path]:
    """The final submitted structures, written out of the archive into ``directory`` so that PyMOL can read them."""
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    with zipfile.ZipFile(ROOT / OUT / "final/structures_final.zip") as archive:
        for member in archive.namelist():
            if member.endswith(".pdb"):
                path = directory / member
                path.write_bytes(archive.read(member))
                out[Path(member).stem] = path
    return out


@figure("struct_pharm_official20_3d", GROUP, "The 20 official ligands in their pockets with the pharmacophore model", official_3d_caption,
        data=(f"{OFFICIAL}/summary.csv", f"{OUT}/final/structures_final.zip", "data/tdi_track/outputs/caches/pocket_fields_cyp3a4.npz"), section="9.15")
def official_3d(theme_name):
    th = apply(theme_name)
    from gen.pymol_scene import CACHE, autocrop

    smiles = dict(_table("data/structure_track/inputs/structure_track_test_blinded.csv").values)
    ordered = [name for group in chemotypes(smiles) for name in group]
    paths = final_pdb_paths(CACHE / "official_final")
    columns = 5
    rows = -(-len(ordered) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.9 * columns, 3.4 * rows), constrained_layout=True)
    for ax in np.ravel(axes):
        blank(ax)
    for ax, name in zip(np.ravel(axes), ordered, strict=False):
        scene, score, table = pose_scene_for(name, smiles[name], paths[name], theme_name, f"official3d_{name}", None, size=(900, 700))
        ax.imshow(autocrop(plt.imread(scene.png)))
        ax.set_title(f"{name}: recall {score['recall']:.2f} ({int(table.matched.sum())} of {len(table)} sites)", fontsize=8, loc="left")
    family_legend(np.ravel(axes)[-1], th, [f for f in pp.FAMILIES if f != "NegIonizable"], fontsize=7)
    return fig
