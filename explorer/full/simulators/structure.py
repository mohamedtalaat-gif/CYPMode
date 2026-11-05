# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part XI, the logic: reading a complex file (11.1), noise on real coordinates (11.2), the input of a prediction (11.3), an RMSD with and without alignment (11.4), the success of the stored poses (11.5), the memory of a pair tensor (11.9)
and the stereoisomers of a ligand (11.12). The viewer is 3Dmol.js, the copy that the project vendors."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

HERE = Path(__file__).resolve().parents[1]
STRUCTURES = HERE / "data/structures"


def structure_names() -> list[str]:
    return sorted(p.stem for p in STRUCTURES.glob("*.pdb"))


def pdb_text(name: str) -> str:
    return (STRUCTURES / f"{name}.pdb").read_text()


def parse(text: str) -> pd.DataFrame:
    rows = [{"record": line[:6].strip(), "name": line[12:16].strip(), "resname": line[17:20].strip(), "chain": line[21], "resseq": int(line[22:26]), "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]), "element": line[76:78].strip().upper()}
            for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    return pd.DataFrame(rows)


def residue_table(atoms: pd.DataFrame) -> pd.DataFrame:
    """What the file holds: the protein chain with its residues and atoms, and every other molecule (the heme, the ligand) with its own."""
    protein = atoms[atoms["record"] == "ATOM"].groupby("chain").agg(residues=("resseq", "nunique"), atoms=("name", "size")).reset_index().assign(kind="protein")
    other = atoms[atoms["record"] == "HETATM"].groupby(["chain", "resname"]).agg(residues=("resseq", "nunique"), atoms=("name", "size")).reset_index().rename(columns={"resname": "kind"})
    return pd.concat([protein, other], ignore_index=True)[["chain", "kind", "residues", "atoms"]]


def iron_nitrogen(atoms: pd.DataFrame, ligand: str = "LIG") -> dict | None:
    iron = atoms[(atoms["element"] == "FE")]
    nitrogens = atoms[(atoms["resname"] == ligand) & (atoms["element"] == "N")]
    if iron.empty or nitrogens.empty:
        return None
    fe = iron.iloc[0][["x", "y", "z"]].to_numpy(dtype=float)
    distances = np.linalg.norm(nitrogens[["x", "y", "z"]].to_numpy(dtype=float) - fe, axis=1)
    k = int(distances.argmin())
    return {"iron": fe.tolist(), "nitrogen": nitrogens.iloc[k][["x", "y", "z"]].to_numpy(dtype=float).tolist(), "distance": float(distances[k]), "atom": nitrogens.iloc[k]["name"]}


def ligand_xyz(atoms: pd.DataFrame, ligand: str = "LIG") -> np.ndarray:
    return atoms[(atoms["resname"] == ligand) & (atoms["element"] != "H")][["x", "y", "z"]].to_numpy(dtype=float)


def viewer_html(text: str, protein: bool = True, heme: bool = True, ligand: str = "LIG", show_distance: bool = True, height: int = 440) -> str:
    """A complete page that shows the complex with 3Dmol.js: the protein as thin lines, the heme, the ligand as thick sticks, and the dashed line from the iron to the nearest nitrogen of the ligand."""
    script = (HERE / "vendor/3Dmol-min.js").read_text()
    contact = iron_nitrogen(parse(text), ligand) if show_distance else None
    extra = ""
    if contact:
        mid = {k: (a + b) / 2 for k, a, b in zip("xyz", contact["iron"], contact["nitrogen"], strict=True)}
        extra = (f"v.addLine({{dashed:true,color:'red',start:{json.dumps(dict(zip('xyz', contact['iron'], strict=True)))},end:{json.dumps(dict(zip('xyz', contact['nitrogen'], strict=True)))}}});"
                 f"v.addLabel('{contact['distance']:.2f} A',{{position:{json.dumps(mid)},backgroundColor:'white',fontColor:'red',fontSize:13,backgroundOpacity:0.8}});")
    return ("<html><body style='margin:0'><div id='v' style='width:100%;height:" + str(height) + "px;position:relative'></div><script>" + script + "</script><script>"
            "const v=$3Dmol.createViewer('v',{backgroundColor:'white'});v.addModel(" + json.dumps(text) + ",'pdb');v.setStyle({},{});"
            + ("v.setStyle({hetflag:false},{line:{color:'#b0bec5'}});" if protein else "")
            + ("v.setStyle({resn:'HEM'},{stick:{radius:0.15,colorscheme:'orangeCarbon'}});v.addStyle({elem:'Fe'},{sphere:{radius:0.7,color:'#e65100'}});" if heme else "")
            + f"v.setStyle({{resn:'{ligand}'}},{{stick:{{radius:0.25,colorscheme:'greenCarbon'}}}});" + extra + f"v.zoomTo({{resn:'{ligand}'}});v.render();</script></body></html>")


def noisy(atoms_xyz: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    return atoms_xyz + sigma * np.random.default_rng(seed).normal(size=atoms_xyz.shape)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def rmsd_demo(smiles: str, shift: float, angle: float) -> dict | None:
    """Two conformers of one molecule, the second moved as a rigid body by ``shift`` angstrom and ``angle`` degrees: the RMSD as it stands, after the optimal alignment and with the best matching of symmetric atoms."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMultipleConfs(mol, 2, randomSeed=7) != [0, 1] and mol.GetNumConformers() < 2:
        return None
    heavy = Chem.RemoveHs(mol)
    first = heavy.GetConformer(0).GetPositions()
    second = heavy.GetConformer(1).GetPositions()
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    centre = second.mean(axis=0)
    moved = (second - centre) @ rotation.T + centre + np.array([shift, 0.0, 0.0])
    probe = Chem.Mol(heavy, confId=1)
    conf = probe.GetConformer(1)
    for i, p in enumerate(moved):
        conf.SetAtomPosition(i, p.tolist())
    other = Chem.Mol(probe)
    aligned = rdMolAlign.AlignMol(probe, heavy, prbCid=1, refCid=0)
    best = rdMolAlign.GetBestRMS(Chem.Mol(other), heavy, 1, 0)
    return {"in place": rmsd(first, moved), "of the conformers themselves, in place": rmsd(first, second), "after the best rigid alignment": float(aligned), "after alignment with the best match of symmetric atoms": float(best)}


def success(poses: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per engine, the share of ligands whose first sample, and whose best of all samples, is within ``threshold`` angstrom of the crystal pose."""
    rows = []
    for engine, group in poses.groupby("engine"):
        first = group[group["sample"] == group["sample"].min()].groupby("compound")["true_rmsd"].first()
        best = group.groupby("compound")["true_rmsd"].min()
        rows.append({"engine": engine, "ligands": len(best), "first sample": float((first <= threshold).mean()), "best of the samples": float((best <= threshold).mean())})
    return pd.DataFrame(rows)


def pair_tensor_memory(tokens: int, channels: int, bytes_per_number: int, copies: int, heads: int) -> dict:
    """An order-of-magnitude estimate of the memory of the pair representation of a co-folding network: copies of N x N x C numbers, and the N x N x N x heads logits of one triangle attention."""
    pair = tokens**2 * channels * bytes_per_number * copies / 1e9
    attention = tokens**3 * heads * bytes_per_number / 1e9
    return {"pair tensors (GB)": pair, "logits of one triangle attention (GB)": attention, "total (GB)": pair + attention}


def stereoisomers(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return {"centres": centres, "unassigned": [c for c in centres if c[1] == "?"], "isomers": [Chem.MolToSmiles(m) for m in EnumerateStereoisomers(mol)]}
