"""The confirmatory restraint gates of ``restraint_expansion`` repeated with OpenStructure's LDDT-PLI.

``ost_official`` registered that, if our LDDT-PLI did not agree with the organisers' scorer, the gates that rest on it are repeated with OpenStructure's value. This module repeats the two
gates of the registered confirmation (``restraint_bond_expansion`` and ``restraint_bond_harm``, primary stratum: the paired Wilcoxon on the per-ligand difference between the restrained and
the unguided first samples, robust to dropping a ligand) on the poses that OpenStructure scored, and prints them next to the values of our measure.

``python -m cypmode.validation.ost_regate`` reads ``restraint_expansion_poses.csv`` and ``ost_vs_ours.csv``. The verdict lines carry the suffix ``_ost``.
"""

from __future__ import annotations

import sys

import pandas as pd

from cypmode.validation import restraint_expansion as expansion

KEYS = ["compound", "engine", "sample"]


def regate(poses: pd.DataFrame, ost: pd.DataFrame, ligands: pd.DataFrame, samples: int, alpha: float, stratum: str = "primary") -> dict[str, dict]:
    """The paired statistics of the restraint effect on the ligands of a stratum, once with our LDDT-PLI and once with OpenStructure's, over the poses that both measures scored."""
    joined = poses.merge(ost[[*KEYS, "ost"]], on=KEYS, how="inner").dropna(subset=["lddt_pli", "ost"])
    members = {f"exp-{c}" for c, s in zip(ligands["comp_id"], ligands["stratum"], strict=True) if s == stratum}
    part = joined[joined["compound"].str.removesuffix(expansion.RESTRAINED_SUFFIX).isin(members)]
    out = {}
    for label, column in (("ours", "lddt_pli"), ("openstructure", "ost")):
        per = expansion.per_ligand(part.rename(columns={column: "value"}), "value", samples)
        out[label] = expansion.paired_statistics(per["difference"].to_numpy(), alpha)
    return out


def main() -> int:
    plan = expansion.load_plan()
    alpha, samples = plan["endpoints"]["alpha"], plan["design"]["samples_per_condition"]
    poses = pd.read_csv(expansion.OUT / "restraint_expansion_poses.csv")
    ost = pd.read_csv(expansion.OUT / "ost_vs_ours.csv")
    ligands = pd.read_csv(expansion.INPUTS / "restraint_expansion_ligands.csv").assign(comp_id=lambda t: t["comp_id"].astype(str))
    stats = regate(poses, ost, ligands, samples, alpha)
    for label, s in stats.items():
        print(expansion.verdict_line(f"LDDT-PLI ({label}), restrained minus unguided, primary stratum", s))
    ost_stats = stats["openstructure"]
    print(f"STRUCTURE GATE restraint_bond_expansion_ost (the registered confirmation repeated with OpenStructure's LDDT-PLI, one-sided Wilcoxon p < {alpha}, robust to dropping a ligand): "
          f"{'MET' if ost_stats['confirmed'] else 'NOT MET'}")
    print(f"STRUCTURE GATE restraint_bond_harm_ost (the harm check repeated with OpenStructure's LDDT-PLI, one-sided Wilcoxon that the restraint is worse p < {alpha}): "
          f"{'MET' if ost_stats['harm'] else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
