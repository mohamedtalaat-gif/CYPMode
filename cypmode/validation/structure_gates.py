# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Machine-readable verdicts for the restraint and the pose-diversity gates (job 165).

The restraint and pilot runs already produce pose tables (job 140); this module applies the gate texts of the checklist
literally and prints one ``STRUCTURE GATE <name> ...: MET|NOT MET`` line per gate so ``final_recipe`` can read them:

  restraint_bond / restraint_contact   mean first-pick RMSD of the restrained runs is lower than the unguided runs on the
        same ligands (3 Boltz samples each) AND the share of first picks that pass PoseBusters is not lower; the paired
        drop-one statistics are printed for information (6 ligands cannot carry a significance test);
  pilot_seeds / pilot_conformers / pilot_stepscale / pilot_cryoem   the lever widens the pool (spread >= 1.5 x the unguided pool's in at least
        half of the ligands) AND reaches the right pose (best pose in the pool at least 0.75 A closer to the crystal pose
        than the unguided pool's best in at least half of the ligands), the thresholds of ``diversity_pilot``.

Usage: python -m cypmode.validation.structure_gates
Exit status 75 when an input table does not exist yet (the queue then retries later).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/structure_track/outputs/cyp3a4")
IN = Path("data/structure_track/inputs")
TABLES = {
    "unguided": OUT / "benchmark_engine_pose_validation_unguided_boltz.csv",
    "febond": OUT / "benchmark_engine_pose_validation_febond.csv",
    "fecontact": OUT / "benchmark_engine_pose_validation_fecontact.csv",
}
PILOT_POSES = OUT / "diversity_pilot_poses.csv"
CRYOEM_POSES = OUT / "diversity_pilot_poses_cryoem.csv"  # cryo-EM template runs (cryoem_templates.py), same pilot ligands


def restraint_gates(unguided: pd.DataFrame, variants: dict[str, pd.DataFrame], validity) -> dict[str, dict]:
    """``validity(compound, variant) -> bool`` says whether that run's first pick passes PoseBusters."""
    from cypmode.validation.selection_validation import paired_gate
    from cypmode.validation.structure_benchmark_report import variant_comparison

    out = {}
    for name, table in variants.items():
        comparison = variant_comparison({"unguided": unguided, name: table}, samples=3)
        wide = comparison.pivot(index="compound", columns="variant", values="first_pick")
        if name not in wide or wide[name].notna().sum() < 2:
            continue
        shared = wide.dropna(subset=["unguided", name])
        pass_unguided = float(np.mean([validity(c, "unguided") for c in shared.index]))
        pass_variant = float(np.mean([validity(c, name) for c in shared.index]))
        stats = paired_gate(shared["unguided"], shared[name])
        better = float(shared[name].mean()) < float(shared["unguided"].mean())
        out[f"restraint_{name.removeprefix('fe')}"] = {
            "ligands": len(shared), "mean_unguided": float(shared["unguided"].mean()), "mean_variant": float(shared[name].mean()),
            "valid_unguided": pass_unguided, "valid_variant": pass_variant, "paired": stats, "met": bool(better and pass_variant >= pass_unguided),
        }
    return out


def pilot_gates(pools: pd.DataFrame) -> dict[str, dict]:
    from cypmode.validation.diversity_pilot import verdicts

    table = verdicts(pools)
    return {f"pilot_{r.lever}": {"ligands": int(r.ligands), "wider": float(r.ligands_with_wider_pool), "better_best": float(r.ligands_with_better_best_pose),
                                 "met": bool(r.creates_diversity and r.reaches_right_pose)} for r in table.itertuples()}


def _validity_fn(smiles: dict[str, str]):
    from cypmode.validation.pose_consensus import boltz_cif_paths
    from cypmode.validation.pose_validity import check_pose_validity
    from cypmode.validation.structures import load_structure

    def check(compound: str, variant: str) -> bool:
        name = compound if variant == "unguided" else f"{compound}-{variant}"
        paths = boltz_cif_paths(name)
        if not paths:
            return False
        try:
            return bool(check_pose_validity(load_structure(paths[0]), smiles[compound])["all_passed"])
        except Exception:
            return False

    return check


def main() -> None:
    from cypmode.validation.diversity_pilot import _report

    if not TABLES["unguided"].exists():
        print("unguided pose table missing; run job 140 first")
        sys.exit(75)
    unguided = pd.read_csv(TABLES["unguided"])
    smiles = dict(zip(*[pd.read_csv(IN / "structure_benchmark.csv")[c] for c in ("Molecule_Name", "SMILES")], strict=True))
    results: dict[str, dict] = {}
    variants = {n: pd.read_csv(p) for n, p in TABLES.items() if n != "unguided" and p.exists()}
    if variants:
        results.update(restraint_gates(unguided, variants, _validity_fn(smiles)))
    pilot_tables = [pd.read_csv(path) for path in (PILOT_POSES, CRYOEM_POSES) if path.exists()]
    if pilot_tables:
        pools = _report(pd.concat(pilot_tables, ignore_index=True), smiles, unguided)
        if len(pools):
            print(pools.round(2).to_string(index=False))
            results.update(pilot_gates(pools))
    if not results:
        print("no restraint or pilot table available yet")
        sys.exit(75)
    for name, r in results.items():
        detail = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("met", "paired")}
        print(f"  {name}: {detail}")
        if "paired" in r:
            p = r["paired"]
            print(f"    paired over {p['n']} ligands: mean improvement {p['mean_improvement']:+.3f} A, {p['wins']} wins, Wilcoxon p {p['p']:.3f}, worst drop-one {p['min_drop_one']:+.3f} (information only)")
        print(f"STRUCTURE GATE {name}: {'MET' if r['met'] else 'NOT MET'}")


if __name__ == "__main__":
    main()
