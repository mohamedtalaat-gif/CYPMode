"""PoseBusters physical/chemical validity checks on an already-predicted
Boltz-2 structure -- confirmed directly real and working on our own
output (unlike ProLIF, which repeatedly failed on the same kind of
structure: missing VDW radii, an explicit-hydrogen requirement Boltz-2's
heavy-atom-only output doesn't meet, a kekulization failure, and finally
a hard segfault -- see cypmode.validation.pocket_geometry's module
docstring for why ProLIF was declined).

This isn't just a nice-to-have QC step: OpenADMET's own Structure Track
(launched 2026-09-25, see CYPMode_Architecture_Specification.md's
"Official challenge update" section) made PoseBusters a mandatory,
scored part of the real competition -- any submitted structure that
fails these checks scores LDDT-PLI=0 regardless of how geometrically
accurate the pose is otherwise. Running this locally, before submission,
is how a real structure that would score 0 gets caught early.
"""

import tempfile
from pathlib import Path

from Bio.PDB import PDBIO, Select

from cypmode.validation.structures import _ligand_atoms, _ligand_mol_3d, load_structure


class _ExcludeLigand(Select):
    """A Bio.PDB.PDBIO Select that writes every residue except the
    ligand's own -- PoseBusters' mol_cond argument (the "condition", i.e.
    everything the ligand is checked against: protein, heme, waters)
    must not itself contain the ligand.
    """

    def __init__(self, ligand_atom_ids: set):
        self._ligand_atom_ids = ligand_atom_ids

    def accept_residue(self, residue) -> bool:
        return not any(id(atom) in self._ligand_atom_ids for atom in residue.get_atoms())


def check_pose_validity(structure, ligand_smiles: str, config: str = "dock") -> dict:
    """Runs PoseBusters on one predicted structure's ligand pose within
    its real protein/heme/cofactor context. config="dock" is PoseBusters'
    own preset for "a single predicted pose, no reference structure to
    compare against" -- exactly this project's situation (see
    cypmode.validation.pose_comparison's own module docstring for why a
    real reference isn't available locally yet).

    Always uses PoseBusters' summary (boolean-per-check) report, not its
    full numeric one: the full report mixes booleans with raw distances/
    atom IDs, and computing "did everything pass" over that mix would
    silently misinterpret a numeric column's truthiness as a pass/fail
    bit. Call posebusters.PoseBusters directly for the full numeric
    report; this function's job is only the pass/fail question.

    Returns a dict with `all_passed` (bool -- every individual check
    True) and `checks` (the full per-check True/False dict from
    PoseBusters itself), so a caller can see exactly which check failed,
    not just a single pass/fail bit.
    """
    from posebusters import PoseBusters

    ligand_mol, _ = _ligand_mol_3d(structure, ligand_smiles)
    ligand_atom_ids = {id(atom) for atom in _ligand_atoms(structure)}

    with tempfile.TemporaryDirectory() as tmp_dir:
        protein_path = Path(tmp_dir) / "protein.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(protein_path), _ExcludeLigand(ligand_atom_ids))

        pb = PoseBusters(config=config)
        result = pb.bust(ligand_mol, mol_cond=str(protein_path), full_report=False)

    checks = {k: bool(v) for k, v in result.iloc[0].to_dict().items() if k not in ("file", "molecule", "position")}
    return {"all_passed": all(checks.values()), "checks": checks}


def select_valid_model(predictions_dir: str | Path, ligand_smiles: str, max_models: int = 10) -> dict:
    """Real, concrete fix for a compound whose default model_0 fails
    PoseBusters: when a batch was run with --diffusion_samples > 1,
    Boltz-2 writes multiple ranked candidate structures to the same
    predictions directory (name_model_0.cif, name_model_1.cif, ...,
    confidence-ranked, 0 = best). Tries them in that ranked order and
    returns the first one that passes every PoseBusters check, rather
    than always trusting model_0 -- a real steric clash or geometry
    artifact in the top-confidence sample doesn't mean every sampled
    diffusion trajectory has the same problem.

    Returns {"model_index": int, "cif_path": Path, "result": dict} for
    the first passing model, or {"model_index": None, "cif_path": None,
    "result": None} if none of the available models (up to max_models)
    pass. Does not run boltz itself -- the caller is responsible for
    having generated model_0..model_N first (e.g. via --diffusion_samples
    N on the original batch, or a targeted re-run for compounds that
    failed with a single sample).
    """
    predictions_dir = Path(predictions_dir)
    stem = predictions_dir.name  # e.g. "cyp3a4_itraconazole"

    for i in range(max_models):
        cif_path = predictions_dir / f"{stem}_model_{i}.cif"
        if not cif_path.exists():
            break
        structure = load_structure(cif_path)
        try:
            result = check_pose_validity(structure, ligand_smiles)
        except Exception:
            continue
        if result["all_passed"]:
            return {"model_index": i, "cif_path": cif_path, "result": result}

    return {"model_index": None, "cif_path": None, "result": None}


def _main() -> None:
    """Runs check_pose_validity across every real, already-completed
    structure this project has (data/structure_track/outputs/out/boltz_results_*/),
    reading each compound's own exact ligand SMILES from its own input
    YAML -- guaranteed to match what was actually predicted, rather than
    cross-referencing separate CSV/dict sources that could drift out of
    sync with the real inputs.

    --target names the protein these structures are against (used only
    in file/directory naming) -- defaults to "cyp3a4" to match every
    existing structure this project has, but was a hardcoded literal
    until now, which would have silently found zero CYP2D6 structures
    rather than a real, distinct set of them.
    """
    import argparse

    import pandas as pd
    import yaml

    from cypmode.validation.structures import load_structure

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--target", default="cyp3a4", help="protein name used in file/directory naming")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="defaults to data/structure_track/outputs/posebusters_validity.csv -- pass a "
        "different path for a non-cyp3a4 target so it doesn't overwrite the real CYP3A4 one",
    )
    args = parser.parse_args()
    target = args.target

    input_dir = Path("data/structure_track/inputs")
    out_dir = Path("data/structure_track/outputs/out")

    rows = []
    for yaml_path in sorted(input_dir.glob(f"{target}_*.yaml")):
        name = yaml_path.stem.removeprefix(f"{target}_")
        data = yaml.safe_load(yaml_path.read_text())
        ligand_smiles = next(
            seq["ligand"]["smiles"] for seq in data["sequences"] if "ligand" in seq and "smiles" in seq["ligand"]
        )

        cif_path = out_dir / f"boltz_results_{target}_{name}" / "predictions" / f"{target}_{name}" / f"{target}_{name}_model_0.cif"
        if not cif_path.exists():
            continue

        try:
            structure = load_structure(cif_path)
            result = check_pose_validity(structure, ligand_smiles)
            rows.append({"compound": name, "all_passed": result["all_passed"], **result["checks"]})
        except Exception as exc:
            rows.append({"compound": name, "all_passed": False, "error": str(exc)})

    df = pd.DataFrame(rows)
    n_passed = df["all_passed"].sum()
    print(f"{n_passed}/{len(df)} real completed structures pass every PoseBusters check")

    failed = df[~df["all_passed"]]
    if not failed.empty:
        print("\nfailed compounds and their failing checks:")
        for _, row in failed.iterrows():
            if "error" in row and pd.notna(row.get("error")):
                print(f"  {row['compound']}: ERROR -- {row['error']}")
                continue
            failing_checks = [c for c in df.columns if c not in ("compound", "all_passed", "error") and row.get(c) is False]
            print(f"  {row['compound']}: failed {failing_checks}")

    out_path = args.out_path or Path("data/structure_track/outputs/posebusters_validity.csv")
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    _main()
