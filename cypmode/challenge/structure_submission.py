# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Builds the real Structure Track submission zip.

The official submission format (verified directly against the live HF
Space's submission.py, 2026-09-26 -- see submissions/structure_track/README.md):
a .zip containing exactly 20 .pdb files, one per official test compound,
named "{Molecule_Name}.pdb", each a full protein-ligand complex with the
ligand residue named LIG.

What existed before this module: real Boltz-2 predictions as .cif, with
the ligand residue named LIG1 (Boltz-2's own convention, 4 characters) --
not submission-formatted. This module does the real conversion: CIF -> PDB
via Biopython (already a project dependency, see cypmode.validation.structures),
rename LIG1 -> LIG, name each file by Molecule_Name, zip all 20 together.
Also re-runs PoseBusters on exactly these 20 compounds first (not the
other ~55 unrelated structures also sitting under the same input/output
directories -- Tier 3 and the TDI-reference panel), since a submitted
structure that fails PoseBusters scores LDDT-PLI=0 (and, per the real
CYP-challenge backend code read directly 2026-09-27, LDDT-LP=0 and a 20A
BiSyRMSD penalty too -- all three metrics, not just LDDT-PLI as the HF
Space's own summary text implies) regardless of geometric accuracy.

Real Boltz-2 output is organized by target -- data/structure_track/outputs/out/cyp3a4/,
mirroring cypmode.validation.pipeline's own target-nested out_dir -- rather
than one flat directory mixing CYP3A4 and CYP2D6 results together.

Picks a pose via cypmode.validation.pose_validity.select_valid_model
rather than always trusting model_0: any compound re-run with
diffusion_samples>1 (see cypmode.validation.pipeline) gets its first
PoseBusters-passing sample used instead of whichever one the model
ranked first internally -- confirmed unreliable for pose selection on
P450s specifically (a real external finding: Boltz-2/OpenFold3 confidence
scores "offer no useful signal" for CYP targets). A compound with only a
single sample (model_0) behaves exactly as before -- select_valid_model
tries model_0 first and returns immediately if it already passes.
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ZIP_FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # the earliest time a zip entry can carry

import pandas as pd
from Bio.PDB import PDBIO

from cypmode.validation.pose_validity import select_valid_model
from cypmode.validation.structures import load_structure

TEST_SET_PATH = Path("data/structure_track/inputs/structure_track_test_blinded.csv")
OUT_DIR = Path("data/structure_track/outputs/out")
TARGET = "cyp3a4"
EXPECTED_COUNT = 20


def _predictions_dir(name: str, target: str = TARGET) -> Path:
    return OUT_DIR / target / f"boltz_results_{target}_{name}" / "predictions" / f"{target}_{name}"


def _cif_path(name: str, target: str = TARGET) -> Path:
    """The default (model_0) CIF path -- used only as a last-resort fallback
    when no sampled model passes PoseBusters at all (see
    revalidate_posebusters), so a compound with zero passing poses still
    ships its best-available structure rather than being silently dropped.
    """
    return _predictions_dir(name, target) / f"{target}_{name}_model_0.cif"


def _read_ligand_smiles_for(name: str, target: str = TARGET) -> str:
    import re

    yaml_path = Path("data/structure_track/inputs") / f"{target}_{name}.yaml"
    match = re.search(r"smiles:\s*'([^']*)'", yaml_path.read_text())
    if not match:
        raise ValueError(f"no smiles: '...' line found in {yaml_path}")
    return match.group(1)


def revalidate_posebusters(names: list[str], target: str = TARGET) -> pd.DataFrame:
    """Runs select_valid_model fresh on exactly these compounds -- not the
    other real structures (Tier 3, TDI-reference panel) sharing the same
    directories, whose validity is a separate, already-tracked question.
    Records which model index actually passed (or None, meaning no
    sampled model passed and the fallback model_0 will be used as-is).
    """
    rows = []
    for name in names:
        try:
            smiles = _read_ligand_smiles_for(name)
            selection = select_valid_model(_predictions_dir(name), smiles)
            if selection["model_index"] is not None:
                rows.append(
                    {
                        "compound": name,
                        "all_passed": True,
                        "model_index": selection["model_index"],
                        **selection["result"]["checks"],
                    }
                )
            else:
                # No sampled model passed -- fall back to model_0's own checks
                # so the report still shows exactly what's wrong with it.
                structure = load_structure(_cif_path(name))
                from cypmode.validation.pose_validity import check_pose_validity

                fallback = check_pose_validity(structure, smiles)
                rows.append({"compound": name, "all_passed": False, "model_index": None, **fallback["checks"]})
        except Exception as exc:
            rows.append({"compound": name, "all_passed": False, "model_index": None, "error": str(exc)})
    return pd.DataFrame(rows)


def convert_cif_to_pdb(cif_path: Path, out_pdb_path: Path) -> None:
    """Real CIF -> PDB conversion, with the ligand residue renamed to LIG
    so the output matches the official 3-character PDB field width the
    submission format requires. Boltz-2's CIF names the ligand LIG1 and
    OpenFold3's names it LIG0 -- both 4-character, both renamed here, so a
    pose from either engine converts identically.
    """
    structure = load_structure(cif_path)
    for residue in structure.get_residues():
        if residue.resname in ("LIG0", "LIG1"):
            residue.resname = "LIG"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb_path))


def write_deterministic_zip(pdb_paths: list[Path], zip_path: Path, names: list[str] | None = None) -> Path:
    """Zip the PDB files with a fixed timestamp and mode, so that the archive of the same structures has the same bytes (and SHA-256) whenever it is rebuilt.

    ``names`` gives the member names (default: the file names), for a member whose content comes from a file with another name.
    """
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for index, pdb_path in enumerate(pdb_paths):
            member = ZipInfo(names[index] if names else pdb_path.name, date_time=ZIP_FIXED_TIME)
            member.compress_type = ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            zf.writestr(member, pdb_path.read_bytes())
    return zip_path


def build_structure_submission(
    test_set_path: Path = TEST_SET_PATH,
    target: str = TARGET,
    out_dir: Path = Path("submissions/structure_track"),
    zip_name: str = "structure_track_submission.zip",
    skip_posebusters_failures: bool = False,
    consensus_csv: Path | None = None,
) -> dict:
    """End to end: load the real 20-compound official test set, re-run
    PoseBusters on exactly those, convert every one's real Boltz-2 CIF to a
    submission-formatted PDB, and zip them. Returns a summary dict rather
    than raising on a PoseBusters failure by default (skip_posebusters_failures=False
    means "include it anyway, but report it" -- excluding a compound instead
    of submitting a poor pose isn't obviously better since the format
    requires exactly 20 files; see the printed warning for what to do).

    With consensus_csv (the table cypmode.validation.pose_consensus
    writes), each compound's pose comes from whichever engine -- Boltz-2
    or OpenFold3 -- that table chose for it, and PoseBusters validity is
    read from that same table (it already holds the chosen pose's own
    result) instead of being re-run on Boltz-2 alone. Without it, the
    behaviour is exactly the original Boltz-2-only path.
    """
    test_df = pd.read_csv(test_set_path)
    names = test_df["Molecule_Name"].tolist()
    if len(names) != EXPECTED_COUNT:
        raise ValueError(f"expected {EXPECTED_COUNT} official test compounds, found {len(names)}")

    if consensus_csv is not None:
        consensus = pd.read_csv(consensus_csv).set_index("compound")
        missing = [n for n in names if n not in consensus.index]
        if missing:
            raise ValueError(f"consensus table {consensus_csv} has no row for: {missing}")
        source_cif_by_name = {n: Path(consensus.loc[n, "chosen_cif"]) for n in names}
        missing_cif = [n for n, p in source_cif_by_name.items() if not p.exists()]
        if missing_cif:
            raise FileNotFoundError(f"consensus-chosen structures missing on disk for: {missing_cif}")
        passed_col = {"boltz": "boltz_posebusters_ok", "openfold3": "of3_posebusters_ok"}
        failing = [n for n in names if not bool(consensus.loc[n, passed_col[consensus.loc[n, "engine"]]])]
    else:
        missing_cif = [n for n in names if not _cif_path(n, target).exists()]
        if missing_cif:
            raise FileNotFoundError(f"missing real Boltz-2 predictions for: {missing_cif}")

        posebusters_df = revalidate_posebusters(names, target)
        posebusters_df.to_csv(f"data/structure_track/outputs/{target}/posebusters_validity_official20.csv", index=False)
        failing = posebusters_df[~posebusters_df["all_passed"]]["compound"].tolist()
        model_index_by_name = dict(zip(posebusters_df["compound"], posebusters_df["model_index"]))
        source_cif_by_name = {
            n: _predictions_dir(n, target) / f"{target}_{n}_model_{int(model_index_by_name[n])}.cif"
            if pd.notna(model_index_by_name.get(n))
            else _cif_path(n, target)
            for n in names
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = out_dir / "_pdb_files"
    pdb_dir.mkdir(exist_ok=True)

    written = []
    for name in names:
        if name in failing and skip_posebusters_failures:
            continue
        pdb_path = pdb_dir / f"{name}.pdb"
        convert_cif_to_pdb(source_cif_by_name[name], pdb_path)
        written.append(pdb_path)

    zip_path = out_dir / zip_name
    write_deterministic_zip(written, zip_path)

    return {
        "zip_path": zip_path,
        "n_written": len(written),
        "n_posebusters_failures": len(failing),
        "posebusters_failures": failing,
    }


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--consensus",
        type=Path,
        default=None,
        help="engine-consensus CSV from cypmode.validation.pose_consensus; omit for the original Boltz-2-only build",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("submissions/structure_track"))
    parser.add_argument("--zip-name", default="structure_track_submission.zip")
    args = parser.parse_args()

    summary = build_structure_submission(consensus_csv=args.consensus, out_dir=args.out_dir, zip_name=args.zip_name)
    print(f"wrote {summary['zip_path']} with {summary['n_written']} PDB files")
    print(f"PoseBusters: {summary['n_posebusters_failures']}/{EXPECTED_COUNT} of the official compounds fail at least one check")
    if summary["posebusters_failures"]:
        print("failing compounds (included anyway -- format requires exactly 20 files):")
        for name in summary["posebusters_failures"]:
            print(f"  {name}")


if __name__ == "__main__":
    _main()
