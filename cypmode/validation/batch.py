"""Batch orchestration: run structural validation across many compounds, one call.

cypmode.validation.pipeline.run_structural_validation handles one compound
end to end, but a real batch (e.g. every compound in an external blinded
test set) needs that call repeated across many compounds, with two things
a plain loop doesn't give for free: one compound's failure (a bad SMILES,
a Boltz-2 crash) shouldn't lose every other compound's result, and a run
spanning many hours needs its results checkpointed as it goes, not only
written out at the very end.
"""

import json
from pathlib import Path

import pandas as pd

from cypmode.validation.pipeline import run_structural_validation

_TABLE_COLUMNS = [
    "compound",
    "smiles",
    "coordinated",
    "closest_motif_nitrogen_distance_angstrom",
    "closest_motif",
    "closest_motif_atom",
    "confidence_score",
    "affinity_pred_value",
    "affinity_probability_binary",
    "error",
]


def to_dataframe(results: dict, compounds: dict[str, str] | None = None) -> pd.DataFrame:
    """Flatten a results dict (name -> summarize_compound()'s output, or
    {"error": ...} for a failed compound) into one row per compound.

    Keeps the scalar fields a quick read or a spreadsheet needs; the full
    nested per-nitrogen distance lists (motif_nitrogen_distances,
    all_ligand_nitrogen_distances) stay in the source JSON only -- they
    don't flatten into a row without inventing a schema for "however many
    nitrogens this particular ligand happens to have". Pass `compounds`
    (the same name->SMILES mapping given to run_batch) to include SMILES in
    the table; without it, the SMILES column is left empty.
    """
    rows = []
    for name, result in results.items():
        row = {col: None for col in _TABLE_COLUMNS}
        row["compound"] = name
        row["smiles"] = compounds.get(name) if compounds else None
        if "error" in result:
            row["error"] = result["error"]
        else:
            closest = result["motif_nitrogen_distances"][0] if result["motif_nitrogen_distances"] else None
            row.update(
                coordinated=result["coordinated"],
                closest_motif_nitrogen_distance_angstrom=result["closest_motif_nitrogen_distance_angstrom"],
                closest_motif=closest["motif"] if closest else None,
                closest_motif_atom=closest["atom"] if closest else None,
                confidence_score=result["confidence_score"],
                affinity_pred_value=result["affinity_pred_value"],
                affinity_probability_binary=result["affinity_probability_binary"],
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


def run_batch(
    compounds: dict[str, str],
    results_path: str | Path,
    table_path: str | Path | None = None,
    **pipeline_kwargs,
) -> dict:
    """Run run_structural_validation for every name->SMILES pair in `compounds`.

    Writes results_path after every compound, so a multi-hour run's
    progress survives an interruption. Resumes automatically: names already
    present in an existing results_path are skipped rather than re-run. A
    per-compound failure is recorded as {"error": ...} rather than raised,
    so one bad compound doesn't stop the rest of the batch.

    If table_path is given, also (re)writes table_path.csv and
    table_path.parquet from the results so far after every compound -- a
    flat, shareable snapshot (see to_dataframe) kept in sync with the JSON
    checkpoint rather than only generated once at the end.
    """
    results_path = Path(results_path)
    results: dict = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name, smiles in compounds.items():
        if name in results:
            continue
        try:
            results[name] = run_structural_validation(name, smiles, **pipeline_kwargs)
        except Exception as exc:
            results[name] = {"error": str(exc)}
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        if table_path is not None:
            df = to_dataframe(results, compounds)
            df.to_csv(Path(table_path).with_suffix(".csv"), index=False)
            df.to_parquet(Path(table_path).with_suffix(".parquet"), index=False)

    return results


def _main() -> None:
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV with Molecule_Name,SMILES columns")
    parser.add_argument("results_path", help="JSON file to checkpoint results to (resumes if it already exists)")
    parser.add_argument(
        "--no-table",
        dest="write_table",
        action="store_false",
        help="skip writing a .csv/.parquet snapshot alongside the JSON (written by default, same base name)",
    )
    parser.add_argument("--no-stream", dest="stream_output", action="store_false")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        compounds = {row["Molecule_Name"]: row["SMILES"] for row in csv.DictReader(f)}

    results_path = Path(args.results_path)
    table_path = results_path.with_suffix("") if args.write_table else None
    results = run_batch(compounds, results_path, table_path=table_path, stream_output=args.stream_output)
    n_errors = sum(1 for r in results.values() if "error" in r)
    print(f"done: {len(results)} compound(s), {n_errors} error(s), written to {args.results_path}")


if __name__ == "__main__":
    _main()
