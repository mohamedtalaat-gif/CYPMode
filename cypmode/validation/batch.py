# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

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
import logging
from pathlib import Path

import pandas as pd

from cypmode.validation.pipeline import run_structural_validation

logger = logging.getLogger(__name__)

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
                # Absent, not just None, when the batch ran with --no-affinity
                # (summarize_compound(expect_affinity=False) never adds these
                # keys at all) -- .get() rather than indexing is required, not
                # a defensive nicety here.
                affinity_pred_value=result.get("affinity_pred_value"),
                affinity_probability_binary=result.get("affinity_probability_binary"),
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


def run_batch(
    compounds: dict[str, str],
    results_path: str | Path,
    table_path: str | Path | None = None,
    per_compound: dict[str, dict] | None = None,
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
            results[name] = run_structural_validation(name, smiles, **{**pipeline_kwargs, **(per_compound or {}).get(name, {})})
        except Exception as exc:
            results[name] = {"error": str(exc)}
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        if table_path is not None:
            # The JSON checkpoint above is the real result -- already
            # written and safe. This snapshot is a convenience re-export
            # that regenerates in full from `results` on the very next
            # iteration anyway, so a failure here (confirmed directly: a
            # process without pyarrow/fastparquet installed raises
            # ImportError from to_parquet) must not be allowed to kill the
            # rest of a real multi-hour batch over one missing snapshot
            # write -- that would lose every compound still queued after
            # the one being processed when it happened, not just this
            # snapshot.
            try:
                df = to_dataframe(results, compounds)
                df.to_csv(Path(table_path).with_suffix(".csv"), index=False)
                df.to_parquet(Path(table_path).with_suffix(".parquet"), index=False)
            except Exception as exc:
                logger.warning("could not write table snapshot for %r (continuing the batch): %s", table_path, exc)

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
    parser.add_argument(
        "--no-affinity",
        dest="predict_affinity",
        action="store_false",
        help="skip the affinity property entirely -- only the structure/coordination geometry is needed "
        "for compounds that don't require a predicted affinity value (confirmed: the affinity head alone "
        "runs 5x the diffusion sampling of the base structure by default)",
    )
    parser.add_argument("--recycling-steps", type=int, default=None, help="passed through to boltz predict")
    parser.add_argument("--max-msa-seqs", type=int, default=None, help="passed through to boltz predict")
    parser.add_argument("--num-subsampled-msa", type=int, default=None, help="passed through to boltz predict")
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=None,
        help="structure diffusion samples per compound (boltz predict --diffusion_samples); more samples "
        "give a pool of poses to select from at a small extra cost over the single trunk pass",
    )
    parser.add_argument(
        "--max-parallel-samples",
        type=int,
        default=None,
        help="cap on diffusion samples held in memory at once (boltz predict --max_parallel_samples); "
        "1 runs the samples one after another, so peak memory stays that of a single sample -- the "
        "unbounded default put a 16 GB Mac into a swap-exhaustion kernel panic with --diffusion-samples 5",
    )
    parser.add_argument(
        "--protein-msa",
        default=None,
        help="precomputed MSA file for the protein chain (reused for every ligand: the sequence is the same)",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=None,
        help="Boltz sampling step scale (boltz predict --step_scale); lower values give more diverse samples",
    )
    parser.add_argument(
        "--donor-restraint",
        choices=("bond", "contact"),
        default=None,
        help="steer each ligand onto the heme iron through its predicted donor nitrogen (needs --no-affinity); "
        "ligands without a Type II donor run unrestrained",
    )
    parser.add_argument(
        "--diffusion-samples-affinity",
        type=int,
        default=None,
        help="passed through to boltz predict -- default is 5, the single biggest real cost/memory lever "
        "for compounds that do need affinity (confirmed directly: this is on top of, not instead of, the "
        "base structure's own single diffusion sample)",
    )
    parser.add_argument(
        "--target",
        default="cyp3a4",
        help="protein name used in file/directory naming only -- must be passed explicitly "
        "alongside --protein-sequence for a different real isoform (e.g. --target cyp2d6), "
        "or its files would silently collide with cyp3a4's own",
    )
    parser.add_argument(
        "--protein-sequence",
        default=None,
        help="defaults to CYP3A4 (UniProt P08684) if not given -- pass the real sequence "
        "for whatever --target names, they are independent flags",
    )
    parser.add_argument(
        "--write-embeddings",
        action="store_true",
        help="also save Boltz-2's own pooled single/pair trunk representations (a real, first-class "
        "--write_embeddings CLI flag, confirmed in third_party/boltz-main's own source) -- a richer "
        "structural feature than a single geometric distance, at roughly the cost of a slightly larger "
        "output write, no extra compute (the representations are already computed either way)",
    )
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    compounds = {row["Molecule_Name"]: row["SMILES"] for row in rows}
    per_compound = {}
    for row in rows:  # optional per-row options, e.g. from cypmode.validation.conformer_benchmark / diversity_pilot
        options = {}
        if row.get("Template_PDB"):
            from cypmode.validation.conformers import template_spec

            path, chain = template_spec(row["Template_PDB"])
            options["templates"] = [(path, chain) if chain else path]
        if row.get("Seed"):
            options["seed"] = int(float(row["Seed"]))
        if row.get("Step_Scale"):
            options["step_scale"] = float(row["Step_Scale"])
        if row.get("Diffusion_Samples"):
            options["diffusion_samples"] = int(float(row["Diffusion_Samples"]))
        if options:
            per_compound[row["Molecule_Name"]] = options

    results_path = Path(args.results_path)
    table_path = results_path.with_suffix("") if args.write_table else None
    protein_kwargs = {"protein_sequence": args.protein_sequence} if args.protein_sequence is not None else {}
    results = run_batch(
        compounds,
        results_path,
        table_path=table_path,
        target=args.target,
        stream_output=args.stream_output,
        predict_affinity=args.predict_affinity,
        recycling_steps=args.recycling_steps,
        max_msa_seqs=args.max_msa_seqs,
        num_subsampled_msa=args.num_subsampled_msa,
        diffusion_samples=args.diffusion_samples,
        diffusion_samples_affinity=args.diffusion_samples_affinity,
        max_parallel_samples=args.max_parallel_samples,
        donor_restraint=args.donor_restraint,
        step_scale=args.step_scale,
        protein_msa=args.protein_msa,
        per_compound=per_compound,
        write_embeddings=args.write_embeddings,
        **protein_kwargs,
    )
    n_errors = sum(1 for r in results.values() if "error" in r)
    print(f"done: {len(results)} compound(s), {n_errors} error(s), written to {args.results_path}")


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    _main()
