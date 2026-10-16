# TDI track: submission files

The files below are the candidates of the final build (`cypmode.challenge.final_tdi_build`, files and hashes in `data/tdi_track/outputs/final/manifest.json`), copied here
unchanged and checked again with the portal's own rules. **None of them has been uploaded.** The portal takes one file per submission; which one to upload is decided before submitting.

## Required format (read from the challenge portal's `config.py` and `submission.py`, 2026-09-26)

- A `.parquet` or `.csv` file with exactly **750 rows** and exactly these 4 columns, in this order, case-sensitive: `SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`, `CYP3A4_is_TDI`.
- The two endpoint columns are boolean (`True`/`False` or `1`/`0`), without missing values, and neither may hold a single class.

## Files

| file | candidate | rows | CYP3A4 positive rate | CYP2D6 positive rate | schema check | portal check | sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tdi_track_candidate_A_hits_only.csv` | candidate_A_hits_only | 750 | 0.372 | 0.059 | valid | valid | `546ee71e6381ab3e50f85069f48fb31d78854e3cb4ccc931453feae18355a45b` |
| `tdi_track_candidate_B_selection_aware.csv` | candidate_B_selection_aware | 750 | 0.372 | 0.059 | valid | valid | `701c4e80b71023e207fe518090b2fe68551e263ed9c0afe2c264158ba3a74e7c` |

## What the candidates are

- `candidate_A_hits_only`: the stack fitted on the compounds that were measured (the scored population is assumed to be the measured compounds).
- `candidate_B_selection_aware`: for CYP2D6, the stack fitted on the measured compounds plus the never-measured compounds of the training panel as negatives (the scored population is assumed to look like the training panel); CYP3A4 is the same as in A.

The first two candidates differ in 0 CYP3A4 calls and 40 CYP2D6 calls.

## How they were made

- CYP3A4: stack `default` of 23 candidate models; the top 37.2% of the test compounds are called (1.2 times the prevalence 0.310 that the public leaderboard implies).
- CYP2D6: stack `default` of 23 candidate models; the top 5.9% of the test compounds are called (1 times the prevalence 0.059 that the public leaderboard implies).

The method, every experiment that was tried and the gate each one met or missed are in the project's method report and in `data/tdi_track/outputs/final/final_report.md`.
