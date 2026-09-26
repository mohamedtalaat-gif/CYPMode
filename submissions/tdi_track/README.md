# TDI track — submission

## Required format (verified directly against the live HF Space's `config.py`/`submission.py`, 2026-09-26)

- File: `.parquet` (preferred) or `.csv`.
- Exactly **750 rows**.
- Exactly these 4 columns, in this order, case-sensitive:
  `SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`, `CYP3A4_is_TDI`.
- `CYP2D6_is_TDI` / `CYP3A4_is_TDI` must be boolean (`True`/`False` or `1`/`0`), no NaN.
- The live validator also rejects a column that's all one class (`"Error: {col} predictions are all one class"`) — a real, useful pre-flight check beyond just dtype/shape.

This matches `cypmode/challenge/submission.py`'s `REQUIRED_COLUMNS`/`EXPECTED_ROW_COUNT`
exactly — that module already builds and validates a candidate submission
against this schema (`build_submission`, `validate_submission`). No changes
needed there; it was already schema-correct before this check.

## Current status

**Not yet built.** Per `CYPMode_Architecture_Specification.md`'s "Status:
done vs. remaining," the final assembled model (Tier 1/2/3 + the tested
augmentation layer's practical default + OOF bridge + the per-isoform
model choice) doesn't exist yet — it's blocked on Tier 3's structural
features being wired in. Once it does, running it on
`data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv` and passing the
result through `cypmode.challenge.submission.build_submission` +
`validate_submission` produces the real file that belongs in this folder.
