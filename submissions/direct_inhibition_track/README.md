# Direct Inhibition track — submission

**Not started.** This project's real work so far is on the TDI and
Structure Prediction tracks only. This folder exists only so the schema
is on record if/when this track is picked up — see the still-open
question in the main conversation about whether to extend the current
engine to cover it.

## Required format (verified directly against the live HF Space's `config.py`/`submission.py`, 2026-09-26)

- File: `.parquet` (preferred) or `.csv`.
- Exactly **750 rows**.
- Exactly these 6 columns, case-sensitive:
  `SMILES`, `Molecule_Name`, `CYP1A2_pIC50_direct_inhibition`,
  `CYP2C9_pIC50_direct_inhibition`, `CYP2D6_pIC50_direct_inhibition`,
  `CYP3A4_pIC50_direct_inhibition`.
- All four endpoint columns must be finite floats (no NaN/inf).
- The live validator also rejects a column with near-constant predictions
  below a `MIN_PREDICTION_STD` threshold (`"Error: {col} predictions are
  constant or near-constant"`) — a real check beyond dtype/shape.

Scoring is MA-ST-RAE (Macro-Averaged Soft-Threshold Relative Absolute
Error) across the 4 isoforms, per the official FAQ already documented in
`CYPMode_Architecture_Specification.md`.
