# Structure track — outputs

Everything this project has produced for the Structure Prediction
(pose) track: real Boltz-2 and OpenFold3 predictions, physical-validity
checks, the cross-engine consensus, and the derived calculation sheets. Committed vs. regenerable is split
deliberately, not by accident — see below.

## Organized by target, not flat

**Reorganized 2026-09-29**: this whole tree used to mix CYP3A4 and
CYP2D6 Boltz-2 output in one flat directory (CYP2D6 work only exists
because `cypmode.validation.pipeline`'s default `out_dir` was reused for
the CYP2D6 due-diligence batch, not because it's part of the official
Structure Track, which is CYP3A4-only). Every module that writes or
reads these paths (`cypmode.validation.pipeline`, `build_summary`,
`pose_validity`, `cypmode.challenge.structure_submission`) now nests by
target automatically:

```
data/structure_track/outputs/
├── out/                          # gitignored, real Boltz-2 raw output
│   ├── cyp3a4/                   # boltz_results_cyp3a4_* (reference panel, official 20, Tier 3)
│   └── cyp2d6/                   # boltz_results_cyp2d6_* (CYP2D6 due-diligence batch)
├── openfold3/                    # gitignored, real OpenFold3 raw output (second engine)
│   └── cyp3a4/                   # {name}/ per compound + results.json checkpoint
├── cyp3a4/                       # committed derived results, see table below
│   ├── validation_summary.json
│   ├── structure_track_results.{json,csv,parquet}
│   ├── posebusters_validity.csv
│   ├── posebusters_validity_official20.csv
│   ├── engine_consensus.csv      # Boltz-2 vs OpenFold3 pose choice per compound
│   ├── pose_rescoring_validation.csv  # how well the rescoring scores separate near-native from wrong poses
│   └── *.log
└── README.md
```

There is no top-level `cyp2d6/` here yet: CYP2D6 has no official
Structure Track submission (that track is CYP3A4-only by the real
challenge rules), so its derived results live under
`data/tdi_track/outputs/cyp2d6/` instead, where the due-diligence work
they actually serve lives.

## What's committed here (`cyp3a4/`)

| File | Produced by | What it is |
|---|---|---|
| `validation_summary.json` | `python -m cypmode.validation.build_summary` | Fe–ligand-nitrogen coordination distances for the 4 structurally-validated reference compounds (ketoconazole, ritonavir, azamulin, vardenafil) |
| `structure_track_results.{json,csv,parquet}` | `python -m cypmode.validation.batch data/structure_track/inputs/structure_track_test_blinded.csv data/structure_track/outputs/cyp3a4/structure_track_results.json` | Real Boltz-2 structural predictions for the official Structure Track compound set |
| `posebusters_validity.csv` | `python -m cypmode.validation.pose_validity` | The mandatory PoseBusters physical/chemical sanity check (see `CYPMode_Architecture_Specification.md`'s "Official challenge update" section for why this is scored, not optional) run across every real completed CYP3A4 structure this project has — TDI reference panel, Structure Track compounds, and Tier 3 |
| `posebusters_validity_official20.csv` | `python -m cypmode.challenge.structure_submission` | The same check, scoped to exactly the 20 official Structure Track compounds (not the other ~55 CYP3A4 structures sharing the same `out/cyp3a4/`) |
| `engine_consensus.csv` | `python -m cypmode.validation.pose_consensus` | Per official compound: which engine's pose goes into the submission and why (PoseBusters gate, pocket-aligned symmetry-aware ligand RMSD between engines, heme-coordination prior), plus both engines' own PoseBusters/Fe–N measurements. Valid at any point of a partially finished OpenFold3 batch (a compound without an OpenFold3 prediction keeps Boltz-2). Feeds `python -m cypmode.challenge.structure_submission --consensus <this file>` |
| `pose_rescoring_validation.csv` | `python -m cypmode.validation.pose_rescoring_validation` | One row per (crystal ligand, leave-out regime): Spearman / AUROC / true-pose rank of the two rescoring scores against rigidly perturbed decoys of that ligand's own crystal pose, with the compound's own crystal entries removed from the reference set (and, in the second regime, every reference ≥ 0.5 Morgan-similar to it). The honest measure of how much a score can be trusted as a tie-breaker; see the caveat below |
| `*.log` | Individual `boltz predict` runs | Real run logs kept for specific compounds worth a paper trail (heme handling, ritonavir, vardenafil) |

## The two-engine pipeline (Boltz-2 + OpenFold3)

OpenFold3 (OpenBind-0 checkpoint, run from the isolated gitignored
`.venv-openfold3`, Apple MPS) is an **added second engine**, not a
replacement: Boltz-2 stays the default and a compound's pose only changes
engine on concrete evidence. Both engines read the same ligand/protein
from the compound's Boltz-2 input YAML, so they always predict the same
system.

```
python -m cypmode.validation.openfold3 [--names ... | --limit N] [--samples K]   # ~16 min/compound, strictly one at a time, resumable
python -m cypmode.validation.reference_ligands                                   # once: RCSB co-crystal ligands -> inputs/cyp3a4_reference_ligands.json
python -m cypmode.validation.pose_consensus [--no-rescoring]                     # -> cyp3a4/engine_consensus.csv
python -m cypmode.challenge.structure_submission --consensus data/structure_track/outputs/cyp3a4/engine_consensus.csv
```

Selection rule (`cypmode.validation.pose_consensus.choose_pose`), applied
in order, first match wins:

1. **PoseBusters gate.** A failure scores zero on all three metrics, so
   exactly one passing engine wins.
2. **Agreement.** Both pass and the pocket-aligned, symmetry-aware ligand
   RMSD is ≤ 2.0 Å: keep Boltz-2.
3. **Heme-coordination prior.** Both pass but disagree and only one
   engine's Type II motif nitrogen coordinates the iron: that engine wins.
4. **Rescoring tie-breaker** (below), only when a decision is still open.
5. Otherwise keep Boltz-2.

Model confidence is never used (unreliable on P450s). Memory is the
binding constraint (16 GB; one prediction already pushes free RAM to
~9%), so never run OpenFold3 alongside Boltz-2.

### Rescoring against experimental co-crystal ligands

The Structure Track is scored against sequestered structures, so no
predicted pose can be scored locally. What can be asked instead is how
well a pose agrees with where experimentally resolved ligands sit in this
pocket. `cypmode.validation.reference_ligands` builds the reference set
(`data/structure_track/inputs/cyp3a4_reference_ligands.json`) from the
RCSB entries for CYP3A4 (UniProt P08684): drug-sized ligands whose nearest
heavy atom is within 12 Å of the heme iron (the peripheral site is
excluded on purpose), plus each entry's pocket C-alpha atoms. The raw
mmCIFs are cached in the gitignored `inputs/pdb_cache/`.
`cypmode.validation.pose_rescoring` superposes every reference into the
candidate's frame on the pocket C-alpha atoms (a reference that does not
superpose within 1.5 Å is dropped, never forced) and computes two scores
from heavy-atom coordinates alone:

- `template_score` — similarity-weighted mean Gaussian-overlap Tanimoto
  against the 5 chemically most similar crystal ligands; `template_support`
  is the best Morgan similarity found, and the rule ignores the template
  score below 0.5 because a weak analog is not evidence.
- `occupancy_score` — the average, over the candidate's heavy atoms, of
  the fraction of all reference ligands that occupy that space. Works for
  a novel scaffold; penalizes poses in the access channel or on the
  surface, says little about two poses that are both in the active site.

Both engines are rescored with the same references, and a tie-breaker only
fires when the difference clears a margin (`TEMPLATE_MARGIN`,
`OCCUPANCY_MARGIN` in `pose_consensus.py`); inside the margin Boltz-2
stays. The margins were set from the decoy-gap statistics rather than
tuned to the compounds with crystal structures.

**Caveat.** The decoy validation (rigid perturbations of a crystal pose)
overstates what these scores can do on real engine errors: the poses the
engines get wrong are chemically plausible, near the same site, and
usually wrong in the same way as each other. `pose_rescoring_validation.validate_engine_poses`
measures the scores against the true crystal RMSD of real Boltz-2 /
OpenFold3 poses for the compounds that have crystal structures; that, not
the decoy table, is the evidence for how far to trust the tie-breaker.

## What's real but NOT committed, and why

`out/` (gitignored) holds the actual predicted structures Boltz-2
writes per compound — mmCIF coordinates, MSA search results,
confidence/affinity JSON. These are real simulation outputs, not
placeholders, but they're regenerated on demand from `data/structure_track/inputs/*.yaml`
by `cypmode/validation/pipeline.py` / `cypmode/validation/batch.py`,
and they're large (each compound's MSA search alone can be tens of MB).
Committing them would mean committing a multi-GB, fully-regenerable
binary cache — the same reasoning `third_party/boltz_cache/` (Boltz-2's
own 7.6GB model-weight cache) is excluded for. The small, real numbers
*derived* from `out/` (coordination distances, PoseBusters pass/fail,
BiSyRMSD, pocket geometry) are what's tracked above and in
`data/tdi_track/outputs/` — that's the reproducible record; `out/`
itself is regenerated by re-running the pipeline against the committed
input YAMLs.

## Figures — real gap, not yet filled

`docs/figures/tdi_track/` exists (MCC-by-pipeline-stage charts, see its
own README). **There is no `docs/figures/structure_track/` yet** —
PoseBusters pass-rate breakdowns, BiSyRMSD/pocket-geometry
distributions, and real docking-pose visualizations for this track
haven't been generated. This is a known, explicit gap, not an
oversight to be quietly left — see the architecture spec's decision log
for it as an open item.
