# Structure track — inputs

| Path | What it is |
|---|---|
| `*.yaml` (78 files) | Real Boltz-2 job specs, one per compound: protein sequence + heme (covalently bonded to the real axial cysteine) + ligand SMILES. Built by `cypmode.validation.build_input`, not hand-written (except the original 4 reference compounds — see the root README's Validation section). |
| `structure_track_test_blinded.csv` | The real official Structure Track compound set |
| `structure_track_retry.csv` | Compounds re-queued after a real MSA-server timeout on the first attempt |

Running `boltz predict` against any of these YAMLs (via
`cypmode.validation.pipeline.run_structural_validation` or
`cypmode.validation.batch.run_batch`) regenerates everything in
`../outputs/out/` from scratch — that's the reproducibility contract:
these inputs plus the pipeline code are the real, complete recipe.
