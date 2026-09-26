# TDI track — inputs

| Path | What it is | Source |
|---|---|---|
| `cyp-challenge-TRAIN_TDI.csv` | Real official training data (6,145 compounds, TDI + direct-inhibition labels) | `openadmet/cyp-challenge-train-test` on Hugging Face, Apache 2.0 |
| `cyp-challenge-TEST-BLINDED.csv` | Real official blinded test set (750 compounds, SMILES + name only) | same |
| `pubchem_cache/` | Targeted PubChem bioassay pulls (AID1851, AID891, AID884) | `cypmode.challenge.pubchem_auxiliary` |
| `bindingdb_cache/` | Targeted BindingDB CYP-binding-affinity pulls | `cypmode.challenge.bindingdb_auxiliary` |
| `tdc_cache/` | TDC's 5-isoform CYP inhibition panel | `cypmode.data.panel` |
| `octant_cache/` | OpenADMET/Octant's real CYP3A4 potency screen + the 6-drug TDI reference panel | `cypmode.data.octant_benchmark`, `cypmode.validation.tdi_reference` |

All real, narrowly-scoped external pulls (not bulk database dumps) —
see each source module's own docstring for exactly what was fetched
and why.
