# Final TDI build (generated; every number is computed by `final_tdi_build.py`)

## CYP3A4
Final stack: **default+all_survivors** (28 candidates); surviving groups: ['chemprop_foundation', 'neural']; called rate 0.403 against target prevalence 0.310.

| stack | n_candidates | auc_z1.0 | mcc_z1.0 | auc_z1.96 | mcc_z1.96 | eligible | diff_z1.0 | p_z1.0 | diff_z1.96 | p_z1.96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 23 | 0.817 | 0.4415 | 0.8443 | 0.4867 | True | nan | nan | nan | nan |
| default+chemprop_foundation | 25 | 0.8303 | 0.4707 | 0.8587 | 0.5286 | True | 0.0133 | 0.0024 | 0.0144 | 0.0053 |
| default+neural | 26 | 0.8276 | 0.4609 | 0.855 | 0.5131 | True | 0.0106 | 0.0142 | 0.0106 | 0.0294 |
| default+all_survivors | 28 | 0.8338 | 0.476 | 0.8622 | 0.5388 | True | 0.0168 | 0.0002 | 0.0179 | 0.0009 |

## CYP2D6
Final stack: **default** (23 candidates); surviving groups: none; called rate 0.059 against target prevalence 0.059.

| stack | n_candidates | auc_z1.0 | mcc_z1.0 | auc_z1.96 | mcc_z1.96 | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| default | 23 | 0.7312 | 0.1672 | 0.7536 | 0.2036 | True |

## Files

| file | sha256 | rows | CYP3A4_positive_rate | CYP2D6_positive_rate |
| --- | --- | --- | --- | --- |
| candidate_A_hits_only.csv | 4776089ea44347c4db1935793d0e767eef9f36bc53d96974eea3fdd1f86e847c | 750 | 0.4026666666666667 | 0.058666666666666666 |
| candidate_B_selection_aware.csv | 2d51eea29ec529d49feb471c78de890ad859eecc2e5b27e24903fb4d29931a6e | 750 | 0.4026666666666667 | 0.058666666666666666 |
