# Final TDI build (generated; every number is computed by `final_tdi_build.py`)

## CYP3A4
Final stack: **default+neural** (26 candidates); surviving groups: ['neural']; called rate 0.340 against target prevalence 0.310.

| stack | n_candidates | auc_z1.0 | mcc_z1.0 | auc_z1.96 | mcc_z1.96 | eligible | diff_z1.0 | p_z1.0 | diff_z1.96 | p_z1.96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 23 | 0.8243 | 0.464 | 0.8503 | 0.5093 | True | nan | nan | nan | nan |
| default+neural | 26 | 0.8352 | 0.4836 | 0.8618 | 0.5341 | True | 0.0109 | 0.0256 | 0.0115 | 0.0382 |

## CYP2D6
Final stack: **default** (23 candidates); surviving groups: none; called rate 0.059 against target prevalence 0.059.

| stack | n_candidates | auc_z1.0 | mcc_z1.0 | auc_z1.96 | mcc_z1.96 | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| default | 23 | 0.7318 | 0.1354 | 0.7444 | 0.1582 | True |

## Files

| file | sha256 | rows | CYP3A4_positive_rate | CYP2D6_positive_rate |
| --- | --- | --- | --- | --- |
| candidate_A_hits_only.csv | 2e280f5604aa713e7093d93cc5e80eddde02339d81b017bb8ede644cb2ae62ae | 750 | 0.34 | 0.058666666666666666 |
| candidate_B_selection_aware.csv | 202d43a10d58cca6ab08fae210b293b25edbb88694e9f5d96594a3f7441f913c | 750 | 0.34 | 0.058666666666666666 |
