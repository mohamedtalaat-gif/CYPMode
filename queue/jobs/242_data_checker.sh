#!/bin/bash
source queue/lib.sh
# The organisers' data checker (OpenADMET/challenge-data-checker, Apache-2.0, in its own environment .venv-checker with Python 3.12): the official training, Emax and direct-inhibition files against the blinded test file.
[ -x .venv-checker/bin/challenge-data-checker ] || defer "the data checker is not installed in .venv-checker"
export NO_COLOR=1
for c in tdi_train_vs_test emax_vs_test inhibition_vs_test; do
  .venv-checker/bin/challenge-data-checker audit/data_checker/$c.toml > $L/data_checker_$c.log 2>&1 || exit 1  # a FAIL banner here is a finding
done
.venv-checker/bin/challenge-data-checker audit/data_checker/tdi_train_vs_test_0.9.toml > $L/data_checker_tdi_train_vs_test_0.9.log 2>&1 || true  # near-duplicates at 0.9 are information, not a failure
