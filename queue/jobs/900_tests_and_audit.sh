#!/bin/bash
source queue/lib.sh
# The full audit with the test suite, twice, byte-for-byte; only when every earlier job is finished (a frozen state) and the notebook is written.
all_lower_done 900_tests_and_audit || defer "earlier jobs not finished"
notebook_written  # every part and appendix of the notebook is written before the final audit
bash scripts/run_audit.sh > $L/audit_run.log 2>&1
