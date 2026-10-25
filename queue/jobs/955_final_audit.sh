#!/bin/bash
source queue/lib.sh
# The second audit of the final sequence: the full audit with the test suite, twice, byte-for-byte, on the state that is submitted. Job 900 ran the same audit before the method report, the documents, the explorer bundle, the notebook,
# the film and the files of submissions/ were generated from its result; this run checks them and everything else again. It starts when job 950 is done and the generated files are committed, and the pre-submission gate (job 960) reads its report.
need_file scripts/run_audit.sh
need_done 950_submissions_final_sync
all_lower_done 955_final_audit || defer "earlier jobs not finished"
tree_committed
bash scripts/run_audit.sh > $L/audit_run.log 2>&1
