#!/bin/bash
source queue/lib.sh
# The first audit of the final sequence: the full audit with the test suite, twice, byte-for-byte; only when every earlier job is finished (a frozen state), the notebook is written and the working tree is committed.
# Everything that quotes the audit or counts the files of the repository (the method report, the documents, the notebook, the film, the files of submissions/) is generated after this job from the state it describes;
# job 955 runs the same audit again on the generated state.
all_lower_done 900_tests_and_audit || defer "earlier jobs not finished"
notebook_written  # every part and appendix of the notebook is written before the final audit
tree_committed
bash scripts/run_audit.sh > $L/audit_run.log 2>&1
