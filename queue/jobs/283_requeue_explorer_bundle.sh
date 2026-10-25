#!/bin/bash
source queue/lib.sh
# The explorer gained the outer-fold and feature-block controls (2026-10-25): the data bundle holds two new tables (candidates.csv, blocks.csv) and the fold of every compound in each repeat (the columns fold|r of the out-of-fold files),
# and section 12.7 of the notebook describes them. The bundle, the documents that quote it, the readiness check, the notebook and the last steps of the chain are run again, after the film.
all_lower_done 283_requeue_explorer_bundle || defer "earlier jobs not finished"
for id in 900_tests_and_audit 906_explorer_bundle 910_method_report 921_docs_state 930_public_readiness 942_build_notebook 947_notebook_public 950_submissions_final_sync 960_pre_submission_gate; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 900, 906, 910, 921, 930, 942, 947, 950, 960"
