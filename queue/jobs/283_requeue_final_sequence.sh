#!/bin/bash
source queue/lib.sh
# The order of the final sequence (decided 2026-10-25): the audit twice in a row (job 900), then everything that the audit, the code or the counts of the repository change (the explorer bundle, the method report, the documents, the
# readiness scan, the notebook, the film, the public edition of the notebook, the files of submissions/), then the audit twice in a row again (job 955), then the gate. The explorer gained the outer-fold and feature-block controls
# (candidates.csv, blocks.csv and the fold columns of the bundle), section 12.7 describes them, and the film drew a count of queue jobs and a hash of the audit report of an earlier state, so the film, the notebook and the documents
# are drawn again after job 900. The results of the film and of the cards are those of the stack that is submitted (docs/video/facts.py read the default stack of CYP3A4 before), so the two cards are built again as well.
all_lower_done 283_requeue_final_sequence || defer "earlier jobs not finished"
for id in 900_tests_and_audit 906_explorer_bundle 910_method_report 921_docs_state 930_public_readiness 941_card_build 942_build_notebook 945_video_render 946_interactive_card 947_notebook_public 950_submissions_final_sync 955_final_audit 960_pre_submission_gate; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 900, 906, 910, 921, 930, 941, 942, 945, 946, 947, 950, 955, 960"
