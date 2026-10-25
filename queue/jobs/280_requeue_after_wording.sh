#!/bin/bash
source queue/lib.sh
# The wording of the project was corrected (no text names a person by a role; cypmode/audit/prose_rules.py, audit section 18, tests/test_prose_rules.py), and two generated documents carried the old words: the method
# report and the ledger of negative results are generated again here, before the audit, so that the audit and its test suite read the corrected text. Then the final chain runs again from the audit on, in order,
# on a quiet repository: the audit (900), the method report (910), the documents (921), the public-readiness scan (930), the notebook (942) and its public edition (947), the submission copies (950)
# and the gate (960). The figures, the card, the clips and the film do not read the corrected text and are not repeated.
all_lower_done 280_requeue_after_wording || defer "earlier jobs not finished"
python -m cypmode.audit.method_report > $L/method_report.log 2>&1 || exit 1
python -m cypmode.audit.method_report --check >> $L/method_report.log 2>&1 || exit 1
for id in 900_tests_and_audit 910_method_report 921_docs_state 930_public_readiness 942_build_notebook 947_notebook_public 950_submissions_final_sync 960_pre_submission_gate; do
  rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id
done
echo "re-queued: 900, 910, 921, 930, 942, 947, 950 and 960"
