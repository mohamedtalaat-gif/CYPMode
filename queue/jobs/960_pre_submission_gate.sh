#!/bin/bash
source queue/lib.sh
# The last gate: whether the queue, the checklist, the audit, the submission files, the documents, the notebook and the film are finished, consistent and fresh.
# It informs and never fails the queue; the verdict is in the log. The commit and the submission itself are separate steps.
need_file cypmode/audit/pre_submission.py
all_lower_done 960_pre_submission_gate || defer "earlier jobs not finished"
python3 -m cypmode.audit.pre_submission > $L/pre_submission_gate.log 2>&1
exit 0
