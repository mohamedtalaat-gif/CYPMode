#!/bin/bash
source queue/lib.sh
# Method report and negative-results ledger regenerated from the final logs and audit (job 900 runs first).
need_file cypmode/audit/method_report.py
all_lower_done 910_method_report || defer "earlier jobs not finished"
python -m cypmode.audit.method_report > $L/method_report.log 2>&1
python -m cypmode.audit.method_report --check >> $L/method_report.log 2>&1
