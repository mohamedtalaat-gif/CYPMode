#!/bin/bash
source queue/lib.sh
# Scan for private paths, credentials, foreign e-mail addresses, oversized files; findings go to the log.
need_file cypmode/audit/public_readiness.py
all_lower_done 930_public_readiness || defer "earlier jobs not finished"
if grep -q "HELD BACK UNTIL AFTER THE DEADLINE" .gitignore; then AFTER=""; else AFTER="--after-deadline"; fi  # once the challenge has closed the results are meant to be published
python -m cypmode.audit.public_readiness $AFTER > $L/public_readiness.log 2>&1 || true
