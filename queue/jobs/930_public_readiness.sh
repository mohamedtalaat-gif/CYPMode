#!/bin/bash
source queue/lib.sh
# Scan for private paths, credentials, foreign e-mail addresses, oversized files; findings go to the log.
need_file cypmode/audit/public_readiness.py
all_lower_done 930_public_readiness || defer "earlier jobs not finished"
python -m cypmode.audit.public_readiness > $L/public_readiness.log 2>&1 || true
