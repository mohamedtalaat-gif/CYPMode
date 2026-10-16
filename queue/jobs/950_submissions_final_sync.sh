#!/bin/bash
source queue/lib.sh
# The very last job: submissions/ is rewritten from the final TDI candidates and the final structure zip, each checked with the portal's own rules (it never uploads anything).
need_file cypmode/challenge/submissions_sync.py
all_lower_done 950_submissions_final_sync || defer "earlier jobs not finished"
python3 -m cypmode.challenge.submissions_sync > $L/submissions_sync.log 2>&1
