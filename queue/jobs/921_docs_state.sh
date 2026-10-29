#!/bin/bash
source queue/lib.sh
# The generated blocks of README.md and CYPMode_Architecture_Specification.md (the final stack per isoform, the verdicts of the gates, the structure recipe): rewritten from the final build, the logs of the
# gates and the recipe, then checked against their sources; the pages of docs/map are written and checked with them. Runs after the method report, which reads the same logs.
need_file cypmode/audit/docs_state.py
all_lower_done 921_docs_state || defer "earlier jobs not finished"
python -m cypmode.audit.docs_state > $L/docs_state.log 2>&1
python -m cypmode.audit.docs_state --check >> $L/docs_state.log 2>&1
python -m cypmode.audit.project_map >> $L/docs_state.log 2>&1
python -m cypmode.audit.project_map --check >> $L/docs_state.log 2>&1
