#!/bin/bash
source queue/lib.sh
# One run from a clean clone: the repository is cloned at its last commit, the data record is restored from its archives, the audit is run there and its checks are compared with those of the committed report.
need_file scripts/clean_clone_check.sh
need_done 970_data_record_build
all_lower_done 975_clean_clone_check || defer "earlier jobs not finished"
tree_committed || defer "the working tree is not committed"
ARCHIVES="$(cd .. && pwd)/$(basename "$PWD")-backup/data_record_archives"
bash scripts/clean_clone_check.sh "$ARCHIVES" > $L/clean_clone_check.log 2>&1 || exit 1
