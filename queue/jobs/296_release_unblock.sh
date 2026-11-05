#!/bin/bash
source queue/lib.sh
# The release after the deadline: the held-back block is removed, one local commit holds what becomes visible, and the final chain 900 to 960 is queued again; it waits for the end of the challenge and for the version of the release.
need_file scripts/release_after_deadline.py
all_lower_done 296_release_unblock || defer "earlier jobs not finished"
grep -q "HELD BACK UNTIL AFTER THE DEADLINE" .gitignore || { echo "the held-back block is already removed: nothing to release"; exit 0; }
grep -q '^version: 0.3.0' CITATION.cff || defer "the version of the release is not set (scripts/bump_version.py)"
tree_committed || defer "the working tree is not committed"
python scripts/release_after_deadline.py > $L/release_after_deadline.log 2>&1 || defer "the challenge has not closed, or the release script stopped: see $L/release_after_deadline.log"
python -m cypmode.audit.public_readiness --after-deadline > $L/public_readiness_after_deadline.log 2>&1 || exit 1
git add -A
git commit -q -m "The results that were held back until the challenge closed, the full card, the film and the complete notebook are published: the block of the ignore file is removed, the README shows the full card and the film, the public edition of the notebook gives way to the complete one and the repository of the explorer is recorded by its files" || exit 1
for job in queue/jobs/9[0-6]*.sh; do id=$(basename "$job" .sh); rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id; done
echo "released and committed; re-queued: the jobs 900 to 960"
