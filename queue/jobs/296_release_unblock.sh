#!/bin/bash
source queue/lib.sh
# The release after the deadline: the held-back block is removed, the full card is drawn, one local commit holds what becomes visible, and the final chain 900 to 975 is queued again; it waits for the end of the challenge and for the version of the release.
need_file scripts/release_after_deadline.py
all_lower_done 296_release_unblock || defer "earlier jobs not finished"
if grep -q "HELD BACK UNTIL AFTER THE DEADLINE" .gitignore; then
  grep -q '^version: 0.3.0' CITATION.cff || defer "the version of the release is not set (scripts/bump_version.py)"
  tree_committed || defer "the working tree is not committed"
  python scripts/release_after_deadline.py > $L/release_after_deadline.log 2>&1 || defer "the challenge has not closed, or the release script stopped: see $L/release_after_deadline.log"
fi
# what the release script leaves to be drawn: the card of the submitted structures and its interactive page (PyMOL and Chrome); a second attempt does not draw what exists
[ -f docs/card/cypmode_card_full-light.gif ] || python docs/card/build_card.py --full > $L/card_full_build.log 2>&1 || exit 1
[ -f docs/card/interactive/full/index.html ] || python docs/card/build_interactive_card.py --full > $L/interactive_card_full_build.log 2>&1 || exit 1
python -m cypmode.audit.public_readiness --after-deadline > $L/public_readiness_after_deadline.log 2>&1 || exit 1
git add -A
git diff --cached --quiet || git commit -q -m "The results that were held back until the challenge closed, the full card, the film and the complete notebook are published: the block of the ignore file is removed, the README shows the full card and the film, the public edition of the notebook gives way to the complete one and the repository of the explorer is recorded by its files" || exit 1
for job in queue/jobs/9[0-7]*.sh; do id=$(basename "$job" .sh); rm -f queue/state/done/$id queue/state/failed/$id queue/state/attempts/$id; done
echo "released and committed; re-queued: the jobs 900 to 975"
