#!/bin/bash
source queue/lib.sh
# The data record: the files that git does not hold, packed into the archives of data_record/record.json next to the repository, with their manifest and lists of paths committed locally; it follows the final chain and the release.
need_file scripts/fetch_data.py
all_lower_done 970_data_record_build || defer "earlier jobs not finished"
grep -q "HELD BACK UNTIL AFTER THE DEADLINE" .gitignore && defer "the held-back block is still there: the record is made after the release"
ARCHIVES="$(cd .. && pwd)/$(basename "$PWD")-backup/data_record_archives"
python scripts/fetch_data.py build --out-dir "$ARCHIVES" > $L/data_record_build.log 2>&1 || exit 1
git add data_record
git diff --cached --quiet || git commit -q -m "The manifest of the data record, with the size and the SHA-256 of each archive, and the list of the paths of each group" || exit 1
echo "archives in $ARCHIVES"
