#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The data record of the project: ``python scripts/fetch_data.py build --out-dir DIR | check | fetch [--source FOLDER_OR_URL] [--group NAME ...]``.

``build`` packs the files that git does not track into the archives of ``data_record/record.json`` and writes the manifest (size and SHA-256 of every archive) and the list of paths of every group.
``check`` lists what is missing on this machine. ``fetch`` copies or downloads the archives, refuses any whose SHA-256 differs from the manifest and extracts them where the code looks for them.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cypmode import data_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="pack the groups and write the manifest")
    build.add_argument("--out-dir", type=Path, required=True, help="where the archives are written (outside the repository)")
    sub.add_parser("check", help="list the files of the record that are not on this machine")
    fetch = sub.add_parser("fetch", help="restore the record from its archives")
    fetch.add_argument("--source", default=None, help="a folder or a URL prefix that holds the archives (default: the address in the manifest)")
    fetch.add_argument("--group", action="append", default=None, help="restore only this group (repeat for several)")
    args = parser.parse_args()
    if args.command == "build":
        manifest = data_record.build(ROOT, args.out_dir)
        for e in manifest["archives"]:
            print(f"{e['archive']}: {e['files']} files, {e['bytes'] / 1e9:.2f} GB, sha256 {e['sha256']}")
        stray = data_record.left_out(ROOT)
        if stray:
            print(f"{len(stray)} untracked file(s) under the folders of the policy belong to no group and are not left out by name, e.g. {stray[0]}")
            return 1
        return 0
    if args.command == "check":
        gone = data_record.missing(ROOT)
        print(data_record.message_for(gone) if gone else "nothing is missing" if (ROOT / data_record.MANIFEST).exists() else "no manifest yet: the data record has not been made")
        return 1 if gone else 0
    print("restored:", ", ".join(data_record.fetch(ROOT, args.source, args.group)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
