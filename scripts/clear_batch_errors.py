# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Remove failed entries from batch results JSON files so the next run retries them (run_batch skips any name already present)."""

import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    if not path.exists():
        continue
    results = json.loads(path.read_text())
    kept = {k: v for k, v in results.items() if "error" not in v}
    if len(kept) != len(results):
        path.write_text(json.dumps(kept, indent=2) + "\n")
        print(f"{path.name}: cleared {len(results) - len(kept)} failed entries, kept {len(kept)}")
