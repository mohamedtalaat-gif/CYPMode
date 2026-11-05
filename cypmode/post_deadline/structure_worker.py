# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The scoring of poses with the organisers' own functions, run in the environment that holds OpenStructure and PoseBusters.

``python structure_worker.py JOB.json OUT.json ORGANISERS_DIR`` reads a job, imports the evaluation code of the organisers from ORGANISERS_DIR and writes the scores. This file imports nothing of the
project, so that the environment of OpenStructure needs no more than the packages of the evaluation code.

A job of mode ``submission`` holds ``predicted`` and ``reference`` (molecule name to path) and ``smiles`` (molecule name to SMILES) and is scored by ``score_structure_predictions`` exactly as the
board does, penalties included. A job of mode ``poses`` holds a list of ``items`` with a ``key``, a ``model`` and a ``reference`` path and the ``smiles``; each is scored by ``score_single_structure`` with the same
settings, which is the call that ``score_structure_predictions`` makes for each compound, and the failed PoseBusters checks are returned with it.
"""

import json
import math
import sys


def main(job_path: str, out_path: str, organisers_dir: str) -> int:
    sys.path.insert(0, organisers_dir)
    from loguru import logger

    logger.remove()
    from evaluation import evaluate_predictions as evaluate
    from evaluation.config import POSEBUSTERS_MAX_FAILURES

    with open(job_path) as handle:
        job = json.load(handle)
    if job["mode"] == "submission":
        per_compound, failures = evaluate.score_structure_predictions(job["predicted"], job["reference"], max_pb_failures=POSEBUSTERS_MAX_FAILURES, smiles_map=job["smiles"])
        result = {"per_compound": per_compound.to_dict(orient="records"), "pb_failures": failures}
    else:
        rows = []
        for item in job["items"]:
            scores, failed = evaluate.score_single_structure(item["model"], item["reference"], max_pb_failures=POSEBUSTERS_MAX_FAILURES, smiles=item["smiles"])
            rows.append({"key": item["key"], **{k: (None if math.isnan(v) else v) for k, v in scores.items()}, "pb_failed": failed})
        result = {"rows": rows}
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
