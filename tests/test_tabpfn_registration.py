# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""TabPFN is registered like TabICL (a cached candidate group) and its jobs wait for the licence acceptance instead of failing."""

from pathlib import Path

from cypmode.challenge.cached_candidates import GROUPS
from cypmode.challenge.final_tdi_build import GROUPS as FINAL_GROUPS
from cypmode.challenge.nested_runners_torch import CANDIDATES
from cypmode.challenge.nested_runners_torch import GROUPS as RUNNERS

JOBS = Path("queue/jobs")


def test_the_group_is_known_to_the_runner_the_cache_reader_and_the_final_build():
    assert "tabpfn" in GROUPS and [c.name for c in GROUPS["tabpfn"]()] == CANDIDATES["tabpfn"] == ["tabpfn_svd"]
    assert "tabpfn" in RUNNERS and "tabpfn" in FINAL_GROUPS


def test_the_jobs_defer_until_the_licence_marker_exists_and_the_addition_waits_for_both_caches():
    for job in ("221_tabpfn_CYP3A4.sh", "222_tabpfn_CYP2D6.sh"):
        text = (JOBS / job).read_text()
        assert "need_file data/tdi_track/inputs/tabpfn_licence_accepted.txt" in text and "--groups tabpfn" in text
    assert "need_done 221_tabpfn_CYP3A4 222_tabpfn_CYP2D6" in (JOBS / "223_tabpfn_addition.sh").read_text()
