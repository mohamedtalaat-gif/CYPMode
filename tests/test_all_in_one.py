# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The all-in-one comparison: how the two stacks are named and when the registered outcome is MET."""

from cypmode.challenge import all_in_one as module


def test_the_stacks_keep_each_name_once_and_in_the_order_of_the_groups():
    groups = {"physics": ["phys_a", "shared"], "neural": ["net_a", "shared"], "mmp": ["mmp_a"]}
    assert module.stack_names(["d1", "d2"], groups) == ["d1", "d2", "phys_a", "shared", "net_a", "mmp_a"]
    assert module.stack_names(["d1"], groups, ["neural"]) == ["d1", "net_a", "shared"]
    assert module.stack_names(["d1"], groups, []) == ["d1"]


def test_the_outcome_needs_both_cuts_and_the_floor():
    both = [{"z0": 1.0, "diff": 0.012, "p": 0.01}, {"z0": 1.96, "diff": 0.011, "p": 0.04}]
    assert module.verdict_line("CYP3A4", both).endswith(": MET")
    assert module.verdict_line("CYP3A4", [both[0], {"z0": 1.96, "diff": 0.011, "p": 0.06}]).endswith("NOT MET")
    assert module.verdict_line("CYP3A4", [both[0], {"z0": 1.96, "diff": 0.004, "p": 0.01}]).endswith("NOT MET")
    assert module.verdict_line("CYP3A4", [{"z0": 1.0, "diff": -0.01, "p": 0.001}, both[1]]).endswith("NOT MET")


def test_the_log_is_named_so_that_no_gate_parser_reads_it():
    from cypmode.challenge import final_tdi_build as build

    patterns = [pattern for _, pattern in build.GROUPS.values()]
    assert not any(pattern.format(iso="CYP3A4") == "exploratory_all_in_one_CYP3A4.log" for pattern in patterns)


def test_a_candidate_without_a_deployment_fit_is_still_complete_for_the_comparison():
    class Task:
        def __init__(self, repeat):
            self.repeat = repeat

    class Cache:
        def __init__(self, missing):
            self._missing = missing

        def missing(self, name):
            return self._missing[name]

    cache = Cache({"cross_validated_only": [Task(-1)], "complete": [], "lacks_a_fold": [Task(0), Task(-1)]})
    assert module.cv_complete(cache, "cross_validated_only") and module.cv_complete(cache, "complete")
    assert not module.cv_complete(cache, "lacks_a_fold")
