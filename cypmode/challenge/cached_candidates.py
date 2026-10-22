# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Candidates whose predictions are produced outside the LightGBM environment and only read here (TabICL, the multitask network, Chemprop and CheMeleon).

Their predictions live in the shared prediction cache (``nested_runners_torch`` and ``nested_runners_chemprop`` fill it from ``.venv-ml``). A candidate of family
``cached`` is never fitted in this process: when its cache is complete the runners skip it and the group joins the addition test and the final build like any
other, and when it is incomplete the fit raises instead of silently producing nothing.
"""

from cypmode.challenge.nested_runners import Candidate


def tabicl_candidates() -> list[Candidate]:
    return [Candidate("tabicl_svd", "cached")]


def mlp_candidates() -> list[Candidate]:
    return [Candidate("mlp_multitask_label", "cached"), Candidate("mlp_multitask_shift", "cached")]


def mlp_aux_candidates() -> list[Candidate]:
    return [Candidate("mlp_multitask_aux_label", "cached"), Candidate("mlp_multitask_aux_shift", "cached")]


def atomjepa_head_candidates() -> list[Candidate]:
    return [Candidate("atomjepa_head", "cached")]


def tabpfn_candidates() -> list[Candidate]:
    return [Candidate("tabpfn_svd", "cached")]


def tabicl_monroe_candidates() -> list[Candidate]:
    return [Candidate("tabicl_monroe", "cached")]


def tabicl_chemeleon_candidates() -> list[Candidate]:
    return [Candidate("tabicl_chemeleon", "cached")]


def tabpfn_chemeleon_candidates() -> list[Candidate]:
    return [Candidate("tabpfn_chemeleon", "cached")]


def chemprop_candidates() -> list[Candidate]:
    return [Candidate("chemprop_mt_shift", "cached")]


def chemeleon_candidates() -> list[Candidate]:
    return [Candidate("chemeleon_mt_shift", "cached")]


def neural_candidates() -> list[Candidate]:
    """The two learned-representation candidates whose near-misses joined into one lead on CYP3A4 (the multitask network and plain Chemprop); a group of its own, confirmed on the 10-outer-fold plan."""
    return mlp_candidates() + chemprop_candidates()


def chemprop_foundation_candidates() -> list[Candidate]:
    return [Candidate("chemeleon_minimol_mt_shift", "cached"), Candidate("chemeleon_surrogate_mt_shift", "cached")]


GROUPS = {"atomjepa_head": atomjepa_head_candidates, "chemprop_foundation": chemprop_foundation_candidates, "tabicl": tabicl_candidates, "tabpfn": tabpfn_candidates, "tabicl_monroe": tabicl_monroe_candidates, "tabicl_chemeleon": tabicl_chemeleon_candidates, "tabpfn_chemeleon": tabpfn_chemeleon_candidates, "mlp": mlp_candidates, "mlp_aux": mlp_aux_candidates, "chemprop": chemprop_candidates, "chemeleon": chemeleon_candidates, "neural": neural_candidates}
