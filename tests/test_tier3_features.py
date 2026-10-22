# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge.tier3_features import (
    NO_MOTIF_DISTANCE_SENTINEL,
    add_tier3_features,
    load_tier3_structural_features,
    repeated_cv_tier3_comparison,
)


def _fake_results_file(tmp_path):
    results = {
        "OCNT-A": {
            "coordinated": True,
            "closest_motif_nitrogen_distance_angstrom": 2.1,
            "confidence_score": 0.9,
        },
        "OCNT-B": {
            "coordinated": False,
            "closest_motif_nitrogen_distance_angstrom": None,  # no Type II motif at all
            "confidence_score": 0.85,
        },
        "OCNT-C": {"error": "Too many failed attempts for the MSA generation request."},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    return path


def test_load_tier3_structural_features_parses_real_schema(tmp_path):
    df = load_tier3_structural_features(_fake_results_file(tmp_path))
    assert set(df["Molecule_Name"]) == {"OCNT-A", "OCNT-B"}  # OCNT-C errored, correctly excluded

    a = df[df["Molecule_Name"] == "OCNT-A"].iloc[0]
    assert a["tier3_coordinated"] == 1.0
    assert a["tier3_fe_n_distance"] == pytest.approx(2.1)

    b = df[df["Molecule_Name"] == "OCNT-B"].iloc[0]
    assert b["tier3_coordinated"] == 0.0
    # no motif nitrogen at all -> the real, distinct sentinel, not a fabricated small distance
    assert b["tier3_fe_n_distance"] == NO_MOTIF_DISTANCE_SENTINEL


def test_add_tier3_features_leaves_uncovered_rows_as_real_missing_not_zero(tmp_path):
    df = pd.DataFrame({"Molecule_Name": ["OCNT-A", "OCNT-B", "OCNT-NOT-IN-TIER3"]})
    out = add_tier3_features(df, results_path=_fake_results_file(tmp_path))

    covered = out[out["Molecule_Name"] == "OCNT-A"].iloc[0]
    assert covered["has_tier3_data"]
    assert covered["tier3_coordinated"] == 1.0

    uncovered = out[out["Molecule_Name"] == "OCNT-NOT-IN-TIER3"].iloc[0]
    assert not uncovered["has_tier3_data"]
    assert pd.isna(uncovered["tier3_coordinated"])
    assert pd.isna(uncovered["tier3_fe_n_distance"])  # missing, not 0 and not the no-motif sentinel


def test_repeated_cv_tier3_comparison_only_scores_tier3_covered_validation_rows():
    rng = np.random.default_rng(0)
    n = 200
    m3_features = rng.normal(size=(n, 10))
    y = (rng.random(n) > 0.5).astype(int)

    # only the first 40 rows carry real Tier3 data -- mimics the real
    # ~50-of-3584 sparsity ratio, small enough to actually exercise
    n_covered = 40
    has_tier3 = np.zeros(n, dtype=bool)
    has_tier3[:n_covered] = True
    tier3_features = np.full((n, 3), np.nan)
    tier3_features[has_tier3] = rng.normal(size=(n_covered, 3))

    results = repeated_cv_tier3_comparison(
        m3_features, tier3_features, y, has_tier3, n_repeats=2, n_splits=5
    )

    assert set(results["variant"]) == {"M3", "M3+Tier3"}
    # every scored fold's n_val_tier3 should roughly match n_covered/n_splits,
    # never the full fold size (n/n_splits) -- confirms scoring really is
    # restricted to Tier3-covered rows, not silently falling back to everyone
    assert results["n_val_tier3"].max() < (n / 5)
    assert results["n_val_tier3"].sum() > 0

    # a real 40-row-covered-of-200 run should produce output for both
    # repeats (5 folds x 2 variants x 2 repeats, minus any single-class
    # dropouts) -- not silently empty
    assert len(results) > 0
