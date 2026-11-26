# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The table of the five OECD principles: the literals read from a module, the curation counts, the machine-readable algorithm, the enrichment of alerts, and the layout of the table."""

import json

import numpy as np
import pandas as pd
import pytest

from cypmode.challenge import oecd_principles as oecd


def test_module_constants_reads_literals_and_skips_the_rest(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text("import torch\nEPOCHS = 30\nLR: float = 1e-3\nSIZES = (64, 128)\nPATH = Path('x') / 'y'\nlower = 3\nDEVICE = torch.device('cpu')\n")
    assert oecd.module_constants(source) == {"EPOCHS": 30, "LR": 1e-3, "SIZES": (64, 128)}


def test_the_curation_counts_are_those_of_the_files():
    table = oecd.curation_table()
    value = dict(zip(table.check, table.value, strict=True))
    assert value["compounds in the training file"] == 6145 and value["compounds in the blinded file"] == 750
    assert value["blinded compounds whose canonical SMILES is in the training file"] == 0
    assert value["CYP3A4: labelled compounds with both arms measured (the pool of every model)"] == 2334
    assert value["CYP2D6: labelled compounds with both arms measured (the pool of every model)"] == 1493
    for iso in oecd.ISOFORMS:
        assert value[f"{iso}: rows of the pool where the rule s = t - max(d, 4) > log10 2 differs from the given label"] == 0


def test_the_algorithm_manifest_lists_every_member_of_each_final_stack_and_is_json():
    manifest = json.loads((oecd.OUT / "manifest.json").read_text())
    algorithm = oecd.algorithm_manifest(manifest)
    text = json.dumps(algorithm, sort_keys=True)
    assert json.loads(text)["blend"]["selector"] == "StackedLogit"
    for iso in oecd.ISOFORMS:
        members = algorithm["isoforms"][iso]["members"]
        assert len(members) == manifest["isoforms"][iso]["n_candidates"]
        assert all({"name", "family", "blocks", "params", "group"} <= set(m) for m in members)
        assert len({m["name"] for m in members}) == len(members)
    assert "lgbm_A_deep" in {m["name"] for m in algorithm["isoforms"]["CYP3A4"]["members"]}


def test_the_enrichment_finds_a_planted_alert_and_leaves_noise_alone():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    planted = ((y == 1) & (rng.random(600) < 0.5)) | ((y == 0) & (rng.random(600) < 0.1))
    noise = rng.random(600) < 0.3
    rare = np.zeros(600)
    rare[:3] = 1
    table = oecd.enrichment_table("CYP3A4", {"planted": planted.astype(float), "noise": noise.astype(float), "rare": rare}, y)
    assert "rare" not in set(table.feature)
    planted_row = table[table.feature == "planted"].iloc[0]
    assert planted_row.odds_ratio > 4 and planted_row.p_holm < 1e-6 and planted_row.or_low > 1
    noise_row = table[table.feature == "noise"].iloc[0]
    assert noise_row.p_holm > 0.05 and noise_row.or_low < 1 < noise_row.or_high
    assert table.iloc[0].feature == "planted"


def test_the_markdown_has_one_row_per_cell_without_a_stray_bar():
    rows = [{"principle": "1. A defined endpoint", "isoform": "CYP3A4", "evidence": "a rule", "source": "a file"}]
    curation = pd.DataFrame({"check": ["rows"], "value": pd.Series([6145], dtype=object)})
    text = oecd.markdown(rows, curation)
    table_lines = [line for line in text.splitlines() if line.startswith("| 1.")]
    assert table_lines == ["| 1. A defined endpoint | CYP3A4 | a rule. | a file |"]
    assert "| rows | 6145 |" in text and "6145.0" not in text


def test_the_pending_marker_names_the_job():
    assert oecd.pending("287_external_dataset") == "(pending: 287_external_dataset)"
    with pytest.raises(TypeError):
        oecd.pending()
