from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from cypmode.challenge import final_tdi_build as fb


def test_gate_lines_are_parsed_per_group_and_isoform(tmp_path):
    (tmp_path / "physics_experiment_CYP3A4.log").write_text(
        "noise\nCYP3A4 GATE (alone, corrected p < 0.05 at both z0): NOT MET\nCYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET\n"
    )
    (tmp_path / "pair_transfer_CYP3A4.log").write_text("CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): NOT MET\n")
    (tmp_path / "physics_experiment_CYP2D6.log").write_text("no verdict line here\n")
    gates = fb.parse_gates(tmp_path)
    assert gates[("physics", "CYP3A4")] == {"alone": False, "stacked": True}
    assert gates[("pair_desc", "CYP3A4")] == {"stacked": False}
    assert ("physics", "CYP2D6") not in gates and ("mmp", "CYP3A4") not in gates
    assert fb.survivors(gates, "CYP3A4") == ["physics"] and fb.survivors(gates, "CYP2D6") == []


def test_every_group_names_real_candidates_and_a_log_pattern():
    for group, (builder, pattern) in fb.GROUPS.items():
        names = [c.name for c in builder()]
        assert names and len(set(names)) == len(names) and "{iso}" in pattern, group


def test_written_files_are_validated_reloaded_and_hashed(tmp_path):
    test = pd.read_csv(fb.TEST_CSV)
    rng = np.random.default_rng(0)
    info = fb.write_validated(test, rng.random(len(test)) < 0.07, rng.random(len(test)) < 0.3, tmp_path / "c.csv")
    assert info["rows"] == 750 and len(info["sha256"]) == 64 and 0.2 < info["CYP3A4_positive_rate"] < 0.4
    with pytest.raises((AssertionError, ValueError)):
        fb.write_validated(test.iloc[:700], rng.random(700) < 0.07, rng.random(700) < 0.3, tmp_path / "bad.csv")


def test_leakage_audit_passes_a_clean_plan_and_stops_on_a_leak():
    labels = np.full(20, -1)
    labels[:15] = 1
    plan = SimpleNamespace(external=np.arange(15, 20), pool=np.arange(0, 15))
    ctx = SimpleNamespace(labels=labels)
    assert all(fb.leakage_audit(plan, ctx, 15, 20).values())
    labels[17] = 0  # a stored test label
    with pytest.raises(AssertionError, match="LEAKAGE"):
        fb.leakage_audit(plan, ctx, 15, 20)


def test_markdown_table_has_header_rule_and_rows():
    out = fb.markdown_table(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert out.splitlines()[0] == "| a | b |" and len(out.splitlines()) == 4


def test_portal_checks_match_the_published_rules(tmp_path):
    from cypmode.challenge.submission import portal_validate_classification, portal_validate_structure

    test = pd.read_csv(fb.TEST_CSV)
    rng = np.random.default_rng(1)
    good = pd.DataFrame({"SMILES": test.SMILES, "Molecule_Name": test.Molecule_Name, "CYP2D6_is_TDI": rng.integers(0, 2, 750), "CYP3A4_is_TDI": rng.random(750) < 0.3, "extra": 1})
    good.to_csv(tmp_path / "good.csv", index=False)
    assert portal_validate_classification(tmp_path / "good.csv") == (True, "ok")  # the portal accepts 0/1 and extra columns
    for name, mutate, text in (
        ("short", lambda d: d.iloc[:700], "Expected 750 rows"),
        ("nan", lambda d: d.assign(CYP3A4_is_TDI=[np.nan] + [True] * 749), "NaN"),
        ("nonbinary", lambda d: d.assign(CYP2D6_is_TDI=[2] + [0] * 749), "non-binary"),
        ("constant", lambda d: d.assign(CYP2D6_is_TDI=False), "one class"),
        ("nocolumn", lambda d: d.drop(columns=["CYP3A4_is_TDI"]), "Missing required"),
    ):
        mutate(good).to_csv(tmp_path / f"{name}.csv", index=False)
        ok, message = portal_validate_classification(tmp_path / f"{name}.csv")
        assert not ok and text in message, (name, message)
    assert portal_validate_classification(tmp_path / "x.json")[0] is False
    import zipfile

    with zipfile.ZipFile(tmp_path / "s.zip", "w") as z:
        for i in range(20):
            z.writestr(f"x{i}.pdb", "ATOM")
    assert portal_validate_structure(tmp_path / "s.zip") == (True, "ok")
    with zipfile.ZipFile(tmp_path / "s19.zip", "w") as z:
        for i in range(19):
            z.writestr(f"x{i}.pdb", "ATOM")
    assert portal_validate_structure(tmp_path / "s19.zip")[0] is False


def test_replicate_verdict_lines_are_read_as_stacked_passes(tmp_path):
    (tmp_path / "replicate_verdict_CYP3A4.log").write_text(
        "physics: 5-fold stacked ... | 10-fold stacked ...\n"
        "CYP3A4 GATE (replicate physics, 10 outer folds corrected p < 0.05 at both z0 and the same sign in the 5-fold design): MET\n"
        "CYP3A4 GATE (replicate physics_b, 10 outer folds corrected p < 0.05 at both z0 and the same sign in the 5-fold design): NOT MET\n"
    )
    gates = fb.parse_replicate_gates(tmp_path)
    assert gates[("physics", "CYP3A4")] == {"stacked": True} and gates[("physics_b", "CYP3A4")] == {"stacked": False}
    assert fb.survivors(gates, "CYP3A4") == ["physics"]


def test_new_physics_groups_are_registered_with_log_patterns():
    assert {"physics_compact", "physics_b"} <= set(fb.GROUPS) and fb.DESIGN["outer"] == 5


def test_a_group_found_after_the_15_fold_results_joins_only_when_its_replicate_is_met(tmp_path):
    """The neural pair passes the 15-fold gate on CYP3A4 but joins the final stack only if the 10-outer-fold replicate is MET too; a missing replicate line counts as a fail."""
    from cypmode.challenge import final_tdi_build as build

    (tmp_path / "aux_neural_CYP3A4.log").write_text("CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET\n")
    (tmp_path / "aux_screen_CYP3A4.log").write_text("CYP3A4 GATE (stacked, corrected p < 0.05 at both z0): MET\n")
    assert build.survivors(build.parse_gates(tmp_path), "CYP3A4") == ["screen"]  # no replicate line yet: only the ordinary group survives
    (tmp_path / "replicate_verdict_CYP3A4.log").write_text("CYP3A4 GATE (replicate neural, 10 outer folds corrected p < 0.05 at both z0 and the same sign in the 5-fold design): NOT MET\n")
    assert "neural" not in build.survivors(build.parse_gates(tmp_path), "CYP3A4")
    (tmp_path / "replicate_verdict_CYP3A4.log").write_text("CYP3A4 GATE (replicate neural, 10 outer folds corrected p < 0.05 at both z0 and the same sign in the 5-fold design): MET\n")
    assert sorted(build.survivors(build.parse_gates(tmp_path), "CYP3A4")) == ["neural", "screen"]
