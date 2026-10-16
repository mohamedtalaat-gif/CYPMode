"""The relaxation module's file handling and the arithmetic of its gate (the force-field part needs the conda environment ``openmm`` and is exercised by queue job 218)."""

import numpy as np
import pandas as pd
import pytest

from cypmode.validation.relax import write_pdb
from cypmode.validation.relax_gate import paired_greater, verdict

PDB = (
    "ATOM      1  CA  ALA A   1      10.000  11.000  12.000  1.00 50.00           C\n"
    "HETATM    2  C1  LIG B   1       1.000   2.000   3.000  1.00 80.00           C\n"
    "HETATM    3  O1  LIG B   1       4.000   5.000   6.000  1.00 81.00           O\n"
    "HETATM    4 FE   HEM C   1      -1.000  -2.000  -3.000  1.00 98.00          FE\n"
)


def test_write_pdb_replaces_only_the_ligand_coordinates(tmp_path):
    source = tmp_path / "in.pdb"
    source.write_text(PDB)
    out = tmp_path / "out" / "relaxed.pdb"
    write_pdb(source, np.array([[1.5, 2.5, 3.5], [-4.0, 5.25, 600.125]]), out)
    lines = out.read_text().splitlines()
    assert lines[0] == PDB.splitlines()[0] and lines[3] == PDB.splitlines()[3]
    assert lines[1][30:54] == "   1.500   2.500   3.500" and lines[1][54:] == PDB.splitlines()[1][54:]
    assert lines[2][30:54] == "  -4.000   5.250 600.125"


def test_write_pdb_refuses_a_wrong_number_of_coordinates(tmp_path):
    source = tmp_path / "in.pdb"
    source.write_text(PDB)
    with pytest.raises(ValueError, match="ligand atom lines"):
        write_pdb(source, np.zeros((3, 3)), tmp_path / "out.pdb")


def test_heme_parameters_keep_the_net_charge_and_cover_the_heavy_atoms():
    pytest.importorskip("openmm")
    from cypmode.validation.relax import heme_parameters

    parameters = heme_parameters()
    assert len(parameters) == 43 and parameters["FE"][2] == 0.0
    assert round(sum(v[0] for v in parameters.values()), 6) == -2.0


def test_paired_greater_is_one_sided_and_handles_ties():
    assert paired_greater(np.zeros(5)) == 1.0
    assert paired_greater(np.arange(1, 13, dtype=float)) < 0.01
    assert paired_greater(-np.arange(1, 13, dtype=float)) > 0.99
    assert paired_greater(np.full(12, 0.02), 0.01) < 0.01  # a constant gain above the shift


def table(before_pass, after_pass, lddt_shift, margin_shift, n=24):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"pose": range(n), "compound": [f"c{i // 2}" for i in range(n)], "sample": 0, "passed_before": [before_pass(i) for i in range(n)], "passed_after": [after_pass(i) for i in range(n)],
                         "lddt_pli_before": 0.6 + rng.normal(0, 0.05, n), "margin_before": 0.82 + rng.normal(0, 0.02, n), "rmsd": 0.4, "relax_error": ""}).assign(
        lddt_pli_after=lambda d: d.lddt_pli_before + lddt_shift + rng.normal(0, 0.002, n), margin_after=lambda d: d.margin_before + margin_shift + rng.normal(0, 0.002, n))


def test_the_gate_is_met_when_invalid_poses_are_repaired_and_the_rest_stays_put():
    v = verdict(table(lambda i: i % 3 != 0, lambda i: True, 0.0, 0.03))
    assert v["validity_met"] and v["margin_met"] and v["fidelity_met"] and v["met"]


def test_the_gate_is_not_met_when_relaxation_moves_poses_away_from_the_crystal():
    v = verdict(table(lambda i: True, lambda i: True, -0.05, 0.03))
    assert not v["fidelity_met"] and not v["met"]


def test_a_pose_that_fails_to_relax_is_left_out_of_every_mean():
    t = table(lambda i: i % 3 != 0, lambda i: True, 0.0, 0.03)
    t.loc[0, "relax_error"] = "ValueError: no template"
    assert verdict(t)["poses"] == len(t) - 1 and verdict(t)["failed_to_relax"] == 1
