import numpy as np
import pandas as pd
import pytest

from cypmode.challenge import official_extra as oe

pytestmark = pytest.mark.skipif(not oe.SCREEN_FILE.exists(), reason="requires the official extra files (data/tdi_track/inputs/official_extra)")


def test_a_batch_identifier_gives_its_compound():
    assert oe.compound_of_batch("OCNT-0017486-AA-002") == "OCNT-0017486"


def test_the_guard_removes_only_rows_of_blinded_test_compounds():
    test = sorted(oe.test_compounds())
    frame = pd.DataFrame({"ocnt_batch": [f"{test[0]}-AA-001", "OCNT-0000001-AA-001", f"{test[1]}-AB-002"], "value": [1, 2, 3]})
    kept, dropped = oe.exclude_test_compounds(frame, "ocnt_batch")
    assert dropped == sorted({test[0], test[1]}) and kept.value.tolist() == [2]
    plain, none = oe.exclude_test_compounds(pd.DataFrame({"Molecule_Name": ["OCNT-0000001"]}), "Molecule_Name", batch=False)
    assert len(plain) == 1 and none == []


def test_screen_arrays_align_to_the_requested_names_and_leave_unknown_compounds_empty():
    names = list(pd.read_csv(oe.TRAIN_CSV).Molecule_Name[:50]) + ["OCNT-NOT-THERE"]
    arrays = oe.screen_arrays(names)
    assert all(len(v) == len(names) for v in arrays.values())
    assert np.isnan(arrays["CYP2D6_screen"][-1]) and np.isnan(arrays["CYP2D6_screen_active"][-1])


def test_no_blinded_test_compound_has_a_reading_in_any_official_file():
    names = oe.all_names()
    test_rows = np.arange(len(names))[-len(oe.test_compounds()):]
    arrays = oe.aux_arrays(names)
    assert all(np.isnan(v[test_rows]).all() for v in arrays.values())


def test_the_audit_passes_on_the_files_as_downloaded():
    failed = [c.name for c in oe.audit() if c.ok is False]
    assert not failed, failed


def test_the_pxr_single_concentration_loader_drops_blinded_test_compounds_and_takes_the_median_per_concentration(tmp_path, monkeypatch):
    import pandas as pd

    from cypmode.challenge import official_extra as oe

    rows = [("OCNT-1", "Library", 8.3e-6, 0.2), ("OCNT-1", "Library", 8.3e-6, 0.6), ("OCNT-1", "Library", 3.3e-5, 1.0), ("OCNT-2", "Library", 8.3e-6, 5.0), ("OCNT-3", "Positive Control", 8.3e-6, 9.0)]
    path = tmp_path / "single.csv"
    pd.DataFrame(rows, columns=["OCNT_ID", "compound_class", "concentration_M", "log2_fc_estimate"]).to_csv(path, index=False)
    monkeypatch.setattr(oe, "PXR_SINGLE_FILE", path)
    monkeypatch.setattr(oe, "test_compounds", lambda: {"OCNT-2"})
    table, dropped = oe.load_pxr_single()
    assert dropped == ["OCNT-2"] and list(table.compound) == ["OCNT-1"]
    assert table.loc[0, "PXR_fc_8uM"] == 0.4 and table.loc[0, "PXR_fc_33uM"] == 1.0  # the median of two readings, and the second concentration
