"""GFN2-xTB part of physics tier B. Own pytest invocation (tblite and LightGBM deadlock in one process):
    pytest tests/test_physics_tierb.py
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.tblite

pytest.importorskip("tblite")

from cypmode.challenge import physics_block as pb
from cypmode.challenge import physics_tierb as tb


def sites_of(smiles):
    _, rows = pb.compute_molecule(smiles)
    a = pd.DataFrame(rows).dropna(subset=["hat_kcal"]).sort_values("hat_kcal")
    return [(int(r.atom), float(r.hat_kcal), str(r.ch_role)) for r in a.head(2).itertuples()]


def test_relaxation_lowers_the_abstraction_energy_and_keeps_the_site_ordering():
    toluene = tb.compute_molecule_b("Cc1ccccc1", sites_of("Cc1ccccc1"))
    ethane = tb.compute_molecule_b("CC", sites_of("CC"))
    assert toluene["phys_b_hat_relax_gain_1"] > 5.0  # the unrelaxed radical was too high
    assert toluene["phys_b_hat_relaxed_1"] < ethane["phys_b_hat_relaxed_1"]  # benzylic still easier than ethane
    assert toluene["phys_b_site1_is_benzylic"] == 1.0 and toluene["phys_b_n_conf"] >= 1


def test_features_are_complete_deterministic_and_failure_gives_nan_row():
    smiles = "COc1ccccc1"
    a = tb.compute_molecule_b(smiles, sites_of(smiles))
    b = tb.compute_molecule_b(smiles, sites_of(smiles))
    assert set(a) == set(tb.FEATURE_NAMES) and all(a[k] == pytest.approx(b[k], rel=1e-9) for k in a if not np.isnan(a[k]))
    assert np.isfinite(a["phys_b_ip_ev"]) and a["phys_b_conf_entropy"] >= 0
    bad = tb.compute_molecule_b("not a smiles !!!", [])
    assert all(np.isnan(v) for v in bad.values())


def test_extraction_is_resumable(tmp_path):
    atoms = tmp_path / "atoms.csv"
    smiles = ["Cc1ccccc1", "CCO"]
    rows = []
    for s in smiles:
        _, r = pb.compute_molecule(s)
        rows += r
    pd.DataFrame(rows).assign(SMILES=lambda d: d.SMILES.map(pb.canonical)).to_csv(atoms, index=False)
    cache = tmp_path / "b.csv"
    tb.extract(smiles, cache, atoms, n_workers=1)
    tb.extract(smiles, cache, atoms, n_workers=1)
    assert len(pd.read_csv(cache)) == 2 and tb.load_tierb_features(smiles, cache).shape == (2, len(tb.FEATURE_NAMES))
