import numpy as np
import pytest
from rdkit import Chem

from cypmode.challenge.feature_store import BLOCKS, _alert_context

NAMES = ["classes", "matches", "logP", "any_logP", "matches_logP", "mw", "any_mw", "f_near", "f_total", "cf3", "blocked_frac", "labile", "any_labile", "crowding", "any_arom"]


def _f(smiles):
    return dict(zip(NAMES, _alert_context(Chem.MolFromSmiles(smiles))))


def test_a_molecule_without_an_alert_has_zero_interaction_features_but_keeps_its_plain_descriptors():
    f = _f("CCCCCCCC")
    assert f["classes"] == f["matches"] == f["any_logP"] == f["any_mw"] == f["any_labile"] == f["crowding"] == 0.0
    assert f["logP"] > 0 and f["mw"] > 0


def test_a_blocking_fluorine_is_counted_only_when_it_sits_near_the_alert():
    near = _f("c1ccoc1C(F)(F)F")  # CF3 on the furan itself
    far = _f("c1coc(c1)CCc1ccc(F)cc1")  # fluorine on the distant phenyl ring
    assert near["f_near"] == 3 and near["cf3"] == 1
    assert far["f_near"] == 0 and far["f_total"] == 1


def test_the_alert_lipophilicity_interaction_scales_with_logp_and_vanishes_without_an_alert():
    plain_furan = _f("c1ccoc1C")
    greasy_furan = _f("c1ccoc1CCCCCCCCCC")
    assert greasy_furan["any_logP"] > plain_furan["any_logP"] > 0
    assert greasy_furan["any_logP"] == pytest.approx(greasy_furan["logP"])


def test_the_block_is_registered_and_stacks_one_row_per_molecule():
    mols = [Chem.MolFromSmiles(s) for s in ("c1ccoc1C", "CCO", "c1ccsc1CN(C)C")]
    block = BLOCKS["tdi_alert_context"](mols)
    assert block.shape == (3, len(NAMES)) and np.isfinite(block).all()
