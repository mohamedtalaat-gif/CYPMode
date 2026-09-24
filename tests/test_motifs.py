import pytest

from cypmode.metrics.motifs import classify_mechanism, detect_motifs

# Real ChEMBL SMILES for the four compounds a recent CYP3A4 cryo-EM paper
# solved structures for -- see cypmode/validation/structures.py.
KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
RITONAVIR = "CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1"
AZAMULIN = "CC[C@]1(C)C[C@@H](OC(=O)CSc2nnc(N)[nH]2)[C@]2(C)[C@H](C)CC[C@]3(CCC(=O)[C@H]32)[C@@H](C)[C@@H]1O"
VARDENAFIL = "CCCc1nc(C)c2c(=O)nc(-c3cc(S(=O)(=O)N4CCN(CC)CC4)ccc3OCC)[nH]n12"


def test_ketoconazole_matches_imidazole():
    assert "imidazole_Nsub" in detect_motifs(KETOCONAZOLE)


def test_ritonavir_matches_thiazole():
    assert detect_motifs(RITONAVIR) == ["thiazole"]


def test_azamulin_matches_triazole():
    assert "triazole_1_2_4" in detect_motifs(AZAMULIN)


def test_vardenafil_matches_a_motif():
    # Confirmed false positive (README's Validation section): this hits
    # imidazole_Nsub inside a fused pyrazolo-pyrimidinone core, not a
    # discrete azole ring -- Boltz-2 structural validation puts the closest
    # motif-matched nitrogen 10.54 A from the heme iron, well outside
    # coordination range.
    assert detect_motifs(VARDENAFIL)


def test_no_motif_compound():
    assert detect_motifs("CCO") == []
    assert classify_mechanism("CCO").startswith("No coordinating motif")


def test_classify_mechanism_type_ii():
    assert classify_mechanism(KETOCONAZOLE).startswith("Type II")


def test_empty_smiles_raises():
    with pytest.raises(ValueError):
        detect_motifs("")


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        detect_motifs("not a smiles string!!")
