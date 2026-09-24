import pytest

from cypmode.validation.build_input import build_boltz_yaml, find_coordinating_cysteine

# Real UniProt P08684 (human CYP3A4) sequence.
CYP3A4_SEQUENCE = (
    "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEW"
    "KRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKR"
    "MKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEIN"
    "GMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA"
)
KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"


def test_find_coordinating_cysteine_matches_known_position():
    # Real, previously-verified answer for CYP3A4: Cys442.
    assert find_coordinating_cysteine(CYP3A4_SEQUENCE) == 442
    assert CYP3A4_SEQUENCE[442 - 1] == "C"


def test_find_coordinating_cysteine_no_match_raises():
    with pytest.raises(ValueError, match="no FxxGxxxCxG"):
        find_coordinating_cysteine("ACDEFGHIKLMNPQRSTVWY")


def test_find_coordinating_cysteine_ambiguous_raises():
    motif = "FAAGAAACAG"  # matches the FxxGxxxCxG pattern
    with pytest.raises(ValueError, match="found 2 candidate"):
        find_coordinating_cysteine(motif + "XXXXXXXXXX" + motif)


def test_build_boltz_yaml_matches_hand_written_reference():
    yaml_text = build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 442)
    with open("data/boltz_test/cyp3a4_ketoconazole.yaml") as f:
        original = f.read()
    assert yaml_text == original


def test_build_boltz_yaml_rejects_wrong_residue():
    with pytest.raises(ValueError, match="not cysteine"):
        build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 1)


def test_build_boltz_yaml_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 99999)
