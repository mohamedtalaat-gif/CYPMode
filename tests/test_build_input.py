# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import pytest
import yaml

from cypmode.validation.build_input import (
    boltz_atom_name,
    build_boltz_yaml,
    find_coordinating_cysteine,
    heme_donor_atom,
)

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
    with open("data/structure_track/inputs/cyp3a4_ketoconazole.yaml") as f:
        original = f.read()
    assert yaml_text == original


def test_build_boltz_yaml_rejects_wrong_residue():
    with pytest.raises(ValueError, match="not cysteine"):
        build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 1)


def test_build_boltz_yaml_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 99999)


def test_build_boltz_yaml_predict_affinity_false_omits_properties_block():
    yaml_text = build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 442, predict_affinity=False)
    assert "properties:" not in yaml_text
    assert "affinity:" not in yaml_text
    # everything else stays identical to the affinity-on version
    with_affinity = build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 442, predict_affinity=True)
    assert with_affinity.startswith(yaml_text)


def _constraints(**kwargs):
    text = build_boltz_yaml(CYP3A4_SEQUENCE, kwargs.pop("smiles", KETOCONAZOLE_SMILES), 442, predict_affinity=False, **kwargs)
    return yaml.safe_load(text)["constraints"]


def test_heme_donor_is_the_free_imidazole_nitrogen_not_the_substituted_one():
    from rdkit import Chem

    smiles = "Cn1ccnc1"
    idx, name = heme_donor_atom(smiles)
    atom = Chem.MolFromSmiles(smiles).GetAtomWithIdx(idx)
    assert atom.GetSymbol() == "N" and atom.GetDegree() == 2  # the N-methyl nitrogen has three neighbours
    assert name == boltz_atom_name(smiles, idx) and name.startswith("N")


def test_heme_donor_prefers_the_less_crowded_pyridine_and_is_none_without_a_motif():
    from rdkit import Chem

    smiles = "Cc1ncccc1-c1ccncc1"  # a 2-methylpyridine next to a bare 4-pyridyl
    mol = Chem.MolFromSmiles(smiles)
    idx, _ = heme_donor_atom(smiles)
    ring_neighbours = [nb for nb in mol.GetAtomWithIdx(idx).GetNeighbors()]
    # the bare pyridine's nitrogen has no carbon next to it that carries a methyl group
    assert not any(n.GetSymbol() == "C" and any(m.GetTotalNumHs() == 3 for m in n.GetNeighbors()) for n in ring_neighbours)
    assert heme_donor_atom("CCO") is None


def test_boltz_atom_names_are_unique_within_a_ligand():
    from rdkit import Chem

    n = Chem.AddHs(Chem.MolFromSmiles(KETOCONAZOLE_SMILES)).GetNumAtoms()
    names = [boltz_atom_name(KETOCONAZOLE_SMILES, i) for i in range(n)]
    assert len(set(names)) == n


def test_donor_restraint_adds_a_bond_or_contact_to_the_heme_iron_and_is_off_by_default():
    default = _constraints()
    assert len(default) == 1 and "bond" in default[0]  # only the Cys-heme bond

    bond = _constraints(donor_restraint="bond")
    assert len(bond) == 2
    assert bond[1]["bond"]["atom2"] == ["C", 1, "FE"] and bond[1]["bond"]["atom1"][:2] == ["B", 1]

    contact = _constraints(donor_restraint="contact")[1]["contact"]
    assert contact["token2"] == ["C", "FE"] and contact["max_distance"] == 4 and contact["force"] is False


def test_donor_restraint_is_skipped_for_a_ligand_without_a_donor_and_rejects_bad_settings():
    assert len(_constraints(smiles="CCO", donor_restraint="bond")) == 1
    with pytest.raises(ValueError, match="predict_affinity=False"):
        build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 442, predict_affinity=True, donor_restraint="bond")
    with pytest.raises(ValueError, match="donor_restraint must be one of"):
        _constraints(donor_restraint="glue")


def test_protein_msa_is_written_under_the_protein_chain_only_when_given(tmp_path):
    msa = tmp_path / "p.csv"
    msa.write_text("key,sequence\n-1,AAA\n")
    with_msa = yaml.safe_load(build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 442, predict_affinity=False, protein_msa=msa))
    assert with_msa["sequences"][0]["protein"]["msa"] == str(msa.resolve())
    without = yaml.safe_load(build_boltz_yaml(CYP3A4_SEQUENCE, KETOCONAZOLE_SMILES, 442, predict_affinity=False))
    assert "msa" not in without["sequences"][0]["protein"]


def test_template_entry_may_name_the_chain_of_a_multi_chain_file(tmp_path):
    from cypmode.validation.build_input import (
        build_boltz_yaml,
        find_coordinating_cysteine,
    )
    from cypmode.validation.constants import CYP3A4_SEQUENCE

    plain, trimer = tmp_path / "1ABC.cif", tmp_path / "38FU.cif"
    cys = find_coordinating_cysteine(CYP3A4_SEQUENCE)
    text = build_boltz_yaml(CYP3A4_SEQUENCE, "CCO", cys, predict_affinity=False, templates=[plain, (trimer, "A")])
    block = text.split("templates:\n", 1)[1]
    assert f"  - cif: {plain.resolve()}\n    chain_id: A\n  - cif: {trimer.resolve()}\n    chain_id: A\n    template_id: A\n" in block
