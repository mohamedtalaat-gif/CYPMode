import shutil
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

pytest.importorskip("Bio")

from cypmode.validation.structures import (
    _mol_from_direct_atom_correspondence,
    load_trunk_embeddings,
    pooled_trunk_embedding,
    summarize_compound,
)


class _FakeAtom:
    """Minimal stand-in for a Bio.PDB Atom -- _mol_from_direct_atom_correspondence
    only ever touches .element and .get_coord() on the ligand_atoms it's given.
    """

    def __init__(self, element: str, coord):
        self.element = element
        self._coord = np.array(coord, dtype=float)

    def get_coord(self):
        return self._coord


def test_mol_from_direct_atom_correspondence_builds_a_valid_mol_when_order_matches():
    # Ethylamine (C-C-N), real bond lengths (~1.52, ~1.47 A) at this
    # correspondence -- the case this fallback exists for: geometry that
    # would confuse a distance-based bond-perception heuristic (see below)
    # but is fine once bonds are taken from the template instead of re-derived.
    template = Chem.MolFromSmiles("CCN")
    ligand_atoms = [
        _FakeAtom("C", [0.0, 0.0, 0.0]),
        _FakeAtom("C", [1.52, 0.0, 0.0]),
        _FakeAtom("N", [1.52, 1.47, 0.0]),
    ]
    mol = _mol_from_direct_atom_correspondence(template, ligand_atoms)
    assert mol is not None
    conf = mol.GetConformer()
    np.testing.assert_allclose(list(conf.GetAtomPosition(2)), [1.52, 1.47, 0.0])


def test_mol_from_direct_atom_correspondence_returns_none_when_element_order_differs():
    template = Chem.MolFromSmiles("CCN")
    ligand_atoms = [
        _FakeAtom("N", [0.0, 0.0, 0.0]),  # wrong order: N first, not last
        _FakeAtom("C", [1.52, 0.0, 0.0]),
        _FakeAtom("C", [1.52, 1.47, 0.0]),
    ]
    assert _mol_from_direct_atom_correspondence(template, ligand_atoms) is None


def test_mol_from_direct_atom_correspondence_returns_none_when_a_template_bond_length_is_unreasonable():
    # Real regression case, generalized: element order matches, but two
    # atoms that are actually bonded in the template (0-1) sit far too close
    # together in this geometry -- exactly the situation where trusting an
    # unverified index correspondence would silently build a wrong molecule.
    # Real proximity-bonding fallback data (OCNT-2311782) failed for the
    # opposite reason -- two NON-bonded atoms sitting too close, which is
    # what makes distance-based bond perception misfire in the first place;
    # this checks the same underlying safeguard from the other direction.
    template = Chem.MolFromSmiles("CCN")
    ligand_atoms = [
        _FakeAtom("C", [0.0, 0.0, 0.0]),
        _FakeAtom("C", [0.1, 0.0, 0.0]),  # bonded to atom 0 in the template, but only 0.1 A away
        _FakeAtom("N", [1.52, 1.47, 0.0]),
    ]
    assert _mol_from_direct_atom_correspondence(template, ligand_atoms) is None

KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
VARDENAFIL_SMILES = "CCCc1nc(C)c2c(=O)nc(-c3cc(S(=O)(=O)N4CCN(CC)CC4)ccc3OCC)[nH]n12"
OCNT_2395436_SMILES = "CC(C)(C)NS(=O)(=O)C1(CNC2=C(Br)C=NC=C2C#N)CCC1"

RESULT_DIR = Path("data/structure_track/outputs/out/boltz_results_cyp3a4_ketoconazole")
CIF_PATH = RESULT_DIR / "predictions" / "cyp3a4_ketoconazole" / "cyp3a4_ketoconazole_model_0.cif"

pytestmark = pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)


def test_summarize_ketoconazole_coordinates():
    # Real, known-correct answer: the imidazole nitrogen (CIF atom N32) sits
    # 2.05 A from the heme iron -- confirmed independently by
    # github.com/PatWalters/cyp-heatmap's ProLIF analysis of the same
    # compound (see README.md's Independent cross-check section).
    result = summarize_compound(RESULT_DIR, KETOCONAZOLE_SMILES)
    assert result["compound"] == "cyp3a4_ketoconazole"
    assert result["coordinated"] is True
    assert result["closest_motif_nitrogen_distance_angstrom"] == pytest.approx(2.05, abs=0.01)
    assert result["motif_nitrogen_distances"][0]["atom"] == "N32"
    assert result["motif_nitrogen_distances"][0]["motif"] == "imidazole_Nsub"
    assert "affinity_pred_value" in result


def test_motif_distance_ignores_closer_non_motif_nitrogen():
    # The regression this module exists to prevent: vardenafil's
    # geometrically closest ligand nitrogen (N50, ~4.84 A) is not part of
    # any matched Type II motif. The motif-restricted distance must not
    # silently fall back to it.
    result_dir = Path("data/structure_track/outputs/out/boltz_results_cyp3a4_vardenafil")
    if not (result_dir / "predictions" / "cyp3a4_vardenafil" / "cyp3a4_vardenafil_model_0.cif").exists():
        pytest.skip("requires a local Boltz-2 run for vardenafil")

    result = summarize_compound(result_dir, VARDENAFIL_SMILES)
    motif_atoms = {d["atom"] for d in result["motif_nitrogen_distances"]}
    assert "N50" not in motif_atoms
    assert result["closest_motif_nitrogen_distance_angstrom"] > 4.84
    assert result["coordinated"] is False

    all_atoms = {d["atom"] for d in result["all_ligand_nitrogen_distances"]}
    assert "N50" in all_atoms  # still visible for manual review, just not driving the call


OCNT_2311782_SMILES = "CC(C1=CC=CC(C#N)=C1)N1CCN(C(=O)NC2CCCCC2)CC1"
OCNT_2311782_RESULT_DIR = Path("data/structure_track/outputs/out/boltz_results_cyp3a4_OCNT-2311782")


@pytest.mark.skipif(
    not (OCNT_2311782_RESULT_DIR / "predictions" / "cyp3a4_OCNT-2311782" / "cyp3a4_OCNT-2311782_model_0.cif").exists(),
    reason="requires a local Boltz-2 run for OCNT-2311782",
)
def test_summarize_compound_handles_a_real_proximity_bonding_failure():
    # Real regression: this compound's predicted geometry makes proximity
    # bonding (and RDKit's own DetermineConnectivity, confirmed as an
    # equally-fooled alternative) perceive 29 bonds where the real molecule
    # has 27 -- AssignBondOrdersFromTemplate then raised AtomValenceException
    # ("atom # 10 N, 4, is greater than permitted"). Confirmed the direct
    # atom-correspondence fallback resolves it: this compound has no Type II
    # motif at all (a real, valid "not coordinated" result, not a symptom of
    # the bug), so the check here is just that it doesn't raise.
    result = summarize_compound(OCNT_2311782_RESULT_DIR, OCNT_2311782_SMILES, expect_affinity=False)
    assert result["compound"] == "cyp3a4_OCNT-2311782"
    assert result["coordinated"] is False
    assert result["motif_nitrogen_distances"] == []


OCNT_2395436_RESULT_DIR = Path("data/structure_track/outputs/out/boltz_results_cyp3a4_OCNT-2395436")


@pytest.mark.skipif(
    not OCNT_2395436_RESULT_DIR.exists(),
    reason="requires a local Boltz-2 run for OCNT-2395436",
)
def test_summarize_compound_expect_affinity_false_skips_affinity_gracefully(tmp_path):
    # Real completed result dir, copied so the affinity JSON can be removed
    # without touching the actual data -- simulates a predict_affinity=False
    # run (which never produces that file) using real structure output.
    copy_dir = tmp_path / "boltz_results_cyp3a4_OCNT-2395436"
    shutil.copytree(OCNT_2395436_RESULT_DIR, copy_dir)
    predictions_dir = copy_dir / "predictions" / "cyp3a4_OCNT-2395436"
    for affinity_file in predictions_dir.glob("affinity_*.json"):
        affinity_file.unlink()

    result = summarize_compound(copy_dir, OCNT_2395436_SMILES, expect_affinity=False)
    assert "affinity_pred_value" not in result
    assert "affinity_probability_binary" not in result
    assert "confidence_score" in result  # unaffected, still present

    with pytest.raises(FileNotFoundError):
        summarize_compound(copy_dir, OCNT_2395436_SMILES, expect_affinity=True)


def test_pooled_trunk_embedding_gives_a_fixed_size_vector_regardless_of_token_count():
    # s: (n_tokens, d_s), z: (n_tokens, n_tokens, d_z) -- different token
    # counts for two synthetic complexes must still produce comparable,
    # same-length pooled vectors (d_s + d_z each), since that's the whole
    # point of pooling before using this as an ML feature across compounds.
    small = {"s": np.random.default_rng(0).normal(size=(50, 8)), "z": np.random.default_rng(0).normal(size=(50, 50, 4))}
    large = {"s": np.random.default_rng(1).normal(size=(300, 8)), "z": np.random.default_rng(1).normal(size=(300, 300, 4))}
    pooled_small = pooled_trunk_embedding(small)
    pooled_large = pooled_trunk_embedding(large)
    assert pooled_small.shape == (12,)  # 8 (s) + 4 (z)
    assert pooled_large.shape == pooled_small.shape


def test_load_trunk_embeddings_reads_real_npz_shape(tmp_path):
    s = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    z = np.random.default_rng(0).normal(size=(20, 20, 4)).astype(np.float32)
    np.savez_compressed(tmp_path / "embeddings_test.npz", s=s, z=z)

    embeddings = load_trunk_embeddings(tmp_path)
    assert embeddings["s"].shape == (20, 8)
    assert embeddings["z"].shape == (20, 20, 4)


def test_load_trunk_embeddings_raises_clearly_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="write_embeddings"):
        load_trunk_embeddings(tmp_path)


@pytest.mark.skipif(
    not CIF_PATH.exists(),
    reason="requires a local Boltz-2 run; see README.md's Validation section",
)
def test_summarize_compound_expect_trunk_embedding_true_includes_pooled_vector(tmp_path):
    # Real completed structure output, with a synthetic embeddings_*.npz
    # added -- no real run has used --write_embeddings yet (see
    # module docstring for why the flag+file format are trusted regardless:
    # confirmed directly in third_party/boltz-main's own source).
    copy_dir = tmp_path / "boltz_results_cyp3a4_ketoconazole"
    shutil.copytree(RESULT_DIR, copy_dir)
    predictions_dir = copy_dir / "predictions" / "cyp3a4_ketoconazole"
    s = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
    z = np.random.default_rng(0).normal(size=(100, 100, 4)).astype(np.float32)
    np.savez_compressed(predictions_dir / "embeddings_cyp3a4_ketoconazole.npz", s=s, z=z)

    result = summarize_compound(copy_dir, KETOCONAZOLE_SMILES, expect_trunk_embedding=True)
    assert "pooled_trunk_embedding" in result
    assert isinstance(result["pooled_trunk_embedding"], list)  # JSON-serializable, not a raw ndarray
    assert len(result["pooled_trunk_embedding"]) == 12
