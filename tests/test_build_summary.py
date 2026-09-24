from pathlib import Path

from cypmode.validation.build_summary import _read_ligand_smiles, discover_compounds

KETOCONAZOLE_SMILES = "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"


def test_read_ligand_smiles_from_real_input_file():
    smiles = _read_ligand_smiles(Path("data/boltz_test/cyp3a4_ketoconazole.yaml"))
    assert smiles == KETOCONAZOLE_SMILES


def test_discover_compounds_runs_without_a_local_boltz_run():
    # data/boltz_test/out/ is gitignored -- on a fresh checkout (and in CI)
    # this returns an empty list rather than raising.
    compounds = discover_compounds()
    assert isinstance(compounds, list)
