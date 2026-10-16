import numpy as np
import pandas as pd

from cypmode.validation.cryoem_templates import ENTRIES, build_cryoem_table, pockets_of

RNG = np.random.default_rng(1)
BASE = {str(r): RNG.normal(scale=8.0, size=3).tolist() for r in range(100, 140)}


def _record(tmp_path):
    shifted = {
        pdb_id: {r: (np.asarray(v) + np.array([i * (1 if int(r) % 2 else -1), 0.0, 0.0])).tolist() for r, v in BASE.items()}
        for i, pdb_id in enumerate(ENTRIES)
    }
    for pdb_id in ENTRIES:
        (tmp_path / f"{pdb_id}.cif").write_text("x")
    ligands = [{"pdb_id": "38FV", "smiles": "CCCCCCCCO"}, {"pdb_id": "38FY", "smiles": "c1ccccc1C(=O)N"}]
    return {"pockets": shifted, "ligands": ligands}


def test_pockets_become_arrays_and_empty_pockets_are_dropped():
    pockets = pockets_of({"pockets": {"38FU": {"100": [1, 2, 3]}, "38FV": {}}})
    assert list(pockets) == ["38FU"] and pockets["38FU"][100].tolist() == [1.0, 2.0, 3.0]


def test_table_has_one_row_per_run_named_em_k_with_a_template(tmp_path):
    ligands = pd.DataFrame({"Molecule_Name": ["bench-X"], "SMILES": ["CCN(CC)CCCC"]})
    table = build_cryoem_table(ligands, _record(tmp_path), n_runs=3, directory=tmp_path)
    assert list(table.Molecule_Name) == ["bench-X-em1", "bench-X-em2", "bench-X-em3"]
    assert table.Template_PDB.nunique() == 3 and set(table.Template_PDB) <= set(ENTRIES)
    assert (table.Diffusion_Samples == 1).all()


def test_an_entry_whose_ligand_is_the_compound_is_never_its_template(tmp_path):
    ligands = pd.DataFrame({"Molecule_Name": ["bench-Y"], "SMILES": ["c1ccccc1C(=O)N"]})
    table = build_cryoem_table(ligands, _record(tmp_path), n_runs=5, directory=tmp_path)
    assert "38FY" not in set(table.Template_PDB) and len(table) == 4
