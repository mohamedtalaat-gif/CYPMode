# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

FIGURES = Path("docs/figures")
sys.path[:0] = [str(FIGURES), str(FIGURES.parent.parent)]

from catalog import REGISTRY, caption_text
from gen import structure_pharmacophore as figures

from cypmode.challenge import pocket_pharmacophore as pp


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A tree with the output files of jobs 273, 274 and 925 made of synthetic numbers, so that the three figures are drawn without any real result."""
    monkeypatch.setattr(figures, "ROOT", tmp_path)
    out = tmp_path / figures.OUT
    (out / "pharmacophore_official20").mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 40
    pd.DataFrame({"pdb_id": "X", "comp_id": [f"C{i}" for i in range(n)], "n_heavy": 30, "rotatable": 5, "nearest_rmsd": rng.uniform(0.3, 3.0, n), "nearest_energy": 3.0,
                  "needed_energy": np.where(rng.random(n) < 0.6, rng.uniform(0, 25, n), np.nan), "ensemble_rmsd": rng.uniform(0.5, 4.0, n)}).to_csv(out / "conformer_bias.csv", index=False)
    pd.DataFrame({"pdb_id": "X", "comp_id": "C", "pocket_nearest_rmsd": rng.uniform(0.1, 0.5, n), "outside": rng.uniform(0, 0.4, n)}).to_csv(out / "site_representation_crystal.csv", index=False)
    pd.DataFrame({"compound": "b", "run": "r", "sample": range(30), "true_rmsd": rng.uniform(0.5, 7, 30), "pocket_nearest_rmsd": rng.uniform(0.1, 0.5, 30), "outside": rng.uniform(0, 0.3, 30)}).to_csv(out / "site_representation_poses.csv", index=False)
    (out / "site_representation_flags.json").write_text(json.dumps({"pocket_rmsd_flag": 0.4, "support_cut": 0.35, "core_residues": 62, "percentile": 95}))
    fields = {name: pp.gaussian_field(rng.normal(size=(20, 3)) * 3 + [0, 0, 7], np.ones(20), 1.5) for name in (*pp.FAMILIES, "occupancy")}
    (tmp_path / Path(figures.FIELDS).parent).mkdir(parents=True)
    np.savez_compressed(tmp_path / figures.FIELDS, **fields)
    names = [f"OCNT-{i}" for i in range(20)]
    pd.DataFrame({"ligand": names, "recall": rng.uniform(0.1, 0.6, 20), "pool_recall_mean": rng.uniform(0.1, 0.6, 20), "pool_recall_sd": rng.uniform(0.0, 0.1, 20), "beyond_pocket_flag": [False] * 19 + [True],
                  "beyond_support_cut": False}).to_csv(out / "pharmacophore_official20" / "summary.csv", index=False)
    pd.DataFrame([{"ligand": n, "family": f, "x": rng.normal() * 3, "y": rng.normal() * 3 - 3, "z": 6 + rng.normal() * 2} for n in names for f in ("Donor", "Acceptor", "Aromatic", "Hydrophobe")]).to_csv(out / "pharmacophore_official20" / "features.csv", index=False)
    pd.DataFrame({"family": ["Aromatic", "Hydrophobe"], "x": [1.0, 4.0], "y": [-3.0, -2.0], "z": [5.0, 6.0], "ligands": [15, 16]}).to_csv(out / "pharmacophore_official20" / "common_sites.csv", index=False)
    matrix = np.corrcoef(rng.normal(size=(20, 30)))
    pd.DataFrame(np.abs(matrix), index=names, columns=names).to_csv(out / "pharmacophore_official20" / "similarity.csv")
    return tmp_path


@pytest.mark.parametrize("figure_id", ["struct_conformer_window", "struct_site_check", "struct_pharm_official20"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_each_figure_of_s8_is_drawn_from_its_files_in_both_themes_with_a_caption_from_the_data(root, figure_id, theme):
    import matplotlib.pyplot as plt

    entry = REGISTRY[figure_id]
    fig = entry.make(theme)
    assert len(fig.axes) >= 2
    plt.close(fig)
    assert len(caption_text(entry)) > 200 and entry.group == "structure"


def test_the_official_figure_names_only_files_of_the_held_back_folder_and_the_public_ones_name_none_of_it():
    assert all("pharmacophore_official20" in path or path.endswith(".npz") for path in REGISTRY["struct_pharm_official20"].data)
    for figure_id in ("struct_conformer_window", "struct_site_check"):
        assert not any("official20" in path for path in REGISTRY[figure_id].data)
