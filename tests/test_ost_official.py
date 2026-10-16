import pandas as pd

from cypmode.validation.ost_official import finished_keys


def test_the_poses_already_scored_are_read_by_column_name_even_when_the_table_is_empty():
    assert finished_keys(pd.DataFrame(columns=["compound", "engine", "sample"])) == set()
    done = pd.DataFrame({"compound": ["a", "b"], "engine": ["boltz", "boltz"], "sample": [0, 1]})
    assert finished_keys(done) == {("a", "boltz", 0), ("b", "boltz", 1)}


def test_regate_reads_the_restraint_effect_with_each_measure():
    import numpy as np
    import pandas as pd

    from cypmode.validation import restraint_expansion as expansion
    from cypmode.validation.ost_regate import regate

    ids = [f"L{i}" for i in range(8)]
    ligands = pd.DataFrame({"comp_id": ids, "stratum": ["primary"] * 8})
    rows = []
    for i, c in enumerate(ids):
        for suffix, shift in (("", 0.0), (expansion.RESTRAINED_SUFFIX, 0.1 + 0.01 * i)):
            for sample in (0, 1):
                rows.append({"compound": f"exp-{c}{suffix}", "engine": "boltz", "sample": sample, "lddt_pli": 0.4 + shift + 0.002 * sample})
    poses = pd.DataFrame(rows)
    ost = poses.rename(columns={"lddt_pli": "ost"}).assign(ost=lambda t: t["ost"] - 0.06)  # a constant offset: the verdict must not change
    stats = regate(poses, ost, ligands, samples=2, alpha=0.05)
    assert stats["ours"]["n"] == stats["openstructure"]["n"] == 8
    assert np.isclose(stats["ours"]["mean"], stats["openstructure"]["mean"]) and stats["ours"]["confirmed"] and stats["openstructure"]["confirmed"]
