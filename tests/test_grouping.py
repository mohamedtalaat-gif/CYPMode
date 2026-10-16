
from cypmode.challenge.grouping import butina_groups, morgan_bit_matrix, scaffold_groups


def test_butina_puts_near_duplicates_together_and_distant_compounds_apart():
    smiles = ["c1ccccc1CCO", "c1ccccc1CCN", "c1ccccc1CCCO", "CC(C)(C)C(=O)OC(C)(C)C", "CC(C)(C)C(=O)OC(C)(C)CC", "C1CCNCC1"]
    labels = butina_groups(morgan_bit_matrix(smiles), similarity_cutoff=0.3)
    assert labels[0] == labels[1] == labels[2] or len({labels[0], labels[1], labels[2]}) < 3
    assert labels[3] == labels[4] and labels[3] != labels[0] and labels[5] not in (labels[0], labels[3])
    assert (labels >= 0).all()


def test_a_stricter_similarity_cutoff_gives_more_and_smaller_clusters():
    smiles = ["c1ccccc1CCO", "c1ccccc1CCN", "c1ccccc1CCCO", "c1ccccc1CCCN", "CCCCCCCCO", "CCCCCCCCN"]
    bits = morgan_bit_matrix(smiles)
    assert len(set(butina_groups(bits, 0.9))) >= len(set(butina_groups(bits, 0.2)))


def test_scaffold_groups_share_an_id_across_substituents_and_generic_collapses_heteroatoms():
    smiles = ["c1ccccc1CC", "c1ccccc1CO", "c1ccncc1CC", "CCCC", "CCCCC"]
    exact = scaffold_groups(smiles, generic=False)
    assert exact[0] == exact[1] and exact[0] != exact[2] and exact[3] == exact[4]
    generic = scaffold_groups(smiles, generic=True)
    assert generic[0] == generic[2]  # benzene and pyridine share a carbon skeleton
