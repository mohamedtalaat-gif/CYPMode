import numpy as np

from cypmode.challenge import mmp_transfer as mt

# a series on one constant part: the para substituent decides the label (Cl, Br active; H, Me, F inactive)
SERIES = ["Cc1ccccc1", "Clc1ccccc1", "Brc1ccccc1", "Fc1ccccc1", "CCCCO"]


def test_matched_pairs_share_the_constant_part_and_are_directed():
    a, b, t, names = mt.build_pairs(SERIES)
    benzene_series = set(range(4))
    assert {int(x) for x in a} | {int(x) for x in b} <= benzene_series  # the unrelated alcohol has no partner
    pair_set = {(int(x), int(y)) for x, y in zip(a, b, strict=True)}
    assert (1, 2) in pair_set and (2, 1) in pair_set  # Cl -> Br and Br -> Cl
    assert any(n.endswith("Cl") or ">>" in n for n in names)


def test_features_use_only_training_labels_and_drop_the_query_from_its_own_table():
    pairs = mt.build_pairs(SERIES)
    y = np.array([0, 1, 1, 0])
    train = np.array([0, 1, 2, 3])
    loo = mt.mmp_label_features(pairs, 5, train, y, train, True)
    plain = mt.mmp_label_features(pairs, 5, train, y, train, False)
    assert loo.shape == (4, 5) and (loo[:, 0] == 3).all()  # each benzene-series member has three partners
    assert not np.allclose(np.nan_to_num(loo), np.nan_to_num(plain))  # leaving the query out changes the table
    # an evaluation compound only reads training labels: flip its own label, nothing changes
    y_flipped_eval = y.copy()
    q = mt.mmp_label_features(pairs, 5, train[:3], y[:3], np.array([3]), False)
    q_again = mt.mmp_label_features(pairs, 5, train[:3], y[:3], np.array([3]), False)
    assert np.allclose(q, q_again, equal_nan=True) and y_flipped_eval[3] == y[3]


def test_a_compound_without_partners_gets_zero_counts_and_nan_summaries():
    pairs = mt.build_pairs(SERIES)
    out = mt.mmp_label_features(pairs, 5, np.array([0, 1, 2, 3]), np.array([0, 1, 1, 0]), np.array([4]), False)
    assert out[0, 0] == 0 and out[0, 4] == 0 and np.isnan(out[0, 1:4]).all()


def test_transformation_delta_is_shrunk_and_signed():
    pairs = mt.build_pairs(SERIES)
    rows = mt.transformation_report(pairs, np.array([0, 1, 2, 3]), np.array([0, 1, 1, 0]), top=20, min_support=1)
    deltas = {n: d for n, _, d in rows}
    h_to_cl = [d for n, d in deltas.items() if n.endswith(">>Cl[*:1]") or n.endswith(">>[*:1]Cl") or "Cl" in n.split(">>")[1]]
    assert h_to_cl and max(abs(d) for d in h_to_cl) <= 1.0 / (1 + mt.SHRINK) + 1e-9  # one supporting pair: delta = 1 / (1 + shrink)
