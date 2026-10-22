import numpy as np

from backprop.datasets import GENERATORS, make_splits


def test_point_counts_and_both_classes():
    for name in GENERATORS:
        d = make_splits(name, 0)
        for split in ("train", "val", "test"):
            X, y = d[split]
            assert X.shape == (200, 2) and y.shape == (200,)
            assert 0 < y.mean() < 1                       # both classes present


def test_ranges_match_js():
    X, _ = make_splits("xor", 0)["train"]
    assert np.abs(X).max() < 5 + 4 * 0.5                  # uniform +-5 plus N(0, 0.5) noise
    X, y = make_splits("circle", 0)["train"]
    assert y.mean() == 0.5                                # 100 inside + 100 ring
    X, _ = make_splits("spiral", 0)["train"]
    assert np.abs(X).max() < 6 + 0.5 + 1e-6               # radius <= 6, noise <= 0.5


def test_splits_fixed_per_seed_and_different():
    a, b = make_splits("spiral", 3), make_splits("spiral", 3)
    assert np.array_equal(a["train"][0], b["train"][0])
    assert not np.array_equal(a["train"][0], a["val"][0])
