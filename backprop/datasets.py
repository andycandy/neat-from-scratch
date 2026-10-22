"""Exact port of the dataset generators in hardmaru/backprop-neat-js `dataset.js`
(itself adapted from TensorFlow Playground).

Same as the JS: 200 points per split, noise 0.5, the same sampling formulas.
Differences (logged in DECISIONS.md, Unit 3):
  * the JS makes train + test only; we also make a validation split with the same
    generator, because fitness must not be computed on train or test data;
  * randomness comes from a seeded numpy Generator (JS used Math.random), so
    splits are fixed per seed;
  * the JS shuffles each list; we shuffle too (spiral/circle are generated in class order).
"""
import numpy as np

N_POINTS = 200      # nSize = nTestSize = 200 in the JS
NOISE = 0.5         # noiseLevel = 0.5 in the JS


def xor(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-5, 5, n) + rng.normal(0, noise, n)
    y = rng.uniform(-5, 5, n) + rng.normal(0, noise, n)
    label = ((x > 0) & (y > 0)) | ((x < 0) & (y < 0))
    return np.stack([x, y], 1), label.astype(np.float32)


def spiral(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pts, labels = [], []
    half = n // 2
    for delta_t, label in ((0.0, 1), (np.pi, 0)):          # flip = 0 in the JS
        i = np.arange(half)
        r = i / half * 6.0
        t = 1.75 * i / half * 2 * np.pi + delta_t
        x = r * np.sin(t) + rng.uniform(-1, 1, half) * noise
        y = r * np.cos(t) + rng.uniform(-1, 1, half) * noise
        pts.append(np.stack([x, y], 1))
        labels.append(np.full(half, label))
    return np.concatenate(pts), np.concatenate(labels).astype(np.float32)


def circle(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    radius = 5.0
    half = n // 2
    pts, labels = [], []
    for r_lo, r_hi in ((0.0, radius * 0.5), (radius * 0.75, radius)):   # inside, then ring
        r = rng.uniform(r_lo, r_hi, half)
        angle = rng.uniform(0, 2 * np.pi, half)
        x, y = r * np.sin(angle), r * np.cos(angle)
        label = (x * x + y * y < (radius * 0.5) ** 2)    # label from the NOISE-FREE point, as in JS
        nx = rng.uniform(-radius, radius, half) * noise / 3
        ny = rng.uniform(-radius, radius, half) * noise / 3
        pts.append(np.stack([x + nx, y + ny], 1))
        labels.append(label)
    return np.concatenate(pts), np.concatenate(labels).astype(np.float32)


GENERATORS = {"circle": circle, "xor": xor, "spiral": spiral}


def make_splits(name: str, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{'train': (X, y), 'val': (X, y), 'test': (X, y)}, each 200 points, shuffled, fixed per seed."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("train", "val", "test"):
        X, y = GENERATORS[name](N_POINTS, NOISE, rng)
        order = rng.permutation(len(y))
        out[split] = (X[order].astype(np.float32), y[order])
    return out
