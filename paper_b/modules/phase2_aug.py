"""Phase 2 M0+Aug: static 30% training-window sensor augmentation.

This wrapper monkey-patches ``prepare_train`` before ``run_v0.main`` so the
frozen trainer sees an augmented training split (30% of windows: 1-2 channels
get Gaussian noise sigma~U[0.5,2.5] or are zeroed). It is a static augmented
training set, not per-batch augmentation; documented as a Phase 2 deviation.
"""
import numpy as np

import paper_b.training.train as T
from paper_b.hc import run_v0

_orig_prepare = T.prepare_train


def _prepare_aug(*args, **kwargs):
    ds = _orig_prepare(*args, **kwargs)
    tr = ds['train']
    x = tr['x'].copy()
    rng = np.random.default_rng(20261120)
    n = x.shape[0]
    sel = rng.random(n) < 0.30
    for b in np.where(sel)[0]:
        k = int(rng.integers(1, 3))
        chans = rng.choice(x.shape[-1], size=k, replace=False)
        for j in chans:
            if rng.random() < 0.5:
                x[b, :, j] += rng.normal(0.0, rng.uniform(0.5, 2.5), size=x.shape[1])
            else:
                x[b, :, j] = 0.0
    tr['x'] = x
    return ds


T.prepare_train = _prepare_aug

if __name__ == '__main__':
    run_v0.main()
