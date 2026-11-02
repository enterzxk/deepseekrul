"""C-MAPSS loading for the B line, numpy only.

The A-line pipeline lived in ``src/dataset.py`` and deliberately threw the three
operating-setting columns away.  The B line needs them, so this module keeps
them and adds the one thing the A line never needed: a condition assignment.

Design choices worth stating, because the rest of the line depends on them:

* Setting columns carry simulator jitter (177 distinct rows in FD002 for what
  are physically 6 setpoints), so conditions are recovered by k-means on the
  settings, fitted on training engines only.  ``fit_conditions`` returns the
  centres so a test file can be assigned with the same map.
* Sensors are the standard 14 informative channels plus the 7 that are usually
  dropped, i.e. all 21.  Deciding which channels are redundant is part of the
  research question, so pre-selecting 14 would beg it.  ``INFORMATIVE`` is kept
  for the comparison against the A-line pipeline.
* Row order inside an engine is cycle order; nothing here shuffles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "CMAPSS"

SETTINGS = ("os1", "os2", "os3")
ALL_SENSORS = tuple(range(1, 22))
INFORMATIVE = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)


@dataclass
class Frame:
    """One C-MAPSS split, flattened to cycle rows."""

    subset: str
    unit: np.ndarray      # (N,) int engine id, 1-based as in the file
    cycle: np.ndarray     # (N,) int
    u: np.ndarray         # (N,3) operating settings as recorded
    x: np.ndarray         # (N,21) sensors as recorded
    rul: np.ndarray       # (N,) cycles to failure; for test files the file value

    def __len__(self) -> int:
        return len(self.unit)

    def engines(self) -> np.ndarray:
        return np.unique(self.unit)

    def index_of(self, unit_id: int) -> np.ndarray:
        return np.flatnonzero(self.unit == unit_id)

    def lengths(self) -> dict[int, int]:
        out = {}
        for i in self.engines():
            out[int(i)] = int(self.cycle[self.unit == i].max())
        return out


def _read(path: Path) -> np.ndarray:
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 26:
        raise ValueError(f"{path}: expected 26 columns, got {a.shape}")
    return a


def load(subset: str, split: str = "train", cap: int | None = None) -> Frame:
    """Read one split.  ``cap`` clips the label at the usual 125 plateau."""
    path = DATA / f"{split}_{subset}.txt"
    a = _read(path)
    unit = a[:, 0].astype(int)
    cycle = a[:, 1].astype(int)
    u = a[:, 2:5]
    x = a[:, 5:26]
    if split == "train":
        last = {int(e): int(cycle[unit == e].max()) for e in np.unique(unit)}
        rul = np.array([last[int(e)] - int(c) for e, c in zip(unit, cycle)], float)
    else:
        r = np.loadtxt(DATA / f"RUL_{subset}.txt", dtype=np.float64).ravel()
        table = {i + 1: float(v) for i, v in enumerate(r)}
        rul = np.array([table[int(e)] + (int(c) - int(cycle[unit == e].max()))
                        for e, c in zip(unit, cycle)], float)
    if cap is not None:
        rul = np.minimum(rul, float(cap))
    return Frame(subset=subset, unit=unit, cycle=cycle, u=u, x=x, rul=rul)


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #

def _kmeans(a: np.ndarray, k: int, seed: int = 0, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd k-means with k-means++ seeding.  Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    n = len(a)
    centres = np.empty((k, a.shape[1]))
    centres[0] = a[rng.integers(n)]
    d2 = ((a - centres[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        centres[j] = a[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((a - centres[j]) ** 2).sum(1))
    lab = np.zeros(n, int)
    for _ in range(iters):
        dist = ((a[:, None, :] - centres[None]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                centres[j] = a[m].mean(0)
    return centres, lab


def fit_conditions(frames: list[Frame], k: int = 6, seed: int = 0) -> np.ndarray:
    """Cluster the setting columns of the given *training* frames."""
    u = np.concatenate([f.u for f in frames], axis=0)
    centres, _ = _kmeans(u, k, seed=seed)
    # order by the first setting then the second, so labels are stable
    order = np.lexsort((centres[:, 2], centres[:, 1], centres[:, 0]))
    return centres[order]


def assign_conditions(f: Frame, centres: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Nearest-centre assignment under the fitted setting scale."""
    if scale is None:
        scale = np.ones(centres.shape[1])
    d = ((f.u[:, None, :] - centres[None]) / scale[None, None, :]) ** 2
    return d.sum(-1).argmin(1)


def setting_scale(frames: list[Frame]) -> np.ndarray:
    u = np.concatenate([f.u for f in frames], axis=0)
    s = u.std(0)
    return np.where(s > 0, s, 1.0)


if __name__ == "__main__":
    for sub in ("FD002", "FD004"):
        tr = load(sub, "train")
        te = load(sub, "test")
        sc = setting_scale([tr])
        cen = fit_conditions([tr], k=6)
        lab = assign_conditions(tr, cen, sc)
        # how tight are the clusters, in units of the setting scale?
        dist = np.sqrt((((tr.u - cen[lab]) / sc) ** 2).sum(1))
        print(f"{sub}: train engines={len(tr.engines())} rows={len(tr)} test rows={len(te)}")
        print(f"  centres (scaled):\n{np.round(cen / sc, 2)}")
        print("  cluster sizes:", np.bincount(lab, minlength=6).tolist())
        print(f"  assignment distance (scaled): median={np.median(dist):.3f} "
              f"p99={np.percentile(dist, 99):.3f}")
        per_eng = np.array([np.bincount(lab[tr.unit == e], minlength=6).argmax()
                            for e in tr.engines()])
        print("  engines per condition:", np.bincount(per_eng, minlength=6).tolist())
        print("  label noise within engine:",
              float(np.mean([1 - np.bincount(lab[tr.unit == e], minlength=6).max()
                             / (tr.unit == e).sum() for e in tr.engines()])))
