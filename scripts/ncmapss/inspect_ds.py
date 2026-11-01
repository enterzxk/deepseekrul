"""Inspect an N-CMAPSS h5 and report exactly what a RUL pipeline needs to know.

Run on the A40, where h5py/numpy exist.

    python scripts/ncmapss/inspect_ds.py ~/zxk/ncmapss/N-CMAPSS_DS02-006.h5

Prints array shapes, the auxiliary columns, the units present in dev and test, the
timestamps-per-cycle density, and the RUL range. Those four numbers decide the whole
design: units give the bootstrap block size, timestamps-per-cycle decides whether we
window on timestamps or aggregate to cycles, and the RUL range decides whether the
125-cycle cap convention carries over at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def s(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def main(path: str):
    p = Path(path)
    print(f"{p.name}  {p.stat().st_size/1e9:.3f} GB\n")
    with h5py.File(p, "r") as h:
        keys = list(h.keys())
        print("keys:", keys, "\n")

        for k in ["W_var", "X_s_var", "X_v_var", "T_var", "A_var"]:
            if k in h:
                v = np.array(h[k])
                print(f"{k:8s} ({len(v)}) {[s(x) for x in v]}")
        print()

        for split in ["dev", "test"]:
            A = np.array(h.get(f"A_{split}"))
            Y = np.array(h.get(f"Y_{split}"))
            Xs = np.array(h.get(f"X_s_{split}"))
            print(f"--- {split} ---")
            print(f"  X_s {Xs.shape} {Xs.dtype}   A {A.shape}   Y {Y.shape}")
            avar = [s(x) for x in np.array(h["A_var"])]
            iu = avar.index("unit") if "unit" in avar else 0
            ic = avar.index("cycle") if "cycle" in avar else 1
            units = np.unique(A[:, iu])
            print(f"  units ({len(units)}): {units.astype(int).tolist()}")
            y = Y.ravel()
            print(f"  RUL: min {y.min():.1f}  max {y.max():.1f}  "
                  f"mean {y.mean():.1f}  share<=125 {np.mean(y <= 125):.3f}  "
                  f"share<=0 {np.mean(y <= 0):.4f}")
            # timestamps per flight cycle
            per = []
            for u in units:
                m = A[:, iu] == u
                cyc = A[m, ic]
                n_uniq = len(np.unique(cyc))
                per.append(len(cyc) / max(n_uniq, 1))
            print(f"  timestamps/cycle: median {np.median(per):.0f}  "
                  f"range {min(per):.0f}-{max(per):.0f}")
            # cycles per unit
            cps = [len(np.unique(A[A[:, iu] == u, ic])) for u in units]
            print(f"  cycles/unit: min {min(cps)} max {max(cps)} "
                  f"median {int(np.median(cps))}")
            if "Fc" in avar:
                fc = A[:, avar.index("Fc")]
                print(f"  flight classes: {np.unique(fc).astype(int).tolist()}")
            print()


if __name__ == "__main__":
    main(sys.argv[1])
