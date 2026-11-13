"""Create a development-only, fixed-point flight cache from original NASA files.

No official-test array, virtual sensor, health state, or health parameter is read.
Reading the source file for a cryptographic digest does not decode its datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


FILES = {"DS02": "N-CMAPSS_DS02-006.h5", "DS05": "N-CMAPSS_DS05.h5",
         "DS06": "N-CMAPSS_DS06.h5", "DS07": "N-CMAPSS_DS07.h5"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump_new(path, value):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class DevelopmentSource:
    """All decoded dataset access passes through this closed allowlist."""
    ALLOWED = frozenset(("W_dev", "X_s_dev", "A_dev", "Y_dev"))

    def __init__(self, path):
        import h5py
        self._file = h5py.File(path, "r")
        self.access = []

    def read(self, key, selection):
        if key not in self.ALLOWED:
            raise PermissionError(f"forbidden development access: {key}")
        if key == "A_dev":
            # Never decode Fc or hs, even as an unused extra column.
            if not (isinstance(selection, tuple) and len(selection) == 2
                    and selection[1] == slice(0, 2)):
                raise PermissionError("A_dev is restricted to unit/cycle columns")
        self.access.append((key, repr(selection)))
        return self._file[key][selection]

    def close(self):
        self._file.close()


def prepare(source, destination, subset, points=512):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / f"{subset}_dev.npz"
    manifest = destination / f"{subset}_dev_manifest.json"
    if cache.exists() or manifest.exists():
        raise FileExistsError(f"immutable cache already exists: {cache}")
    start_time = time.time()
    source = Path(source)
    source_sha = digest(source)
    dev = DevelopmentSource(source)
    try:
        a = np.asarray(dev.read("A_dev", (slice(None), slice(0, 2))), dtype=np.int64)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("bad unit/cycle metadata shape")
        boundaries = np.r_[0, np.where(np.any(a[1:] != a[:-1], axis=1))[0] + 1, len(a)]
        raw, labels, units, cycles, indices, n_original = [], [], [], [], [], []
        seen = set()
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            key = tuple(map(int, a[lo]))
            if key in seen:
                raise AssertionError(f"noncontiguous flight: {key}")
            seen.add(key)
            count = int(hi - lo)
            if count < 2:
                raise AssertionError(f"invalid short flight {key}: {count}")
            selected = np.rint(np.linspace(0, count - 1, points)).astype(np.int64)
            w = np.asarray(dev.read("W_dev", slice(int(lo), int(hi))), dtype=np.float32)
            xs = np.asarray(dev.read("X_s_dev", slice(int(lo), int(hi))), dtype=np.float32)
            y = np.asarray(dev.read("Y_dev", slice(int(lo), int(hi))), dtype=np.float64).ravel()
            if w.shape != (count, 4) or xs.shape != (count, 14):
                raise AssertionError(f"unexpected observed-input shape for {key}")
            if not np.isfinite(w).all() or not np.isfinite(xs).all():
                raise AssertionError(f"nonfinite observed inputs for {key}")
            if not np.isfinite(y).all() or not np.allclose(y, y[0], rtol=0, atol=1e-8):
                raise AssertionError(f"RUL must be constant inside flight {key}")
            raw.append(np.concatenate((w[selected], xs[selected]), axis=1))
            labels.append(float(y[0])); units.append(key[0]); cycles.append(key[1])
            indices.append(lo + selected); n_original.append(count)
        order = np.lexsort((cycles, units))
        arrays = {"raw": np.stack(raw)[order], "y": np.asarray(labels, np.float32)[order],
                  "unit": np.asarray(units, np.int64)[order],
                  "cycle": np.asarray(cycles, np.int64)[order],
                  "source_indices": np.stack(indices)[order],
                  "original_points": np.asarray(n_original, np.int64)[order]}
        with open(cache, "xb") as f:
            np.savez_compressed(f, **arrays)
        summary = {"subset": subset, "source_name": source.name,
                   "source_bytes": source.stat().st_size, "source_sha256": source_sha,
                   "cache_sha256": digest(cache), "points_per_flight": points,
                   "sampling": "rounded uniform positions including both flight endpoints; repeats if short",
                   "input_columns": ["alt", "Mach", "TRA", "T2", "Wf", "Nf", "Nc", "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50"],
                   "prediction_time": "end of current flight; only current and previous flights",
                   "label": "original Y_dev in flight cycles; no hs-derived plateau",
                   "units": sorted(set(units)), "flights": len(raw),
                   "rul_min_max": [float(arrays["y"].min()), float(arrays["y"].max())],
                   "decoded_keys": sorted(set(k for k, _ in dev.access)),
                   "A_columns_decoded": ["unit", "cycle"],
                   "official_test_decoded": False, "duration_is_model_input": False,
                   "decoded_accesses": len(dev.access), "seconds": time.time() - start_time,
                   "per_unit_flights": {str(u): int(np.sum(arrays["unit"] == u)) for u in sorted(set(units))}}
        dump_new(manifest, summary)
        print(json.dumps({k: summary[k] for k in ("subset", "flights", "units", "source_sha256", "cache_sha256", "seconds")}), flush=True)
    finally:
        dev.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--subsets", nargs="+", choices=tuple(FILES), default=list(FILES))
    p.add_argument("--points", type=int, default=512)
    args = p.parse_args()
    if args.points != 512:
        raise ValueError("frozen development cache uses exactly 512 samples per flight")
    args.destination.mkdir(parents=True, exist_ok=True)
    with open(args.destination / "PREPARATION_CLAIM.json", "x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "subsets": args.subsets, "started": time.time()}, f)
    for subset in args.subsets:
        prepare(args.source_dir / FILES[subset], args.destination, subset, args.points)


if __name__ == "__main__":
    main()
