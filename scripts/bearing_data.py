"""Strict FEMTO snapshot loading + audit inventory.

Rules (enforced, no silent skipping)
  * the delimiter is detected from the file content (some bearings use ';');
  * every snapshot must have >= 6 columns and >= 1024 rows, be numeric and finite;
  * both vibration channels (cols 4 and 5) must have non-zero variance;
  * any violation is RECORDED and the snapshot is excluded explicitly; a bearing
    with any invalid snapshot is marked INVALID and must not enter training,
    early stopping or evaluation.

The inventory is written to docs/bearing_tim/data_inventory.csv and printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEMTO = Path("/home/yndx/femto")
MIN_ROWS = 1024
MIN_COLS = 6
CH_H, CH_V = 4, 5


class SnapshotError(RuntimeError):
    pass


def read_snapshot(path: Path) -> np.ndarray:
    """Parse one snapshot; raise SnapshotError with a reason on any violation."""
    try:
        raw = path.read_text(errors="strict")
    except Exception as e:  # noqa: BLE001
        raise SnapshotError(f"read_error:{type(e).__name__}") from e
    if not raw.strip():
        raise SnapshotError("empty_file")
    # delimiter: the first non-empty line decides
    first = raw.splitlines()[0]
    sep = ";" if first.count(";") >= first.count(",") else ","
    try:
        d = pd.read_csv(path, header=None, sep=sep).to_numpy(np.float64)
    except Exception as e:  # noqa: BLE001
        raise SnapshotError(f"parse_error:{type(e).__name__}") from e
    if d.ndim != 2 or d.shape[1] < MIN_COLS:
        raise SnapshotError(f"cols_{d.shape[1] if d.ndim == 2 else 'ragged'}")
    if d.shape[0] < MIN_ROWS:
        raise SnapshotError(f"rows_{d.shape[0]}")
    if not np.isfinite(d).all():
        raise SnapshotError("non_finite")
    for ch, name in ((CH_H, "h"), (CH_V, "v")):
        x = d[:, ch]
        if float(np.std(x)) <= 0.0:
            raise SnapshotError(f"{name}_constant")
    return d


def load_bearing_strict(path: Path) -> tuple[np.ndarray, dict]:
    files = sorted(path.glob("acc_*.csv"))
    if not files:
        raise SnapshotError("no_files")
    arrays, bad = [], []
    for f in files:
        try:
            arrays.append(read_snapshot(f))
        except SnapshotError as e:
            bad.append((f.name, str(e)))
    report = dict(bearing=path.name, raw=len(files), valid=len(arrays), invalid=len(bad),
                  reasons=dict(pd.Series([b[1] for b in bad]).value_counts()) if bad else {},
                  examples=[b[0] for b in bad[:5]])
    if bad:
        report["status"] = "INVALID"
        raise SnapshotError(json.dumps(report))
    arr = np.stack(arrays)                      # (N, rows, cols)
    report["status"] = "OK"
    report["rows"] = int(arr.shape[1])
    report["h_std_mean"] = float(arr[:, :, CH_H].std(axis=1).mean())
    report["v_std_mean"] = float(arr[:, :, CH_V].std(axis=1).mean())
    return arr, report


def inventory(root: Path = FEMTO) -> pd.DataFrame:
    rows = []
    for base in ["Learning_set", "Full_Test_Set"]:
        d = root / base
        if not d.exists():
            continue
        for b in sorted(p.name for p in d.iterdir() if p.is_dir()):
            name = f"{base[:4]}_{b}"
            try:
                _, rep = load_bearing_strict(d / b)
            except SnapshotError as e:
                try:
                    rep = json.loads(str(e))
                except Exception:  # noqa: BLE001
                    rep = dict(bearing=b, status="INVALID", reasons={"unparsed": 1})
                rep["name"] = name
            rep["name"] = name
            rows.append(rep)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(FEMTO))
    ap.add_argument("--out", default="docs/bearing_tim/data_inventory.csv")
    args = ap.parse_args()
    df = inventory(Path(args.root))
    cols = [c for c in ["name", "status", "raw", "valid", "invalid", "rows", "h_std_mean",
                        "v_std_mean", "reasons", "examples"] if c in df.columns]
    print(df[cols].to_string(index=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_bad = int((df.status != "OK").sum()) if "status" in df else 0
    print(f"\nbearings={len(df)} invalid={n_bad}  -> {out}")
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
