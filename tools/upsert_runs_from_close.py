#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upsert Gate A wave rows into registry/runs.csv from per-run close.json.

Design constraints (agreed 2026-11-12):
  * runs.csv has a SINGLE writer (the registry admin, sha256 optimistic
    concurrency). This tool is meant to be RUN BY THE ADMIN, not by anyone
    else, and it refuses to write if the file changed under it.
  * One row per CELL, and only that cell's status=ok run. Cell rule is the
    same one gate_a_judge.py uses:  HC-C-{variant}-{subset}-s101-W1{suffix}
    with suffix in ['', r2..r6]; the first close.json with status=ok wins.
    Rows for failed/aborted attempts are NOT inserted -- otherwise a table
    keyed on run_id would double count a cell.
  * Key = run_id. Same run_id -> UPDATE in place; new run_id -> APPEND.
    Every other line of runs.csv is left byte-identical.
  * Default is --dry-run. --write does an atomic, optimistically checked
    replace: sha256 before -> write .tmp -> re-check sha256 -> os.replace.
  * audit_verdict / audited_by are deliberately left EMPTY: this tool must
    never certify a run. in_gate_a_table defaults to false for the same
    reason -- promotion into the paper table is an audit decision.

Usage:
    python tools/upsert_runs_from_close.py                 # dry run
    python tools/upsert_runs_from_close.py --write          # apply
    python tools/upsert_runs_from_close.py --variant A3a    # one variant only
    python tools/upsert_runs_from_close.py --check-schema   # schema only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path("/media/sangfor/vdb/user/zxk/deepseek-rul")
CSV_PATH = ROOT / "docs/paper_b_hc/registry/runs.csv"
CLOSE_DIR = ROOT / "docs/paper_b_hc/registry/close"
LAUNCH_DIR = ROOT / "docs/paper_b_hc/registry/launch"
OUT_DIR = ROOT / "experiments/hc"

VARIANTS = ["A0", "A1", "A2", "A3a", "A4", "A5", "A3p"]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SUFFIXES = ["", "r2", "r3", "r4", "r5", "r6"]

# Exact column set the registry uses (see tools/check_runs.py BASE_HDR).
BASE_HDR = [
    "run_id", "stage", "variant", "subset", "seed", "span", "gate", "hypothesis",
    "host", "git_sha", "code_md5", "split_hash", "launched_utc", "closed_utc", "status",
    "epochs_run", "best_epoch", "val_rmse", "val_score", "val_mae", "val_bias",
    "test_rmse", "test_score", "test_mae", "test_bias",
    "ckpt_path", "log_path", "preds_path", "manifest_path",
    "registry_note", "audited_by", "audit_verdict",
    "val_metric_kind", "legacy_metric", "baseline_version", "retired_version",
    "determinism_only", "in_gate_a_table",
]

SHORT = {
    "paper_b/models/dspt.py": "dspt",
    "paper_b/training/data.py": "data",
    "paper_b/training/train.py": "train",
    "paper_b/training/metrics.py": "metrics",
    "paper_b/configs/protocol.json": "protocol",
    "paper_b/hc/config_hc.py": "hc_config",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:  # unreadable json: treat as absent, but say so
        print(f"  WARN unreadable {p}: {e}", file=sys.stderr)
        return None


def code_md5_compact(md5: dict) -> str:
    parts = []
    for path, h in sorted((md5 or {}).items()):
        key = SHORT.get(path, Path(path).stem)
        parts.append(f"{key}={h[:8]}")
    return ";".join(parts)


def ok_cell_runs(variant_filter=None):
    """Yield (cell, run_id, close, launch, metrics) for cells whose first ok run exists."""
    for v in VARIANTS:
        if variant_filter and v != variant_filter:
            continue
        for s in SUBSETS:
            picked = None
            for suf in SUFFIXES:
                rid = f"HC-C-{v}-{s}-s101-W1{suf}"
                close = jload(OUT_DIR / rid / "close.json")
                if close and close.get("status") == "ok":
                    picked = (rid, close)
                    break
            if not picked:
                continue
            rid, close = picked
            launch = jload(LAUNCH_DIR / (rid + ".json")) or {}
            metrics = jload(OUT_DIR / rid / "metrics.json") or {}
            yield f"{v}/{s}", rid, close, launch, metrics


def build_row(cell, rid, close, launch, metrics, header):
    v, s = cell.split("/")
    split = (launch.get("data_split") or {}).get("val_engine_ids_hash", "")
    run_dir = OUT_DIR / rid

    def fnum(x):
        if x is None:
            return ""
        return repr(float(x)) if isinstance(x, float) else str(x)

    m = close.get("metrics") or {}
    val_rmse = m.get("val_engine_mean_rmse", metrics.get("val_engine_mean_rmse"))
    val_score = m.get("val_nasa_mean", metrics.get("nasa_mean"))
    val_mae = m.get("val_mae", metrics.get("mae"))
    val_bias = m.get("val_bias", metrics.get("bias"))

    row = {
        "run_id": rid,
        "stage": launch.get("stage", "C"),
        "variant": v,
        "subset": s,
        "seed": launch.get("seed", 101),
        "span": launch.get("span", ""),
        "gate": launch.get("gate", "A"),
        "hypothesis": launch.get("hypothesis", ""),
        "host": launch.get("host", ""),
        "git_sha": launch.get("code_lock_git_sha") or launch.get("git_sha", ""),
        "code_md5": code_md5_compact(launch.get("code_md5") or {}),
        "split_hash": split,
        "launched_utc": launch.get("process_start_utc") or launch.get("launched_utc", ""),
        "closed_utc": close.get("closed_utc", ""),
        "status": close.get("status", ""),
        "epochs_run": close.get("epochs_run", ""),
        "best_epoch": close.get("best_epoch", ""),
        "val_rmse": fnum(val_rmse),
        "val_score": fnum(val_score),
        "val_mae": fnum(val_mae),
        "val_bias": fnum(val_bias),
        "test_rmse": "",
        "test_score": "",
        "test_mae": "",
        "test_bias": "",
        "ckpt_path": str(run_dir / "ckpt/best.pt"),
        "log_path": str(run_dir / "train.log"),
        "preds_path": str(run_dir / "preds.csv"),
        "manifest_path": str(run_dir / "manifest.json"),
        "registry_note": (close.get("notes") or "").replace("\n", " ")[:400],
        "audited_by": "",
        "audit_verdict": "",
        "val_metric_kind": "engine_mean",
        "legacy_metric": "false",
        "baseline_version": launch.get("baseline_version", ""),
        "retired_version": "",
        "determinism_only": "false",
        "in_gate_a_table": "false",
    }
    return [str(row.get(c, "")) for c in header]


def serialize(header, values, terminator):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=terminator)
    w.writerow(values)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    ap.add_argument("--variant", default=None, help="restrict to one variant, e.g. A3a")
    ap.add_argument("--check-schema", action="store_true", help="only validate the CSV header")
    ap.add_argument("--csv", default=str(CSV_PATH))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"FATAL: {csv_path} not found")
        return 2

    raw = csv_path.read_bytes()
    sha_before = sha256_bytes(raw)
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    terminator = "\r\n" if "\r\n" in text else "\n"

    lines = text.splitlines()
    if not lines:
        print("FATAL: empty csv")
        return 2
    header = next(csv.reader([lines[0]]))
    if header != BASE_HDR:
        missing = [c for c in BASE_HDR if c not in header]
        extra = [c for c in header if c not in BASE_HDR]
        print(f"FATAL: header mismatch. missing={missing} extra={extra}")
        return 2
    if args.check_schema:
        print(f"OK schema: {len(header)} columns, {len(lines) - 1} rows, bom={bom}, eol={terminator!r}")
        return 0

    rid_idx = header.index("run_id")
    row_by_rid = {}
    for i, ln in enumerate(lines[1:], start=1):
        if not ln.strip():
            continue
        vals = next(csv.reader([ln]))
        if len(vals) != len(header):
            print(f"WARN line {i+1}: {len(vals)} fields, expected {len(header)}; skipped")
            continue
        row_by_rid.setdefault(vals[rid_idx], i)

    adds, updates, noops, skipped = [], [], [], []
    for cell, rid, close, launch, metrics in ok_cell_runs(args.variant):
        bv = launch.get("baseline_version", "")
        if bv != "v9":
            skipped.append((rid, f"baseline_version={bv!r} != v9"))
            continue
        values = build_row(cell, rid, close, launch, metrics, header)
        new_line = serialize(header, values, terminator).rstrip("\r\n")
        if rid in row_by_rid:
            i = row_by_rid[rid]
            old = lines[i]
            if old == new_line:
                noops.append(rid)
            else:
                updates.append((i, rid, old, new_line))
        else:
            adds.append((rid, new_line))

    print(f"csv          : {csv_path}")
    print(f"sha256 before: {sha_before[:16]}...  rows={len(lines)-1}  bom={bom}  eol={terminator!r}")
    print(f"ADD {len(adds)} / UPDATE {len(updates)} / NOOP {len(noops)} / SKIP {len(skipped)}")
    for rid, _ in adds:
        print(f"  + {rid}")
    for _, rid, _, _ in updates:
        print(f"  ~ {rid}")
    for rid, why in skipped:
        print(f"  ! {rid}  ({why})")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write to apply.")
        return 0
    if not adds and not updates:
        print("\nNothing to do.")
        return 0

    out = list(lines)
    for i, _rid, _old, new_line in updates:
        out[i] = new_line
    for _rid, new_line in adds:
        out.append(new_line)
    body = terminator.join(out) + terminator
    data = body.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data

    # optimistic concurrency: refuse if the admin wrote meanwhile
    if sha256_bytes(csv_path.read_bytes()) != sha_before:
        print("ABORT: runs.csv changed while we were working; nothing written.")
        return 3

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    tmp.write_bytes(data)
    if sha256_bytes(csv_path.read_bytes()) != sha_before:
        tmp.unlink(missing_ok=True)
        print("ABORT: runs.csv changed before replace; nothing written.")
        return 3
    os.replace(tmp, csv_path)
    print(f"\nWROTE {csv_path}  sha256 after: {sha256_bytes(csv_path.read_bytes())[:16]}...")
    print("NOTE: audit_verdict left empty and in_gate_a_table=false -- promotion is an audit decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
