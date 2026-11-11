"""Standard-library verification after transfer or Git checkout; no GPU required."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest:
        path = root / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
    rows = list(csv.DictReader((root / "runs.csv").open(encoding="utf-8", newline="")))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert sum(group["n_seeds"] for group in summary) == len(rows), "summary is stale relative to runs.csv"
    for row in rows:
        folder = root / row["path"]
        original = folder / "metrics.json"
        corrected = folder / "RECONCILED_metrics.json"
        metric = json.loads((corrected if corrected.exists() else original).read_text(encoding="utf-8"))
        filename = metric.get("prediction_file", "test_predictions.csv" if row["protocol"] == "unified" else "predictions.csv")
        frame = list(csv.DictReader((folder / filename).open(encoding="utf-8", newline="")))
        differences = [float(r["y_pred"]) - float(r["y_true"]) for r in frame]
        assert all(math.isfinite(d) for d in differences)
        rmse = math.sqrt(math.fsum(d*d for d in differences) / len(differences))
        nasa = math.fsum(math.expm1(-d/13) if d < 0 else math.expm1(d/10) for d in differences)
        assert math.isclose(rmse, metric["rmse"], rel_tol=2e-7, abs_tol=1e-6), folder
        assert math.isclose(nasa, metric["nasa_sum"], rel_tol=2e-7, abs_tol=0.01), folder
        assert sha(folder / "checkpoint.pt") == metric["checkpoint_sha256"]
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        expected = config.pop("config_sha256")
        actual = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert actual == expected == metric["config_sha256"]
        if row["protocol"] == "unified":
            count = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}[row["subset"]]
            assert len(frame) == count and {int(r["unit"]) for r in frame} == set(range(1, count+1))
            freeze = json.loads((folder / "FREEZE.json").read_text())
            assert freeze["config_sha256"] == expected and freeze["checkpoint_sha256"] == metric["checkpoint_sha256"]
            access = json.loads((folder / "data_access_audit.json").read_text())
            assert not access["violations"]
            assert all(item["unix"] >= freeze["created_unix"] for item in access["official_test_reads"])
        log = folder / "training_stdout.log.gz"
        if log.exists():
            assert b"Invalid output detected" not in gzip.decompress(log.read_bytes())
    print(f"PASS: {len(manifest)} file hashes and {len(rows)} independently recomputed run metrics")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    verify(a.root)
