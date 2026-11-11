"""Standard-library verification after transfer or Git checkout; no GPU required."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, require_complete=False):
    root = Path(root)
    manifest = json.loads((root / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    inventory = [item["path"] for item in manifest]
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
    assert len(set(inventory)) == len(inventory) and set(inventory) == actual_files, "manifest coverage/uniqueness"
    for item in manifest:
        path = root / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
    rows = list(csv.DictReader((root / "runs.csv").open(encoding="utf-8", newline="")))
    keys = [(r["protocol"], r["model"], r["subset"], int(r["seed"])) for r in rows]
    expected_keys = {(p, m, f"FD00{fd}", seed) for p in ["author", "unified"]
                     for m in ["aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn"]
                     for fd in range(1, 5) for seed in [101, 202, 303]}
    assert len(set(keys)) == len(keys) and set(keys) <= expected_keys, "duplicate/unregistered run"
    if require_complete:
        assert set(keys) == expected_keys, f"incomplete final matrix: {len(keys)}/120"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    groups = {}
    for key, row in zip(keys, rows):
        groups.setdefault(key[:3], []).append(row)
    summary_keys = [(g["protocol"], g["model"], g["subset"]) for g in summary]
    assert len(set(summary_keys)) == len(summary_keys) and set(summary_keys) == set(groups), "summary groups stale"
    for group, key in zip(summary, summary_keys):
        members = groups[key]
        assert group["n_seeds"] == len(members) and group["seeds"] == sorted(int(r["seed"]) for r in members)
        for name in ["rmse", "nasa_sum", "training_time_s", "normalized_author_rmse", "author_reported_rmse_separate_stochastic_draw", "validation_rmse"]:
            values = [float(r[name]) for r in members if r.get(name)]
            if not values:
                continue
            mean = statistics.mean(values)
            assert math.isclose(group[name + "_mean"], mean, rel_tol=1e-10, abs_tol=1e-8), (key, name)
            if len(values) > 1:
                sd = statistics.stdev(values)
                half = {2: 12.706204736432095, 3: 4.302652729911275}[len(values)] * sd / math.sqrt(len(values))
                assert math.isclose(group[name + "_sd"], sd, rel_tol=1e-9, abs_tol=1e-8), (key, name, "sd")
                assert len(group[name + "_ci95"]) == 2, (key, name, "ci95 length")
                assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8) for a, b in zip(group[name + "_ci95"], [mean-half, mean+half])), (key, name, "ci95")
        for name in ["params", "trainable_params"]:
            assert group[name] == sorted({int(r[name]) for r in members}), (key, name)
        assert group["evaluation_units"] == sorted({r["evaluation_unit"] for r in members})
    for row in rows:
        folder = root / row["path"]
        expected_path = Path("runs") / row["protocol"] / row["model"] / row["subset"] / ("s" + row["seed"])
        assert Path(row["path"]) == expected_path and (folder / "COMPLETED.json").is_file(), folder
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
        for name in ["rmse", "nasa_sum"]:
            assert math.isclose(float(row[name]), metric[name], rel_tol=1e-12, abs_tol=1e-10), (folder, name)
        assert int(row["n"]) == metric["n"] == len(frame) and int(row["params"]) == metric["params"], folder
        for name in ["normalized_author_rmse", "author_reported_rmse_separate_stochastic_draw", "validation_rmse"]:
            value = metric.get(name)
            assert bool(row.get(name)) == (value is not None), (folder, name, "presence")
            if value is not None:
                assert math.isclose(float(row[name]), value, rel_tol=1e-12, abs_tol=1e-10), (folder, name)
        trainable = metric.get("trainable_params", metric["params"])
        if row["protocol"] == "author" and row["model"] == "two_phase" and "trainable_params" not in metric:
            shapes = json.loads((folder / "checkpoint_shapes.json").read_text(encoding="utf-8"))
            trainable -= sum(math.prod(item["shape"]) for name, item in shapes.items() if name.startswith("encoder."))
        assert int(row["trainable_params"]) == trainable, (folder, "trainable_params")
        assert row["evaluation_unit"] == metric["evaluation_unit"], folder
        for name in ["config_sha256", "checkpoint_sha256"]:
            assert row[name] == metric[name], (folder, name)
        time = metric.get("training_time_s", metric.get("elapsed_s"))
        assert time is not None and math.isclose(float(row["training_time_s"]), time, rel_tol=1e-12), folder
        assert sha(folder / "checkpoint.pt") == metric["checkpoint_sha256"]
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        expected = config.pop("config_sha256")
        actual = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert actual == expected == metric["config_sha256"]
        if row["protocol"] == "unified":
            count = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}[row["subset"]]
            assert len(frame) == count and {int(r["unit"]) for r in frame} == set(range(1, count+1))
            freeze = json.loads((folder / "FREEZE.json").read_text())
            assert freeze["frozen"] and freeze["source"] == config["source"]
            assert freeze["config_sha256"] == expected and freeze["checkpoint_sha256"] == metric["checkpoint_sha256"]
            access = json.loads((folder / "data_access_audit.json").read_text())
            assert not access["violations"] and access["official_test_unlocked"] and access["official_test_reads"]
            assert all(item["unix"] >= freeze["created_unix"] for item in access["official_test_reads"])
        log = folder / "training_stdout.log.gz"
        if log.exists():
            assert b"Invalid output detected" not in gzip.decompress(log.read_bytes())
    print(f"PASS: {len(manifest)} file hashes and {len(rows)} independently recomputed run metrics; complete={require_complete}; report rows and summary bound")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--require-complete", action="store_true", help="require all 120 unique preregistered runs")
    a = ap.parse_args()
    verify(a.root, a.require_complete)
