"""Collect real artifacts, reconcile author exports and independently audit metrics."""
from __future__ import annotations
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import numpy as np
import pandas as pd
import torch
from .common import ROOT, config_hash, digest, write_json

MODELS = ["aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn"]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def independently_score(frame):
    error = frame.y_pred.to_numpy(float) - frame.y_true.to_numpy(float)
    assert np.isfinite(error).all()
    rmse = math.sqrt(math.fsum(float(x) ** 2 for x in error) / len(error))
    nasa = math.fsum(math.expm1(-float(x) / 13) if x < 0 else math.expm1(float(x) / 10) for x in error)
    return rmse, nasa


def gather(remote_root, data_dir, target):
    remote_root, target = Path(remote_root).resolve(), Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    # CSV/JSON/checkpoint hashes must survive Windows Git checkouts byte-exact.
    (target / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    rows, audits = [], []
    # Hand-check NASA asymmetry and sum, independently of the training code.
    probe = pd.DataFrame({"y_true": [20.0, 20.0, 20.0], "y_pred": [7.0, 30.0, 20.0]})
    assert abs(independently_score(probe)[1] - 2 * math.expm1(1)) < 1e-12
    for protocol in ["author", "unified"]:
        for model in MODELS:
            version = "v2" if protocol == "author" and model == "aidgn" else ("v1" if protocol == "author" and model in ["two_phase", "dual_mixer"] else ("v4" if protocol == "author" else "v3"))
            base = remote_root / version
            for subset in SUBSETS:
                for seed in [101, 202, 303]:
                    source = base / protocol / model / subset / f"s{seed}"
                    if not (source / "metrics.json").exists():
                        continue
                    if protocol == "author" and model in ["two_phase", "dual_mixer"]:
                        with (source / "reconciliation.log").open("w") as stream:
                            subprocess.run([sys.executable, "-m", "paper_b.baselines.open_source.reconcile_author",
                                            "--run-dir", str(source), "--data-dir", str(data_dir)],
                                           stdout=stream, stderr=subprocess.STDOUT, check=True)
                    preferred = source / "RECONCILED_metrics.json"
                    m = json.loads((preferred if preferred.exists() else source / "metrics.json").read_text())
                    pred_name = m.get("prediction_file", "test_predictions.csv" if protocol == "unified" else "predictions.csv")
                    frame = pd.read_csv(source / pred_name)
                    r, n = independently_score(frame)
                    assert math.isclose(r, m["rmse"], rel_tol=2e-7, abs_tol=1e-6), source
                    assert math.isclose(n, m["nasa_sum"], rel_tol=2e-7, abs_tol=0.01), source
                    assert digest(source / "checkpoint.pt") == m["checkpoint_sha256"], source
                    cfg = json.loads((source / "config.json").read_text())
                    expected = cfg.pop("config_sha256")
                    assert config_hash(cfg) == expected == m["config_sha256"], source
                    state = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
                    tensors = list(state.values())
                    if tensors and isinstance(tensors[0], dict):
                        tensors = [tensor for group in tensors for tensor in group.values()]
                    assert all(not isinstance(t, torch.Tensor) or torch.isfinite(t).all() for t in tensors), source
                    audit = {"protocol": protocol, "model": model, "subset": subset, "seed": seed,
                             "recomputed_rmse": r, "recomputed_nasa_sum": n, "config_hash_valid": True,
                             "checkpoint_hash_valid": True, "weights_finite": True, "n": len(frame)}
                    if protocol == "author" and m["evaluation_unit"].startswith("official_test_engine"):
                        expected_n = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}[subset]
                        assert len(frame) == expected_n and set(frame.unit) == set(range(1, expected_n + 1))
                        truth = np.minimum(np.loadtxt(Path(data_dir) / f"RUL_{subset}.txt"), 125)
                        assert np.allclose(frame.sort_values("unit").y_true, truth)
                        audit.update(official_engine_ids_valid=True, official_labels_match=True)
                    if protocol == "unified":
                        assert len(frame) == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}[subset]
                        assert frame.unit.nunique() == len(frame) and set(frame.unit) == set(range(1, len(frame) + 1))
                        freeze = json.loads((source / "FREEZE.json").read_text())
                        assert freeze["checkpoint_sha256"] == m["checkpoint_sha256"] and freeze["frozen"]
                        assert freeze["config_sha256"] == m["config_sha256"] and freeze["source"] == cfg["source"]
                        split = json.loads((source / "split.json").read_text())
                        assert not set(split["train_units"]) & set(split["val_units"])
                        access = json.loads((source / "data_access_audit.json").read_text())
                        assert not access["violations"] and access["official_test_unlocked"]
                        assert access["official_test_reads"] and all(entry["unix"] >= freeze["created_unix"] for entry in access["official_test_reads"])
                        truth = np.minimum(np.loadtxt(Path(data_dir) / f"RUL_{subset}.txt"), 125)
                        assert np.allclose(frame.sort_values("unit").y_true, truth)
                        audit.update(engine_ids_unique=True, official_engine_count_valid=True,
                                     no_test_before_freeze=True, official_labels_match=True)
                    # Keep only reproducibility evidence; runtime upstream copies
                    # are reconstructed by acquire + the recorded runtime patch.
                    dest = target / "runs" / protocol / model / subset / f"s{seed}"
                    dest.mkdir(parents=True, exist_ok=True)
                    shapes = {}
                    for name, value in state.items():
                        if isinstance(value, dict):
                            for nested, tensor in value.items():
                                if isinstance(tensor, torch.Tensor):
                                    shapes[name + "." + nested] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                        elif isinstance(value, torch.Tensor):
                            shapes[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                    write_json(dest / "checkpoint_shapes.json", shapes)
                    if protocol == "unified" and model == "two_phase":
                        write_json(dest / "SEED_POLICY_ANNOTATION.json", {
                            "original_config_sha256": m["config_sha256"],
                            "policy": "whole-pipeline seed; FT initialized before PRE; no phase reseed",
                            "author_policy": "independent reseed for each phase",
                            "architecture_objectives_splits_hyperparameters_changed": False,
                            "disclosure": "seed-matched author phase initializations/trajectories are not claimed"})
                    for path in source.iterdir():
                        if path.is_file() and path.suffix in [".json", ".csv", ".pt", ".patch", ".txt", ".log"]:
                            shutil.copy2(path, dest / path.name)
                    log = base / "logs" / f"{protocol}_{model}_{subset}_{seed}.log"
                    if log.exists():
                        (dest / "training_stdout.log.gz").write_bytes(gzip.compress(log.read_bytes(), mtime=0))
                        audit["upstream_nonfinite_replacement_messages"] = log.read_text(errors="replace").count("Invalid output detected")
                        if audit["upstream_nonfinite_replacement_messages"]:
                            write_json(dest / "INVALID_NONFINITE_OUTPUT.json", audit)
                            raise AssertionError(f"upstream imputed nonfinite predictions; run rejected: {source}")
                    trainable = m.get("trainable_params", m["params"])
                    if protocol == "author" and model == "two_phase" and "trainable_params" not in m:
                        trainable = m["params"] - sum(value.numel() for name, value in state.items() if name.startswith("encoder."))
                    row = {"protocol": protocol, "model": model, "subset": subset, "seed": seed,
                           "rmse": m["rmse"], "nasa_sum": m["nasa_sum"], "n": m["n"],
                           "params": m["params"], "trainable_params": trainable, "training_time_s": m.get("training_time_s", m.get("elapsed_s")),
                           "config_sha256": m["config_sha256"], "checkpoint_sha256": m["checkpoint_sha256"],
                           "evaluation_unit": m["evaluation_unit"], "path": str(dest.relative_to(target)),
                           "normalized_author_rmse": m.get("normalized_author_rmse"),
                           "author_reported_rmse_separate_stochastic_draw": m.get("author_reported_rmse_separate_stochastic_draw"),
                           "validation_rmse": m.get("validation_rmse"), "best_epoch": m.get("best_epoch")}
                    rows.append(row)
                    audits.append(audit)
    table = pd.DataFrame(rows)
    table.to_csv(target / "runs.csv", index=False)
    write_json(target / "metric_audit.json", audits)
    evidence = target / "evidence"
    evidence.mkdir(exist_ok=True)
    deployment = remote_root.parent
    seed_audit = deployment / "docs/paper_b_baselines/audits"
    if seed_audit.exists():
        shutil.copytree(seed_audit, evidence / "seed_policy_audit", dirs_exist_ok=True)
    for name in ["runtime_checks.json", "runtime_checks_ffve_final.json", "candidate_execution.json", "environment_packages.json", "optional_dependencies.json", "runtime_checks.log", "runtime_checks_v2.log", "runtime_checks_ffve_final.log"]:
        if (deployment / name).exists():
            shutil.copy2(deployment / name, evidence / name)
    history = evidence / "execution_code"
    history.mkdir(exist_ok=True)
    # The initial deployment also contained upstream code; extract ONLY our
    # own harness versions, never re-upload the unlicensed upstream archive.
    if (deployment / "source.tar.gz").exists():
        with tarfile.open(deployment / "source.tar.gz", "r:gz") as archive:
            for filename in ["author.py", "common.py", "dispatch.py"]:
                member = "paper_b/baselines/open_source/" + filename
                stream = archive.extractfile(member)
                if stream is not None:
                    (history / ("initial_" + filename)).write_bytes(stream.read())
    code_manifest = []
    for filename in ["author.py", "unified.py", "adapters.py", "data.py", "common.py", "reconcile_author.py", "gather.py", "report.py", "matrix.py"]:
        source = ROOT / filename
        shutil.copy2(source, history / ("final_" + filename))
        code_manifest.append({"path": "paper_b/baselines/open_source/" + filename, "sha256": digest(source)})
    for filename in ["data.py", "metrics.py"]:
        source = deployment / "paper_b/training" / filename
        code_manifest.append({"path": "paper_b/training/" + filename, "sha256": digest(source)})
    write_json(evidence / "execution_code_sha256.json", code_manifest)
    recovered = ROOT / "execution_history"
    if recovered.exists():
        for path in recovered.iterdir():
            if path.is_file():
                shutil.copy2(path, history / path.name)
    versions = {}
    for path in history.glob("*author*.py"):
        versions.setdefault(digest(path), []).append(path.relative_to(target).as_posix())
    provenance = []
    for row in rows:
        if row["protocol"] != "author":
            continue
        cfg = json.loads((target / row["path"] / "config.json").read_text())
        runner = cfg.get("runner_sha256")
        status = "EXACT_SOURCE_RETAINED" if runner in versions else ("HASH_NOT_RECORDED" if runner is None else "EXACT_SOURCE_MISSING")
        provenance.append({"model": row["model"], "subset": row["subset"], "seed": row["seed"],
                           "path": row["path"], "runner_sha256": runner, "status": status,
                           "matching_sources": versions.get(runner, []),
                           "upstream_snapshot_and_runtime_patch_retained": True})
    write_json(evidence / "EXECUTION_PROVENANCE.json", {
        "scope": "author launcher exact source; unified core code hashes audited separately",
        "known_sources": versions, "author_runs": provenance,
        "exact_source_retained": sum(x["status"] == "EXACT_SOURCE_RETAINED" for x in provenance),
        "exact_source_missing": sum(x["status"] == "EXACT_SOURCE_MISSING" for x in provenance),
        "runner_hash_not_recorded": sum(x["status"] == "HASH_NOT_RECORDED" for x in provenance),
        "disclosure": "Missing historical launcher text is not replaced by a different source or claimed byte-exact. Original configs are unchanged; upstream/core/runtime patches and trained predictions remain available."})
    # Explicitly retain the failed original import attempt and its traceback.
    failure = remote_root / "v1/author/aidgn/FD001/s101"
    if failure.exists():
        out = evidence / "aidgn_failed_optional_import"
        out.mkdir(exist_ok=True)
        for path in failure.iterdir():
            if path.is_file():
                shutil.copy2(path, out / path.name)
    for name in ["matrix_complete.json", "matrix_stage_1.json", "matrix_stage_2.json", "matrix_stage_3.json", "matrix_stage_4.json"]:
        if (remote_root / name).exists():
            shutil.copy2(remote_root / name, evidence / name)
    previous = remote_root / "orchestrator_attempt1"
    if previous.exists():
        shutil.copytree(previous, evidence / "orchestrator_attempt1", dirs_exist_ok=True)
    for name in ["matrix.log", "matrix_retry.log"]:
        if (deployment / name).exists():
            (evidence / (name + ".gz")).write_bytes(gzip.compress((deployment / name).read_bytes(), mtime=0))
    failures = []
    for record in remote_root.glob("v*/*/*/FD*/s*/FAILED.json"):
        source = record.parent
        dest = evidence / "failed_attempts" / source.relative_to(remote_root)
        dest.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, dest / path.name)
        version, protocol, model, subset, seed = source.relative_to(remote_root).parts
        log = remote_root / version / "logs" / f"{protocol}_{model}_{subset}_{seed[1:]}.log"
        if log.exists():
            (dest / "training_stdout.log.gz").write_bytes(gzip.compress(log.read_bytes(), mtime=0))
        failures.append({"attempt": str(source.relative_to(remote_root)), "evidence": str(dest.relative_to(target)),
                         "error": (source / "error.txt").read_text() if (source / "error.txt").exists() else "see stdout"})
    write_json(evidence / "FAILURES.json", failures)
    checks = json.loads((evidence / "runtime_checks.json").read_text())
    if (evidence / "runtime_checks_ffve_final.json").exists():
        checks = [x for x in checks if x["model"] != "ffve"] + json.loads((evidence / "runtime_checks_ffve_final.json").read_text())
    write_json(evidence / "runtime_checks_final.json", checks)
    assert len(checks) == 20
    fingerprints = {p.name: digest(p) for p in Path(data_dir).glob("*.txt") if p.name.startswith(("train_FD", "test_FD", "RUL_FD"))}
    write_json(evidence / "dataset_sha256.json", fingerprints)
    inventory = []
    for file in sorted(target.rglob("*")):
        if file.is_file() and file.name != "ARTIFACT_MANIFEST.json":
            inventory.append({"path": str(file.relative_to(target)).replace("\\", "/"), "bytes": file.stat().st_size, "sha256": digest(file)})
    write_json(target / "ARTIFACT_MANIFEST.json", inventory)
    print("COLLECTED", len(rows), "audited runs;", len(inventory), "files", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--target", required=True)
    a = ap.parse_args()
    gather(a.remote_root, a.data_dir, a.target)
