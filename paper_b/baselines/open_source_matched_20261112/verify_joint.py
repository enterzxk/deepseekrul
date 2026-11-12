"""Portable, standard-library acceptance for 94 physical / 168 logical cases."""
from __future__ import annotations
import argparse
import csv
import gzip
import math
from pathlib import Path
import statistics
from .verify import (MODELS, FDS, SEEDS, read, sha, canonical, close, score,
                     replay, summarize_rows, compare_rows, assert_nested_close)


def csvrows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify(root):
    root = Path(root)
    inventory = read(root/"ARTIFACT_MANIFEST.json")
    names = [r["path"] for r in inventory]
    assert len(names) == len(set(names))
    assert set(names) == {p.relative_to(root).as_posix() for p in root.rglob("*")
                          if p.is_file() and p != root/"ARTIFACT_MANIFEST.json"}
    for item in inventory:
        p = root/item["path"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], p
    evidence = root/"evidence"
    master = read(evidence/"fixed40/JOINT_COMPARISON_FREEZE.json")
    joint_sha = sha(evidence/"fixed40/JOINT_COMPARISON_FREEZE.json")
    assert master["frozen"] and master["state"] == "READY"
    assert master["prepared_sha256"] == sha(evidence/"fixed40/JOINT_COMPARISON_PREPARED.json")
    assert master["policy_sha256"] == sha(evidence/"source/AUTHOR_LIMIT_POLICY.json")
    assert master["evaluation_source_manifest_sha256"] == sha(evidence/"source/EVALUATION_SOURCE_MANIFEST.json")
    for item in read(evidence/"source/EVALUATION_SOURCE_MANIFEST.json")["files"]:
        assert sha(evidence/"execution_code"/item["path"]) == item["sha256"]
    batches = {}; identities = {}
    for condition, namespace in [("fixed40", "fixed40"), ("author_limits", "author_limit")]:
        batchpath = evidence/condition/"BATCH_FREEZE.json"
        batch = read(batchpath)
        assert batch["frozen"] and len(batch["runs"]) == 84
        entries = {(r["model"], r["subset"], int(r["seed"])):r for r in batch["runs"]}
        expected = {(m, fd, s) for m in MODELS for fd in FDS for s in SEEDS}
        assert len(entries) == 84 and set(entries) == expected
        assert master["component_batch_sha256"][condition] == sha(batchpath)
        folder = evidence/"source"/namespace
        execution = folder/"EXECUTION_SOURCE_MANIFEST.json"
        assert master["source_manifests"][namespace] == sha(execution) == batch["execution_manifest_sha256"]
        identities[namespace] = {"execution_manifest_sha256":sha(execution),
                                 "snapshot_manifest_sha256":sha(folder/"SNAPSHOT_MANIFEST.json"),
                                 "protocol_sha256":sha(folder/"PROTOCOL.json")}
        for item in read(execution)["files"]:
            code = evidence/"execution_code"/item["path"]
            rel=item["path"]
            required=rel.startswith(("paper_b/baselines/open_source_matched_20261112/","paper_b/baselines/open_source_author_limits_20261112/")) or rel in {"paper_b/baselines/open_source/adapters.py","paper_b/baselines/open_source/common.py","paper_b/baselines/open_source/sources.lock.json"} or rel.endswith("/SOURCE_MANIFEST.json")
            if required: assert code.exists() and sha(code) == item["sha256"], rel
        batches[condition] = entries
    eligibility = read(evidence/"fixed40/AUTHOR_LIMIT_ELIGIBILITY.json")
    policy = read(evidence/"source/AUTHOR_LIMIT_POLICY.json")
    assert eligibility["policy_sha256"] == master["policy_sha256"] and eligibility["frozen_before_TEST"]
    assert len(eligibility["cases"]) == 84
    rules = {(r["model"], r["subset"], int(r["seed"])):r for r in eligibility["cases"]}
    physical = csvrows(root/"physical_runs.csv")
    by = {r["physical_id"]:r for r in physical}
    registered = {r["id"]:r for r in master["physical_runs"]}
    assert len(physical) == len(by) == len(registered) == 94 and set(by) == set(registered)
    truths = {(r["subset"], int(r["unit"])):float(r["y_true"]) for r in csvrows(root/"official_truth.csv")}
    dataset = read(evidence/"source/fixed40/DATASET_EXPECTED_SHA256.json")
    splits = {}; cuts = {}; counts = {}; total_forward = 0; reused = 0
    for pid, row in by.items():
        record = registered[pid]; folder = root/row["path"]
        assert row["path"] == "physical_runs/"+pid
        cfg = read(folder/"config.json"); config_sha = cfg.pop("config_sha256")
        m = read(folder/"metrics.json"); key = (m["model"], m["subset"], int(m["seed"]))
        assert key == (row["model"], row["subset"], int(row["seed"]))
        assert canonical(cfg) == config_sha == m["config_sha256"] == row["config_sha256"] == record["config_sha256"]
        namespace = record["condition"]
        for name, expected in identities[namespace].items(): assert cfg[name] == expected
        assert sha(folder/"checkpoint.pt") == m["checkpoint_sha256"] == row["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert m["state_content_sha256"] == row["state_content_sha256"] == record["state_content_sha256"]
        audit = read(folder/"cpu_weight_audit.json")
        assert audit["finite"] and audit["canonical_state_sha256"] == m["state_content_sha256"]
        assert audit["checkpoint_sha256"] == m["checkpoint_sha256"]
        assert cfg["checkpoint_selection"] == "val_engine_mean_rmse" and cfg["patience"] == 12 and cfg["min_delta"] == 1e-4
        assert cfg["precision"] == "float32" and cfg["tf32"] is False and cfg["training_windows"] == "all_cycle_left_padded"
        assert (folder/"TRAINED.json").exists()
        assert not any((folder/n).exists() for n in ["RUNNING.json", "TEST_CLAIM.json", "FAILED_TEST.json", "FAILED_TRAIN.json"])
        freeze = read(folder/"FREEZE.json"); attempt = read(folder/"TEST_ATTEMPT.json")
        batch_condition = "fixed40" if namespace == "fixed40" else "author_limits"
        batch_sha = sha(evidence/batch_condition/"BATCH_FREEZE.json")
        assert attempt["limit"] == 1 and attempt["batch_freeze_sha256"] == batch_sha == m["batch_freeze_sha256"]
        assert freeze["checkpoint_sha256"] == m["checkpoint_sha256"] and freeze["config_sha256"] == config_sha
        assert freeze["execution_manifest_sha256"] == cfg["execution_manifest_sha256"]
        assert freeze["created_unix"] <= master["created_unix"] <= attempt["started_unix"]
        pred = csvrows(folder/"test_predictions.csv")
        n = {"FD001":100, "FD002":259, "FD003":100, "FD004":248}[m["subset"]]
        assert len(pred) == n == m["n"] and {int(p["unit"]) for p in pred} == set(range(1,n+1))
        for p in pred: assert float(p["y_true"]) == truths[(m["subset"],int(p["unit"]))]
        rmse, nasa = score(pred)
        assert close(rmse,m["rmse"]) and math.isclose(nasa,m["nasa_sum"],rel_tol=2e-7,abs_tol=.01)
        for name in ["rmse","nasa_sum","params","trainable_params","training_time_s","validation_engine_mean_rmse","validation_pooled_rmse","best_epoch","epochs_run","n_train_windows","test_forward_passes"]:
            assert close(row[name],m[name]), (pid,name)
        for kind, h in m["official_test_data_sha256"].items(): assert h == dataset[f'{kind}_{m["subset"]}.txt']
        for fd, h in cfg["training_data_sha256"].items(): assert h == dataset[f"train_{fd}.txt"]
        assert m["evaluation_unit"] == "official_test_engine"
        assert not any(m[flags][name] for flags in ["runtime_flags","test_runtime_flags"] for name in ["matmul_tf32","cudnn_tf32"])
        history = read(folder/"history.json"); replay(history,m,cfg["epochs"])
        assert len(history) == m["epochs_run"]
        va = csvrows(folder/"validation_predictions.csv"); groups = {}
        for p in va: groups.setdefault(int(p["unit"]),[]).append((float(p["y_pred"])-float(p["y_true"]))**2)
        assert all(len(v)==5 for v in groups.values())
        assert close(statistics.mean(math.sqrt(math.fsum(v)/len(v)) for v in groups.values()),m["validation_engine_mean_rmse"])
        assert close(score(va)[0],m["validation_pooled_rmse"])
        split = read(folder/"split.json"); ids = (split["train_units"],split["val_units"])
        assert not set(ids[0]) & set(ids[1])
        cut = sorted((int(p["unit"]),int(p["cut"]),float(p["y_true"])) for p in va)
        if m["subset"] in splits:
            assert splits[m["subset"]] == ids and cuts[m["subset"]] == cut and counts[m["subset"]] == m["n_train_windows"]
        splits[m["subset"]]=ids; cuts[m["subset"]]=cut; counts[m["subset"]]=m["n_train_windows"]
        for phase in ["train","test"]:
            access=read(folder/f"{phase}_data_access_audit.json")
            assert not access["violations"] and access["nonfinite_replacements"] == 0
            if phase == "train": assert not access["official_test_reads"] and not access["official_test_unlocked"]
            else: assert access["official_test_reads"] and access["official_test_unlocked"] and all(a["unix"]>=master["created_unix"] for a in access["official_test_reads"])
        assert b"Invalid output detected" not in gzip.decompress((folder/"training_stdout.log.gz").read_bytes())
        seal=read(folder/"TEST_OUTPUT_SEAL.json")
        assert seal["joint_freeze_sha256"] == joint_sha and seal["config_sha256"] == config_sha and seal["checkpoint_sha256"] == m["checkpoint_sha256"]
        assert seal["metrics_sha256"] == sha(folder/"metrics.json") and seal["predictions_sha256"] == sha(folder/"test_predictions.csv")
        total_forward += m["test_forward_passes"]
        if m["test_forward_passes"] == 1:
            assert (folder/"COMPLETED.json").exists() and not (folder/"TEST_REUSED.json").exists()
            actual=read(folder/"ACTUAL_TEST_INPUTS.json")
            assert actual["checkpoint_sha256"] == m["checkpoint_sha256"] and actual["model_was_replaced"] is False
            assert seal["input_provenance_sha256"] == sha(folder/"ACTUAL_TEST_INPUTS.json")
        else:
            reused += 1
            assert m["test_forward_passes"] == 0 and (folder/"TEST_REUSED.json").exists() and not (folder/"COMPLETED.json").exists()
            assert not (folder/"ACTUAL_TEST_INPUTS.json").exists() and attempt["mode"] == "prediction_reuse_without_forward"
            origin=root/by[m["test_prediction_reused_from"]]["path"]
            proof=read(folder/"test_reuse_input_audit.json")
            assert all(proof["equal"].values()) and all(len(set(v.values()))==1 for v in proof["arrays"].values())
            assert sha(origin/"TEST_OUTPUT_SEAL.json") == proof["source_output_seal_sha256"] == m["source_output_seal_sha256"]
            assert sha(origin/"test_predictions.csv") == sha(folder/"test_predictions.csv") == m["source_prediction_sha256"]
            assert read(origin/"metrics.json")["state_content_sha256"] == m["state_content_sha256"]
            assert seal["input_provenance_sha256"] == sha(folder/"test_reuse_input_audit.json")
    assert total_forward == 91 and reused == 3
    complete=read(evidence/"fixed40/JOINT_TEST_COMPLETE.json")
    assert complete["joint_freeze_sha256"] == joint_sha and complete["physical_runs"] == 94 and complete["unique_TEST_forward_passes"] == 91
    logical=csvrows(root/"logical_runs.csv")
    expected={(c,m,fd,s) for c in ["fixed40","author_limits"] for m in MODELS for fd in FDS for s in SEEDS}
    keys=[(r["condition"],r["model"],r["subset"],int(r["seed"])) for r in logical]
    assert len(keys)==len(set(keys))==168 and set(keys)==expected
    frozen={(r["condition"],r["model"],r["subset"],r["seed"]):r for r in master["logical_conditions"]}
    for row,k in zip(logical,keys):
        pid=row["physical_id"]; assert frozen[k]["physical_id"] == pid
        for name,value in by[pid].items(): assert row[name] == value,(k,name)
        entry=batches[row["condition"]][k[1:]]
        for name in ["config_sha256","checkpoint_sha256"]:assert row[name] == entry[name]
        if row["condition"] == "author_limits":
            rule=rules[k[1:]]; original=by['fixed40/'+k[1]+'/'+k[2]+'/s'+str(k[3])]
            qualifies=int(original["epochs_run"])==40 and 40-int(original["best_epoch"])<12 and rule["author_ceiling"]>40
            assert rule["extension_required"]==qualifies and (pid.startswith("author_limit/"))==qualifies
            assert int(row["registered_author_ceiling"])==rule["author_ceiling"]
    assert sum(r["extension_required"] for r in rules.values())==10
    for item in master["prefix_comparisons"]:
        a=root/by[f'fixed40/{item["model"]}/{item["subset"]}/s{item["seed"]}']["path"]
        b=root/by[f'author_limit/{item["model"]}/{item["subset"]}/s{item["seed"]}']["path"]
        x,y=read(a/"history.json"),read(b/"history.json")
        assert len(x)==40 and len(y)>=40
        assert all(i["state_content_sha256"]==j["state_content_sha256"] and i["val_engine_mean_rmse"]==j["val_engine_mean_rmse"] for i,j in zip(x,y))
    summary=[];paired=[]
    for condition in ["author_limits","fixed40"]:
        subset=[r for r in logical if r["condition"]==condition]
        summary += [{"condition":condition,**r} for r in summarize_rows(subset)]
        paired += [{"condition":condition,**r} for r in compare_rows(subset)]
    assert_nested_close(read(root/"summary.json"),summary)
    assert_nested_close(read(root/"paired_comparisons.json"),paired)
    status=read(root/"STATUS.json")
    assert all(status[m]["matched_execution_status"]=="SUCCESS" and status[m]["primary_logical_runs"]==12 for m in MODELS)
    print(f"PASS:94 physical trainings;168 logical cases;91 actual TESTs+3 proven reuses;{len(inventory)} hashes;metrics/selection/source/split/summary/paired verified")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();verify(a.root)
