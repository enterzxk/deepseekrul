"""Independent standard-library acceptance of the complete matched comparison."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

MODELS=["aidgn","two_phase","dual_mixer","ffve","fcstgnn","dspt_v0_cycle","dspt_train_defaults"]
FDS=["FD001","FD002","FD003","FD004"]
SEEDS=[101,202,303]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def canonical(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-7,abs_tol=1e-6)

def score(rows):
    errors=[float(r["y_pred"])-float(r["y_true"]) for r in rows]
    assert errors and all(math.isfinite(x) for x in errors)
    return math.sqrt(math.fsum(x*x for x in errors)/len(errors)),math.fsum(math.expm1(-x/13) if x<0 else math.expm1(x/10) for x in errors)

def replay(history,metrics):
    assert [r["epoch"] for r in history]==list(range(1,len(history)+1)) and 1<=len(history)<=40
    best,best_ep=float("inf"),0
    for index,row in enumerate(history,1):
        groups=row["engine_squared_errors"]
        assert groups and len({g["unit"] for g in groups})==len(groups) and all(g["n"]==5 for g in groups)
        engine=math.fsum(math.sqrt(g["sum_squared_error"]/g["n"]) for g in groups)/len(groups)
        pooled=math.sqrt(math.fsum(g["sum_squared_error"] for g in groups)/sum(g["n"] for g in groups))
        assert close(engine,row["val_engine_mean_rmse"]) and close(pooled,row["val_pooled_rmse"])
        if engine<best-1e-4:best,best_ep=engine,index
        assert row["selected_epoch_so_far"]==best_ep
        if index<len(history):assert index-best_ep<12
    assert len(history)==40 or len(history)-best_ep>=12
    assert best_ep==metrics["best_epoch"] and close(best,metrics["validation_engine_mean_rmse"])
    assert history[best_ep-1]["state_content_sha256"]==metrics["state_content_sha256"]
    return best_ep

def summarize_rows(rows):
    groups={}
    for row in rows:groups.setdefault((row["model"],row["subset"]),[]).append(row)
    result=[]
    for (model,subset),members in sorted(groups.items()):
        assert sorted(int(r["seed"]) for r in members)==SEEDS
        record={"model":model,"subset":subset,"seeds":SEEDS,"n_seeds":3}
        for key in ["rmse","nasa_sum","training_time_s","validation_engine_mean_rmse","validation_pooled_rmse"]:
            values=[float(r[key]) for r in members];mean=statistics.mean(values);sd=statistics.stdev(values);half=4.302652729911275*sd/math.sqrt(3)
            record.update({key+"_mean":mean,key+"_sd":sd,key+"_ci95":[mean-half,mean+half]})
        record["params"]=sorted({int(r["params"]) for r in members})
        record["trainable_params"]=sorted({int(r["trainable_params"]) for r in members})
        result.append(record)
    return result

def compare_rows(rows):
    by={(r["model"],r["subset"],int(r["seed"])):r for r in rows};result=[]
    for reference in ["dspt_v0_cycle","dspt_train_defaults"]:
        for model in MODELS[:5]:
            for subset in FDS:
                record={"reference":reference,"model":model,"subset":subset,"seeds":SEEDS}
                for key in ["rmse","nasa_sum"]:
                    differences=[float(by[(model,subset,s)][key])-float(by[(reference,subset,s)][key]) for s in SEEDS]
                    mean=statistics.mean(differences);sd=statistics.stdev(differences);half=4.302652729911275*sd/math.sqrt(3)
                    record.update({key+"_paired_differences":differences,key+"_difference_mean":mean,key+"_difference_sd":sd,key+"_difference_ci95":[mean-half,mean+half],key+"_baseline_better_seeds":sum(x<0 for x in differences)})
                result.append(record)
    return result

def assert_nested_close(actual,expected):
    assert type(actual)==type(expected),(type(actual),type(expected))
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:assert_nested_close(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):assert_nested_close(a,b)
    elif isinstance(expected,float):assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-8),(actual,expected)
    else:assert actual==expected,(actual,expected)

def verify(root):
    root=Path(root);inventory=read(root/"ARTIFACT_MANIFEST.json")
    names=[r["path"] for r in inventory]
    assert len(set(names))==len(names)
    assert set(names)=={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"}
    for r in inventory:
        path=root/r["path"];assert path.stat().st_size==r["bytes"] and sha(path)==r["sha256"],path
    rows=list(csv.DictReader((root/"runs.csv").open(encoding="utf-8",newline="")))
    keys=[(r["model"],r["subset"],int(r["seed"])) for r in rows]
    expected={(m,fd,s) for m in MODELS for fd in FDS for s in SEEDS}
    assert len(keys)==len(set(keys))==84 and set(keys)==expected
    batch=read(root/"evidence/BATCH_FREEZE.json");assert batch["frozen"] and len(batch["runs"])==84
    entries={(r["model"],r["subset"],int(r["seed"])):r for r in batch["runs"]}
    assert len(entries)==84 and set(entries)==expected
    execution_sha=sha(root/"evidence/EXECUTION_SOURCE_MANIFEST.json")
    snapshot_sha=sha(root/"evidence/SNAPSHOT_MANIFEST.json");protocol_sha=sha(root/"evidence/PROTOCOL.json")
    assert batch["execution_manifest_sha256"]==execution_sha
    for item in read(root/"evidence/EXECUTION_SOURCE_MANIFEST.json")["files"]:
        rel=item["path"]
        if rel.startswith("paper_b/baselines/open_source_matched_20261112/") or rel in {"paper_b/baselines/open_source/adapters.py","paper_b/baselines/open_source/common.py","paper_b/baselines/open_source/sources.lock.json"} or rel.endswith("/SOURCE_MANIFEST.json"):
            assert sha(root/"evidence/execution_code"/rel)==item["sha256"],rel
    split_ids={};sample_counts={};val_keys={}
    dataset=read(root/"evidence/DATASET_EXPECTED_SHA256.json")
    for row in rows:
        folder=root/row["path"];assert Path(row["path"])==Path("runs")/row["model"]/row["subset"]/("s"+row["seed"])
        assert (folder/"TRAINED.json").exists() and (folder/"COMPLETED.json").exists() and (folder/"TEST_ATTEMPT.json").exists()
        assert not any((folder/p).exists() for p in ["RUNNING.json","TEST_CLAIM.json","FAILED_TRAIN.json","FAILED_TEST.json"])
        m=read(folder/"metrics.json");cfg=read(folder/"config.json");expected_config=cfg.pop("config_sha256")
        key=(row["model"],row["subset"],int(row["seed"]))
        assert (m["model"],m["subset"],m["seed"])==key==(cfg["model"],cfg["subset"],cfg["seed"])
        entry=entries[key]
        assert canonical(cfg)==expected_config==m["config_sha256"]==row["config_sha256"]
        assert entry["config_sha256"]==expected_config and entry["checkpoint_sha256"]==m["checkpoint_sha256"]
        assert entry["training_data_sha256"]==cfg["training_data_sha256"] and entry["data_dir"]==cfg["data_dir"]
        assert cfg["execution_manifest_sha256"]==execution_sha and cfg["snapshot_manifest_sha256"]==snapshot_sha and cfg["protocol_sha256"]==protocol_sha
        assert cfg["checkpoint_selection"]=="val_engine_mean_rmse" and cfg["epochs"]==40 and cfg["patience"]==12 and cfg["min_delta"]==1e-4
        assert cfg["precision"]=="float32" and cfg["training_windows"]=="all_cycle_left_padded" and cfg["tf32"] is False
        assert sha(folder/"checkpoint.pt")==m["checkpoint_sha256"]==row["checkpoint_sha256"]
        predictions=list(csv.DictReader((folder/"test_predictions.csv").open(encoding="utf-8",newline="")))
        count={"FD001":100,"FD002":259,"FD003":100,"FD004":248}[row["subset"]]
        assert len(predictions)==count==int(row["n"])==m["n"] and {int(p["unit"]) for p in predictions}==set(range(1,count+1))
        r,n=score(predictions)
        assert close(r,m["rmse"]) and math.isclose(n,m["nasa_sum"],rel_tol=2e-7,abs_tol=.01)
        for key in ["rmse","nasa_sum","training_time_s","validation_engine_mean_rmse","validation_pooled_rmse","params","trainable_params","n_train_windows","best_epoch","epochs_run"]:
            assert close(row[key],m[key]),(folder,key)
        assert m["test_forward_passes"]==1 and m["evaluation_unit"]=="official_test_engine"
        assert not any(m["runtime_flags"][k] or m["test_runtime_flags"][k] for k in ["matmul_tf32","cudnn_tf32"])
        for fd,h in cfg["training_data_sha256"].items():assert h==dataset[f"train_{fd}.txt"]
        for kind,h in m["official_test_data_sha256"].items():assert h==dataset[f"{kind}_{row['subset']}.txt"]
        freeze=read(folder/"FREEZE.json");attempt=read(folder/"TEST_ATTEMPT.json")
        assert freeze["frozen"] and freeze["checkpoint_sha256"]==m["checkpoint_sha256"] and freeze["config_sha256"]==expected_config
        assert freeze["execution_manifest_sha256"]==cfg["execution_manifest_sha256"]==batch["execution_manifest_sha256"]
        assert freeze["created_unix"]<=batch["created_unix"]<=attempt["started_unix"]
        assert m["batch_freeze_sha256"]==sha(root/"evidence/BATCH_FREEZE.json")==attempt["batch_freeze_sha256"]
        history=read(folder/"history.json");replay(history,m);assert m["epochs_run"]==len(history)
        va=list(csv.DictReader((folder/"validation_predictions.csv").open(encoding="utf-8",newline="")))
        groups={}
        for v in va:groups.setdefault(int(v["unit"]),[]).append((float(v["y_pred"])-float(v["y_true"]))**2)
        em=statistics.mean(math.sqrt(math.fsum(v)/len(v)) for v in groups.values())
        assert close(em,m["validation_engine_mean_rmse"]) and close(score(va)[0],m["validation_pooled_rmse"])
        for phase in ["train","test"]:
            access=read(folder/f"{phase}_data_access_audit.json")
            assert not access["violations"] and not access["nonfinite_replacements"]
            if phase=="train":assert not access["official_test_reads"] and not access["official_test_unlocked"]
            else:assert access["official_test_unlocked"] and access["official_test_reads"] and all(r["unix"]>=batch["created_unix"] for r in access["official_test_reads"])
        split=read(folder/"split.json");assert not set(split["train_units"])&set(split["val_units"])
        key=row["subset"];ids=(tuple(split["train_units"]),tuple(split["val_units"]))
        if key in split_ids:assert split_ids[key]==ids and sample_counts[key]==m["n_train_windows"]
        split_ids[key]=ids;sample_counts[key]=m["n_train_windows"]
        cuts=sorted((int(r["unit"]),int(r["cut"]),float(r["y_true"])) for r in va)
        if key in val_keys:assert val_keys[key]==cuts
        val_keys[key]=cuts
        assert not b"Invalid output detected" in gzip.decompress((folder/"training_stdout.log.gz").read_bytes())
    assert_nested_close(read(root/"summary.json"),summarize_rows(rows))
    assert_nested_close(read(root/"paired_comparisons.json"),compare_rows(rows))
    print(f"PASS:84 exact registered runs; {len(inventory)} hashes; predictions/engine-SSE/state-selection/rows/summary/paired comparisons bound; same splits/cuts/sample counts")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();verify(a.root)
