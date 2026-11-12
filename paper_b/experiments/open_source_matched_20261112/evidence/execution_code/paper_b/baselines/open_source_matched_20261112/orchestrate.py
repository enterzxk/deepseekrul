"""Finite registered matrix; TEST is a separate phase behind an all-run freeze."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .common import MODELS, PROTOCOL, config_hash, digest, state_hash, verify_sources, verify_training_data, write_json
from .runner import folder_for


def freeze(root):
    root=Path(root).resolve()
    assert not (root/"BATCH_FREEZE.json").exists(), "batch freeze is immutable"
    checks=json.loads((root/"preflight.json").read_text())
    assert len(checks)==28 and all(c["pass"] for c in checks)
    identity=verify_sources();rows=[]
    for model in MODELS:
        for subset in PROTOCOL["subsets"]:
            for seed in PROTOCOL["seeds"]:
                out=folder_for(root,model,subset,seed)
                assert (out/"TRAINED.json").exists() and not (out/"FAILED_TRAIN.json").exists(),out
                cfg=json.loads((out/"config.json").read_text());expected=cfg.pop("config_sha256")
                assert config_hash(cfg)==expected and cfg["execution_manifest_sha256"]==identity
                verify_training_data(cfg,cfg["data_dir"])
                m=json.loads((out/"training_metrics.json").read_text())
                assert m["config_sha256"]==expected and m["checkpoint_sha256"]==digest(out/"checkpoint.pt")
                assert state_hash(torch.load(out/"checkpoint.pt",map_location="cpu",weights_only=True))==m["state_content_sha256"]
                history=json.loads((out/"history.json").read_text());best=float("inf");best_ep=0
                assert [r["epoch"] for r in history]==list(range(1,len(history)+1))
                for index,row in enumerate(history,1):
                    parts=row["engine_squared_errors"]
                    score=math.fsum(math.sqrt(r["sum_squared_error"]/r["n"]) for r in parts)/len(parts)
                    pooled=math.sqrt(math.fsum(r["sum_squared_error"] for r in parts)/sum(r["n"] for r in parts))
                    assert abs(score-row["val_engine_mean_rmse"])<1e-8 and abs(pooled-row["val_pooled_rmse"])<1e-8
                    if score<best-PROTOCOL["min_delta"]:best,best_ep=score,row["epoch"]
                    assert row["selected_epoch_so_far"]==best_ep
                    if index<len(history):assert index-best_ep<PROTOCOL["patience"], "training continued after early stop"
                assert best_ep==m["best_epoch"] and len(history)<=40 and abs(best-m["validation_engine_mean_rmse"])<1e-7
                assert len(history)==40 or len(history)-best_ep>=PROTOCOL["patience"], "premature stop"
                assert history[best_ep-1]["state_content_sha256"]==m["state_content_sha256"]
                audit=json.loads((out/"train_data_access_audit.json").read_text())
                assert not audit["official_test_unlocked"] and not audit["official_test_reads"] and not audit["violations"] and not audit["nonfinite_replacements"]
                rows.append({"model":model,"subset":subset,"seed":seed,"config_sha256":expected,"checkpoint_sha256":m["checkpoint_sha256"],"data_dir":cfg["data_dir"],"training_data_sha256":cfg["training_data_sha256"]})
    assert len(rows)==84
    with (root/"BATCH_FREEZE.json").open("x") as stream:
        json.dump({"frozen":True,"created_unix":time.time(),"execution_manifest_sha256":identity,"runs":rows},stream,indent=2)
    print("BATCH_FROZEN",len(rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["train","freeze","test"],required=True)
    ap.add_argument("--root",required=True);ap.add_argument("--data-dir",required=True);ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args();root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.phase=="freeze":freeze(root);return
    if a.phase=="test":assert (root/"BATCH_FREEZE.json").exists()
    claim=root/f"ORCHESTRATOR_{a.phase}_RUNNING.json"
    with claim.open("x") as stream:json.dump({"pid":os.getpid(),"started":time.time()},stream)
    logs=root/"logs";logs.mkdir(exist_ok=True)
    order=["dspt_v0_cycle","dspt_train_defaults","aidgn","two_phase","dual_mixer","ffve","fcstgnn"]
    tasks=[(m,fd,s) for s in PROTOCOL["seeds"] for fd in PROTOCOL["subsets"] for m in order]
    def task(item):
        model,subset,seed=item;out=folder_for(root,*item)
        marker="TRAINED.json" if a.phase=="train" else "COMPLETED.json"
        if (out/marker).exists():return {"task":item,"state":"existing"}
        logfile=logs/f"{a.phase}_{model}_{subset}_{seed}.log"
        cmd=[sys.executable,"-u","-m",__package__+".runner","--phase",a.phase,"--model",model,"--subset",subset,"--seed",str(seed),"--root",str(root),"--data-dir",a.data_dir]
        print("START",json.dumps(item),flush=True)
        with logfile.open("x") as stream:
            code=subprocess.call(cmd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","MPLBACKEND":"Agg"})
        row={"task":item,"returncode":code};print("FINISH",json.dumps(row),flush=True);return row
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(task,tasks))
        write_json(root/f"{a.phase}_matrix.json",rows)
        failed=[r for r in rows if r.get("returncode",0)]
        print("MATRIX_FINISHED",a.phase,"tasks",len(rows),"failed",len(failed),flush=True)
        if failed:raise SystemExit(1)
    finally:claim.rename(root/f"ORCHESTRATOR_{a.phase}_FINISHED_{int(time.time())}.json")


if __name__=="__main__":main()
