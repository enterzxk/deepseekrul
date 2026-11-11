"""Finite original40 plus validation-registered method-ceiling training, no TEST."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import PROTOCOL, ROOT, config_hash, digest, write_json
from .memory_queue_v2 import Adopted, RESERVE, memory

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--author-root",required=True);p.add_argument("--data-dir",required=True)
    p.add_argument("--budget-mib",type=int,default=36000);p.add_argument("--workers",type=int,default=3);a=p.parse_args()
    original=Path(a.root).resolve();extended=Path(a.author_root).resolve();extended.mkdir(parents=True,exist_ok=True)
    policy=json.loads((ROOT/"AUTHOR_LIMIT_POLICY.json").read_text());policy_sha=digest(ROOT/"AUTHOR_LIMIT_POLICY.json")
    claim=original/"COMBINED_QUEUE_RUNNING.json"
    with claim.open("x") as f:json.dump({"pid":os.getpid(),"started_unix":time.time(),"policy_sha256":policy_sha,"launcher_sha256":digest(__file__)},f)
    order=["dspt_v0_cycle","dspt_train_defaults","aidgn","two_phase","dual_mixer","ffve","fcstgnn"]
    keys=[(m,fd,s) for s in PROTOCOL["seeds"] for fd in PROTOCOL["subsets"] for m in order]
    def folder(root,key):return root/"matched"/key[0]/key[1]/f"s{key[2]}"
    active=[]
    for condition,root in [("fixed40",original),("author_limit",extended)]:
        (root/"logs").mkdir(exist_ok=True)
        for key in keys:
            run=folder(root,key)
            assert not (run/"FAILED_TRAIN.json").exists(),run
            if (run/"RUNNING.json").exists():
                record=json.loads((run/"RUNNING.json").read_text())
                active.append({"condition":condition,"root":root,"task":key,"process":Adopted(record["pid"],run),"stream":None})
                print("ADOPT_RUNNING",json.dumps({"condition":condition,"task":key,"pid":record["pid"]}),flush=True)
    last=0
    try:
        while True:
            for job in list(active):
                code=job["process"].poll()
                if code is None:continue
                if job["stream"] is not None:job["stream"].close()
                active.remove(job);print("FINISH",json.dumps({"condition":job["condition"],"task":job["task"],"returncode":code}),flush=True)
                if code:raise RuntimeError("failed retained attempt; no automatic numerical retry")
                assert (folder(job["root"],job["task"])/"TRAINED.json").exists()
            pending=[];eligibility=[]
            active_keys={(j["condition"],j["task"]) for j in active}
            for key in keys:
                old=folder(original,key)
                if not (old/"TRAINED.json").exists():
                    if ("fixed40",key) not in active_keys:pending.append(("fixed40",original,key))
                    continue
                m=json.loads((old/"training_metrics.json").read_text());cfg=json.loads((old/"config.json").read_text())
                fd_index=PROTOCOL["subsets"].index(key[1]);ceiling=policy["ceilings"][key[0]][fd_index]
                qualified=m["epochs_run"]==40 and 40-m["best_epoch"]<12 and ceiling>40
                record={"model":key[0],"subset":key[1],"seed":key[2],"author_ceiling":ceiling,"extension_required":qualified,
                        "original_epochs_run":m["epochs_run"],"original_best_epoch":m["best_epoch"],"original_config_sha256":cfg["config_sha256"],
                        "original_checkpoint_sha256":m["checkpoint_sha256"],"reuse_rationale":None if qualified else ("reference author ceiling remains40" if ceiling==40 else "same trajectory early-stopped by shared patience; max-independent scheduler")}
                eligibility.append(record)
                if qualified and not (folder(extended,key)/"TRAINED.json").exists() and ("author_limit",key) not in active_keys:
                    pending.append(("author_limit",extended,key))
            write_json(original/"AUTHOR_LIMIT_ELIGIBILITY_PROGRESS.json",{"policy_sha256":policy_sha,"n_original_classified":len(eligibility),"cases":eligibility})
            if not pending and not active and len(eligibility)==84:
                write_json(original/"AUTHOR_LIMIT_ELIGIBILITY.json",{"frozen_before_TEST":True,"created_unix":time.time(),"policy_sha256":policy_sha,"cases":eligibility})
                original_rows=[{"task":key,"state":"trained_verified_marker"} for key in keys]
                write_json(original/"train_matrix.json",original_rows)
                write_json(extended/"extensions_matrix.json",[r for r in eligibility if r["extension_required"]])
                print("ALL_CONDITIONS_TRAINED",json.dumps({"fixed40":84,"fresh_author_extensions":sum(r["extension_required"] for r in eligibility),"official_TEST_opened":False}),flush=True)
                break
            used,by_pid=memory();other=max(0,used-sum(by_pid.get(j["process"].pid,0) for j in active))
            charged=sum(max(RESERVE[j["task"][0]],by_pid.get(j["process"].pid,0)) for j in active)
            candidates=[r for r in pending if other+charged+RESERVE[r[2][0]]<=a.budget_mib]
            if candidates and len(active)<a.workers:
                condition,root,key=candidates[0];model,fd,seed=key
                assert not (folder(root,key)/"RUNNING.json").exists() and not (folder(root,key)/"FAILED_TRAIN.json").exists()
                if condition=="author_limit":
                    request=next(r for r in eligibility if (r["model"],r["subset"],r["seed"])==key)
                    write_json(root/"requests"/model/fd/f"s{seed}.json",{**request,"policy_sha256":policy_sha})
                module="paper_b.baselines.open_source_matched_20261112.runner" if condition=="fixed40" else "paper_b.baselines.open_source_author_limits_20261112.runner"
                stream=(root/"logs"/f"train_{model}_{fd}_{seed}.log").open("x")
                command=[sys.executable,"-u","-m",module,"--phase","train","--model",model,"--subset",fd,"--seed",str(seed),"--root",str(root),"--data-dir",a.data_dir]
                proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","MPLBACKEND":"Agg"})
                active.append({"condition":condition,"root":root,"task":key,"process":proc,"stream":stream})
                print("START",json.dumps({"condition":condition,"task":key,"pid":proc.pid,"other_mib":other,"charged_mib":charged+RESERVE[model],"budget_mib":a.budget_mib}),flush=True)
                time.sleep(10);continue
            if time.time()-last>=60:
                print("RESOURCE_WAIT",json.dumps({"pending":len(pending),"active":len(active),"original_classified":len(eligibility),"eligible_extensions":sum(r["extension_required"] for r in eligibility),"other_mib":other,"charged_mib":charged}),flush=True);last=time.time()
            time.sleep(10)
    finally:claim.rename(original/f"COMBINED_QUEUE_FINISHED_{int(time.time())}.json")

if __name__=="__main__":main()
