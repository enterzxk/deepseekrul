"""Finite, memory-aware launch control; frozen training mathematics are untouched."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import MODELS, PROTOCOL, digest, verify_sources, write_json
from .runner import folder_for

RESERVE={"dspt_v0_cycle":16000,"dspt_train_defaults":16000,"aidgn":3000,
         "two_phase":3000,"dual_mixer":4000,"ffve":4000,"fcstgnn":6000}

def memory():
    total=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
    listing=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],text=True)
    processes={}
    for line in listing.splitlines():
        fields=[f.strip() for f in line.split(",")]
        if len(fields)==2 and fields[0].isdigit() and fields[1].isdigit():processes[int(fields[0])]=int(fields[1])
    return total,processes

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--data-dir",required=True)
    ap.add_argument("--budget-mib",type=int,default=36000);ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args();root=Path(a.root).resolve();identity=verify_sources()
    claim=root/"MEMORY_SCHEDULER_train_RUNNING.json"
    with claim.open("x") as f:json.dump({"pid":os.getpid(),"started_unix":time.time(),"execution_manifest_sha256":identity,"launcher_sha256":digest(__file__)},f)
    order=["dspt_v0_cycle","dspt_train_defaults","aidgn","two_phase","dual_mixer","ffve","fcstgnn"]
    tasks=[(m,fd,s) for s in PROTOCOL["seeds"] for fd in PROTOCOL["subsets"] for m in order]
    pending=[];rows=[];active=[]
    for item in tasks:
        out=folder_for(root,*item)
        if (out/"TRAINED.json").exists():rows.append({"task":item,"state":"adopted_complete"});continue
        assert not (out/"RUNNING.json").exists() and not (out/"FAILED_TRAIN.json").exists(),out
        pending.append(item)
    last_notice=0
    try:
        while pending or active:
            for job in list(active):
                code=job["process"].poll()
                if code is None:continue
                job["stream"].close();active.remove(job)
                row={"task":job["task"],"returncode":code};rows.append(row)
                print("FINISH",json.dumps(row),flush=True)
                if code:raise RuntimeError(f"retained failed attempt: {job['task']}; no automatic model retry")
                assert (folder_for(root,*job["task"])/"TRAINED.json").exists()
            used,by_pid=memory()
            other=max(0,used-sum(by_pid.get(j["process"].pid,0) for j in active))
            reserved=sum(RESERVE[j["task"][0]] for j in active)
            candidates=[i for i,t in enumerate(pending) if other+reserved+RESERVE[t[0]]<=a.budget_mib]
            if candidates and len(active)<a.workers:
                item=pending.pop(candidates[0]);m,fd,s=item
                log=root/"logs"/f"train_{m}_{fd}_{s}.log"
                stream=log.open("x")
                cmd=[sys.executable,"-u","-m",__package__+".runner","--phase","train","--model",m,"--subset",fd,"--seed",str(s),"--root",str(root),"--data-dir",a.data_dir]
                proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                      env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","MPLBACKEND":"Agg"})
                active.append({"task":item,"process":proc,"stream":stream})
                print("START",json.dumps({"task":item,"pid":proc.pid,"other_used_mib":other,"reserved_mib":reserved+RESERVE[m],"budget_mib":a.budget_mib}),flush=True)
                time.sleep(10)
                continue
            if time.time()-last_notice>=60:
                print("RESOURCE_WAIT",json.dumps({"pending":len(pending),"active":len(active),"gpu_used_mib":used,"other_used_mib":other,"reserved_mib":reserved,"budget_mib":a.budget_mib}),flush=True)
                last_notice=time.time()
            time.sleep(10)
        assert len(rows)==84
        write_json(root/"train_matrix.json",rows)
        print("MATRIX_FINISHED train tasks84 failed0",flush=True)
    finally:
        claim.rename(root/f"MEMORY_SCHEDULER_train_FINISHED_{int(time.time())}.json")

if __name__=="__main__":main()
