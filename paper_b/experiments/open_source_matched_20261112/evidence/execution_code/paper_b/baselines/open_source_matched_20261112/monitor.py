"""Compact read-only progress from exclusive run claims and retained logs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

def snapshot(root):
    root=Path(root);counts=Counter();active=[];failed=[]
    for run in (root/"matched").glob("*/FD*/s*"):
        model,fd,seed=run.parts[-3:]
        if (run/"COMPLETED.json").exists():counts[(model,"tested")]+=1
        if (run/"TRAINED.json").exists():counts[(model,"trained")]+=1
        for name in ["RUNNING.json","TEST_CLAIM.json"]:
            if (run/name).exists():
                claim=json.loads((run/name).read_text());phase="train" if name=="RUNNING.json" else "test"
                log=root/"logs"/f"{phase}_{model}_{fd}_{seed[1:]}.log"
                lines=log.read_text(errors="replace").splitlines() if log.exists() else []
                active.append({"model":model,"subset":fd,"seed":seed,"phase":phase,"elapsed_s":round(time.time()-claim.get("started_unix",time.time())),"last":lines[-1][:220] if lines else "starting"})
        for name in ["FAILED_TRAIN.json","FAILED_TEST.json"]:
            if (run/name).exists():failed.append(str(run))
    return {"counts":{m:{s:counts[(m,s)] for s in ["trained","tested"]} for m in sorted({k[0] for k in counts})},"active":active,"failed":failed}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args()
    print(json.dumps(snapshot(a.root),ensure_ascii=False,indent=2))
