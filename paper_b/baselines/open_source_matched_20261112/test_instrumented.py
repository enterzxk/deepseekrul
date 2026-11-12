"""Read-only input audit around the exact frozen prediction implementation."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import numpy as np
from .common import digest

def array_digest(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps([str(value.dtype),list(value.shape)]).encode());h.update(value.tobytes());return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--namespace",choices=["fixed40","author_limit"],required=True)
    for name in ["model","subset","data-dir","root"]:p.add_argument("--"+name,required=True)
    p.add_argument("--seed",type=int,required=True);a=p.parse_args()
    name="paper_b.baselines.open_source_matched_20261112.runner" if a.namespace=="fixed40" else "paper_b.baselines.open_source_author_limits_20261112.runner"
    runner=importlib.import_module(name)
    out=runner.folder_for(a.root,a.model,a.subset,a.seed)
    target="predict_reference" if a.model in runner.REFERENCES else "predict"
    original=getattr(runner,target)
    def traced(net,data,*args,**kwargs):
        arrays={k:array_digest(v) for k,v in data.items() if isinstance(v,np.ndarray)}
        record={"phase":"immediately_before_actual_TEST_prediction","namespace":a.namespace,"model":a.model,"subset":a.subset,"seed":a.seed,
                "arrays":arrays,"predict_function_sha256":hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
                "evaluation_seed":a.seed+200000 if target=="predict" else None,"model_was_replaced":False,
                "checkpoint_sha256":digest(out/"checkpoint.pt")}
        # A second whole-dataset predictor call fails before another forward.
        with (out/"ACTUAL_TEST_INPUTS.json").open("x") as stream:json.dump(record,stream,indent=2)
        return original(net,data,*args,**kwargs)
    setattr(runner,target,traced)
    try:runner.test(a.model,a.subset,a.seed,a.data_dir,a.root)
    finally:setattr(runner,target,original)

if __name__=="__main__":main()
