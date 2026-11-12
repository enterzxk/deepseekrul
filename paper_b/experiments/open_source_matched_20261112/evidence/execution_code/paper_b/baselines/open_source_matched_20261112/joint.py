"""Composite freeze and one-forward evaluation for both registered budget conditions."""
from __future__ import annotations
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from . import common as old
from . import data as old_data
from . import runner as old_runner
from .orchestrate import freeze as freeze_fixed40
from .verify import MODELS,FDS,SEEDS,read,replay,canonical,score
from ..open_source_author_limits_20261112 import common as author
from ..open_source_author_limits_20261112 import data as author_data
from ..open_source_author_limits_20261112 import runner as author_runner


def key(record):return (record["model"],record["subset"],int(record["seed"]))
def runpath(root,k):return Path(root)/"matched"/k[0]/k[1]/f"s{k[2]}"


def write_once(path,value):
    path=Path(path)
    with path.open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2)


def verify_eval_sources():
    manifest=old.ROOT/"EVALUATION_SOURCE_MANIFEST.json"
    record=read(manifest)
    for item in record["files"]:assert old.digest(old.PROJECT/item["path"])==item["sha256"],item["path"]
    return old.digest(manifest)


def check_batch(batch_path,master,condition):
    batch=read(batch_path)
    assert batch["frozen"] and len(batch["runs"])==84
    entries={key(r):r for r in batch["runs"]}
    expected={(m,fd,s) for m in MODELS for fd in FDS for s in SEEDS}
    assert len(entries)==84 and set(entries)==expected
    physical={r["id"]:r for r in master["physical_runs"]}
    logical=[r for r in master["logical_conditions"] if r["condition"]==condition]
    assert len(logical)==84 and {key(r) for r in logical}==expected
    for row in logical:
        entry=entries[key(row)];origin=physical[row["physical_id"]]
        for name in ["config_sha256","checkpoint_sha256","training_data_sha256","data_dir"]:assert entry[name]==origin[name],(condition,key(row),name)
    return old.digest(batch_path)


def validate_ready(original,extended):
    original,extended=Path(original).resolve(),Path(extended).resolve()
    master=read(original/"JOINT_COMPARISON_FREEZE.json")
    assert master["frozen"] and master["state"]=="READY"
    assert master["prepared_sha256"]==old.digest(original/"JOINT_COMPARISON_PREPARED.json")
    assert master["evaluation_source_manifest_sha256"]==verify_eval_sources()
    assert old.verify_sources()==master["source_manifests"]["fixed40"] and author.verify_sources()==master["source_manifests"]["author_limit"]
    assert master["policy_sha256"]==old.digest(old.ROOT/"AUTHOR_LIMIT_POLICY.json")
    assert check_batch(original/"BATCH_FREEZE.json",master,"fixed40")==master["component_batch_sha256"]["fixed40"]
    assert check_batch(extended/"BATCH_FREEZE.json",master,"author_limits")==master["component_batch_sha256"]["author_limits"]
    for record in master["physical_runs"]:
        folder=Path(record["path"]);cfg=read(folder/"config.json");identity=cfg.pop("config_sha256")
        assert canonical(cfg)==identity==record["config_sha256"] and old.digest(folder/"checkpoint.pt")==record["checkpoint_sha256"]
    return master


def validate_trained(folder,identity):
    cfg=read(folder/"config.json");expected=cfg.pop("config_sha256");m=read(folder/"training_metrics.json")
    assert canonical(cfg)==expected==m["config_sha256"] and cfg["execution_manifest_sha256"]==identity
    assert (folder/"TRAINED.json").exists() and not (folder/"FAILED_TRAIN.json").exists()
    old.verify_training_data(cfg,cfg["data_dir"])
    state=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    assert old.state_hash(state)==m["state_content_sha256"] and old.digest(folder/"checkpoint.pt")==m["checkpoint_sha256"]
    replay(read(folder/"history.json"),m,cfg["epochs"])
    access=read(folder/"train_data_access_audit.json")
    assert not access["official_test_reads"] and not access["violations"] and not access["nonfinite_replacements"]
    assert not any((folder/n).exists() for n in ["TEST_ATTEMPT.json","COMPLETED.json","TEST_REUSED.json"]),"freeze after TEST not permitted"
    return cfg,m


def freeze(original,extended):
    original,extended=Path(original).resolve(),Path(extended).resolve()
    master=original/"JOINT_COMPARISON_FREEZE.json"
    if master.exists():
        validate_ready(original,extended);print("JOINT_ALREADY_READY",flush=True);return
    eval_identity=verify_eval_sources()
    policy=read(old.ROOT/"AUTHOR_LIMIT_POLICY.json");eligibility=read(original/"AUTHOR_LIMIT_ELIGIBILITY.json")
    assert eligibility["frozen_before_TEST"] and eligibility["policy_sha256"]==old.digest(old.ROOT/"AUTHOR_LIMIT_POLICY.json")
    cases=eligibility["cases"];expected={(m,fd,s) for m in MODELS for fd in FDS for s in SEEDS}
    assert len(cases)==84 and {key(r) for r in cases}==expected
    identities={"fixed40":old.verify_sources(),"author_limit":author.verify_sources()}
    physical=[];logical=[];prefix=[];selected={};extended_expected=set()
    for rule in cases:
        k=key(rule);folder=runpath(original,k);cfg,m=validate_trained(folder,identities["fixed40"])
        ceiling=policy["ceilings"][k[0]][FDS.index(k[1])]
        qualifies=m["epochs_run"]==40 and 40-m["best_epoch"]<12 and ceiling>40
        assert rule["extension_required"]==qualifies and rule["author_ceiling"]==ceiling
        assert cfg["epochs"]==40 and rule["original_config_sha256"]==m["config_sha256"] and rule["original_checkpoint_sha256"]==m["checkpoint_sha256"]
        pid="fixed40/"+"/".join([k[0],k[1],f"s{k[2]}"])
        rec={"id":pid,"condition":"fixed40","root":str(original),"path":str(folder),"model":k[0],"subset":k[1],"seed":k[2],
             "config_sha256":m["config_sha256"],"checkpoint_sha256":m["checkpoint_sha256"],"state_content_sha256":m["state_content_sha256"],
             "execution_manifest_sha256":cfg["execution_manifest_sha256"],"data_dir":cfg["data_dir"],"training_data_sha256":cfg["training_data_sha256"]}
        physical.append(rec);logical.append({"condition":"fixed40","model":k[0],"subset":k[1],"seed":k[2],"physical_id":pid})
        chosen=rec
        if qualifies:
            extended_expected.add(k);folder2=runpath(extended,k);cfg2,m2=validate_trained(folder2,identities["author_limit"])
            assert cfg2["epochs"]==ceiling
            allowed={"epochs","protocol","protocol_sha256","execution_manifest_sha256","budget_condition"}
            assert {n:v for n,v in cfg.items() if n not in allowed}=={n:v for n,v in cfg2.items() if n not in allowed},"unexpected training hyperparameter/data/seed change"
            a,b=read(folder/"history.json"),read(folder2/"history.json")
            prefix.append({"model":k[0],"subset":k[1],"seed":k[2],"first40_state_hashes_identical":all(x["state_content_sha256"]==y["state_content_sha256"] for x,y in zip(a,b)),
                           "compared_epochs":min(len(a),len(b)),"max_first40_validation_difference":max(abs(x["val_engine_mean_rmse"]-y["val_engine_mean_rmse"]) for x,y in zip(a,b))})
            chosen={**rec,"id":"author_limit/"+"/".join([k[0],k[1],f"s{k[2]}"]),"condition":"author_limit","root":str(extended),"path":str(folder2),
                    "config_sha256":m2["config_sha256"],"checkpoint_sha256":m2["checkpoint_sha256"],"state_content_sha256":m2["state_content_sha256"],"execution_manifest_sha256":cfg2["execution_manifest_sha256"]}
            physical.append(chosen)
        selected[k]=chosen
        logical.append({"condition":"author_limits","model":k[0],"subset":k[1],"seed":k[2],"physical_id":chosen["id"],"author_ceiling":ceiling,"reused_early_stop_or_reference":not qualifies,"reuse_rationale":rule["reuse_rationale"]})
    actual_ext={tuple(p.parent.parts[-3:]) for p in extended.glob("matched/*/FD*/s*/TRAINED.json")}
    assert actual_ext=={(m,fd,f"s{s}") for m,fd,s in extended_expected}
    document={"frozen":False,"state":"PREPARED","policy_sha256":eligibility["policy_sha256"],"source_manifests":identities,"physical_runs":physical,"logical_conditions":logical,
              "n_actual_supervised_runs":len(physical),"n_logical_comparisons":168,"prefix_comparisons":prefix,"official_TEST_previously_observed":True,
              "evaluation_source_manifest_sha256":eval_identity}
    prepared=original/"JOINT_COMPARISON_PREPARED.json"
    if prepared.exists():assert read(prepared)==document,"prepared freeze identity changed"
    else:write_once(prepared,document)
    if not (original/"BATCH_FREEZE.json").exists():freeze_fixed40(original)
    entries=[{**r,"model":k[0],"subset":k[1],"seed":k[2]} for k,r in selected.items()]
    if not (extended/"BATCH_FREEZE.json").exists():
        write_once(extended/"BATCH_FREEZE.json",{"frozen":True,"created_unix":time.time(),"execution_manifest_sha256":identities["author_limit"],"source_manifests":identities,
                   "joint_prepared_sha256":old.digest(prepared),"runs":entries,"logical_reuse_entries_are_not_claimed_as_new_training":True})
    hashes={"fixed40":check_batch(original/"BATCH_FREEZE.json",document,"fixed40"),"author_limits":check_batch(extended/"BATCH_FREEZE.json",document,"author_limits")}
    ready={**document,"frozen":True,"state":"READY","created_unix":time.time(),"prepared_sha256":old.digest(prepared),"component_batch_sha256":hashes}
    write_once(master,ready)
    validate_ready(original,extended)
    print("JOINT_FROZEN",len(physical),"actual trainings;168 logical comparisons;0 TEST",flush=True)


def result_identity(record,master_sha):
    folder=Path(record["path"])
    assert (folder/"COMPLETED.json").exists() or (folder/"TEST_REUSED.json").exists()
    cfg=read(folder/"config.json");expected=cfg.pop("config_sha256")
    assert canonical(cfg)==expected==record["config_sha256"]
    assert old.digest(folder/"checkpoint.pt")==record["checkpoint_sha256"]
    m=read(folder/"metrics.json")
    for name in ["config_sha256","checkpoint_sha256","state_content_sha256"]:assert m[name]==record[name],name
    assert m["batch_freeze_sha256"]==old.digest(Path(record["root"])/"BATCH_FREEZE.json")
    frame=list(csv.DictReader((folder/"test_predictions.csv").open()))
    count={"FD001":100,"FD002":259,"FD003":100,"FD004":248}[record["subset"]]
    assert len(frame)==count and {int(r["unit"]) for r in frame}==set(range(1,count+1))
    r,n=score(frame);assert abs(r-m["rmse"])<1e-5 and np.isclose(n,m["nasa_sum"],rtol=2e-7,atol=.01)
    expected_data=read(old.ROOT/"DATASET_EXPECTED_SHA256.json")
    for kind,h in m["official_test_data_sha256"].items():assert h==expected_data[f'{kind}_{record["subset"]}.txt']
    seal={"joint_freeze_sha256":master_sha,"config_sha256":expected,"checkpoint_sha256":record["checkpoint_sha256"],
          "metrics_sha256":old.digest(folder/"metrics.json"),"predictions_sha256":old.digest(folder/"test_predictions.csv"),
          "test_forward_passes":m["test_forward_passes"],
          "input_provenance_sha256":old.digest(folder/("ACTUAL_TEST_INPUTS.json" if m["test_forward_passes"] else "test_reuse_input_audit.json"))}
    path=folder/"TEST_OUTPUT_SEAL.json"
    if path.exists():assert read(path)==seal,"completed output identity changed"
    else:write_once(path,seal)
    return cfg,m,seal


def evaluate_one(record,data_dir,master_sha):
    folder=Path(record["path"])
    assert old.digest(folder/"checkpoint.pt")==record["checkpoint_sha256"]
    if (folder/"COMPLETED.json").exists() or (folder/"TEST_REUSED.json").exists():
        result_identity(record,master_sha);return
    assert not any((folder/n).exists() for n in ["FAILED_TEST.json","TEST_ATTEMPT.json","TEST_CLAIM.json"]),"attempt consumed; recover saved prediction without new forward if possible"
    module="paper_b.baselines.open_source_matched_20261112.test_instrumented"
    log=Path(record["root"])/"logs"/f'test_{record["model"]}_{record["subset"]}_{record["seed"]}.log'
    cmd=[sys.executable,"-u","-m",module,"--namespace",record["condition"],"--model",record["model"],"--subset",record["subset"],"--seed",str(record["seed"]),"--root",record["root"],"--data-dir",data_dir]
    with log.open("x") as stream:code=subprocess.call(cmd,stdout=stream,stderr=subprocess.STDOUT)
    assert code==0,folder
    result_identity(record,master_sha)


def array_hash(value):
    import hashlib
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps([str(value.dtype),list(value.shape)]).encode());h.update(value.tobytes());return h.hexdigest()


def reuse_prediction(original_record,extended_record,data_dir,master_sha):
    before,after=Path(original_record["path"]),Path(extended_record["path"])
    if (after/"COMPLETED.json").exists() or (after/"TEST_REUSED.json").exists():
        result_identity(extended_record,master_sha);return True
    assert not any((after/name).exists() for name in ["TEST_ATTEMPT.json","TEST_CLAIM.json","FAILED_TEST.json"]),"evaluation slot already consumed"
    a,origin,origin_seal=result_identity(original_record,master_sha)
    b=read(after/"config.json");expected_b=b.pop("config_sha256")
    assert canonical(b)==expected_b==extended_record["config_sha256"]
    assert old.digest(after/"checkpoint.pt")==extended_record["checkpoint_sha256"]
    old.verify_training_data(a,data_dir);author.verify_training_data(b,data_dir)
    if original_record["state_content_sha256"]!=extended_record["state_content_sha256"]:return False
    for name in ["model","subset","seed","window","source","ffve_blocks","ffve_long_residual","precision","tf32"]:
        assert a.get(name)==b.get(name),name
    assert inspect.getsource(old_runner.predict)==inspect.getsource(author_runner.predict)
    guard=old.Guard("test_prediction_reuse_input_audit",unlocked=True)
    with guard:
        expected_data=read(old.ROOT/"DATASET_EXPECTED_SHA256.json")
        assert expected_data==read(author.ROOT/"DATASET_EXPECTED_SHA256.json")
        for name in [f'test_{a["subset"]}.txt',f'RUL_{a["subset"]}.txt']:
            assert old.digest(Path(data_dir)/name)==expected_data[name],"input bytes changed since frozen evaluation"
        x=old_data.prepare(a["subset"],Path(data_dir),a,official_test=True)["test"]
        y=author_data.prepare(b["subset"],Path(data_dir),b,official_test=True)["test"]
    actual_input=read(before/"ACTUAL_TEST_INPUTS.json")
    assert actual_input["checkpoint_sha256"]==original_record["checkpoint_sha256"]
    assert actual_input["evaluation_seed"]==a["seed"]+200000
    import hashlib
    predict_sha=hashlib.sha256(inspect.getsource(old_runner.predict).encode()).hexdigest()
    assert actual_input["predict_function_sha256"]==predict_sha
    arrays={k:{"origin_sha256":array_hash(x[k]),"candidate_sha256":array_hash(y[k]),"actual_origin_forward_sha256":actual_input["arrays"][k]} for k in ["x","age","y","unit","cut"]}
    equal={k:v["origin_sha256"]==v["candidate_sha256"]==v["actual_origin_forward_sha256"] for k,v in arrays.items()}
    old.write_json(after/"test_reuse_input_audit.json",{"arrays":arrays,"equal":equal,"guard":guard.record,"source_output_seal_sha256":old.digest(before/"TEST_OUTPUT_SEAL.json")})
    if not all(equal.values()):return False
    # Same permanent O_EXCL quota used by the inherited real forward path.
    # A competing direct runner either owns this token or fails before forward.
    write_once(after/"TEST_ATTEMPT.json",{"started_unix":time.time(),"limit":1,"mode":"prediction_reuse_without_forward",
               "batch_freeze_sha256":old.digest(Path(extended_record["root"])/"BATCH_FREEZE.json"),"joint_freeze_sha256":master_sha})
    assert not (after/"TEST_CLAIM.json").exists() and not (after/"FAILED_TEST.json").exists()
    result_identity(original_record,master_sha)
    trained=read(after/"training_metrics.json")
    test_keys=["rmse","mae","nasa_sum","nasa_mean","bias","late_frac","severe_late","max_late","n","official_test_data_sha256","evaluation_unit"]
    m={**trained,**{k:origin[k] for k in test_keys},"protocol":"author_limits_20261112","test_forward_passes":0,
       "test_prediction_reused_from":original_record["id"],"source_prediction_sha256":origin_seal["predictions_sha256"],
       "batch_freeze_sha256":old.digest(Path(extended_record["root"])/"BATCH_FREEZE.json"),"joint_freeze_sha256":master_sha,
       "test_runtime_flags":origin["test_runtime_flags"],"reuse_input_arrays_equal":equal,"reuse_evaluation_seed":a["seed"]+200000,
       "source_output_seal_sha256":old.digest(before/"TEST_OUTPUT_SEAL.json")}
    shutil.copy2(before/"test_predictions.csv",after/"test_predictions.csv")
    assert old.digest(after/"test_predictions.csv")==m["source_prediction_sha256"]
    old.write_json(after/"metrics.json",m);old.write_json(after/"test_data_access_audit.json",guard.record)
    write_once(after/"TEST_REUSED.json",{"created_unix":time.time(),"source_id":original_record["id"],"source_prediction_sha256":m["source_prediction_sha256"],"state_content_sha256":m["state_content_sha256"],"test_forward_passes":0,"joint_freeze_sha256":master_sha})
    result_identity(extended_record,master_sha)
    return True


def test(original,extended,data_dir):
    original=Path(original).resolve();master=validate_ready(original,extended);master_sha=old.digest(original/"JOINT_COMPARISON_FREEZE.json")
    write_once(original/"JOINT_TEST_CONTROLLER.json",{"pid":os.getpid(),"started_unix":time.time(),"joint_freeze_sha256":master_sha,"evaluation_source_manifest_sha256":verify_eval_sources()})
    original_cases={key(r):r for r in master["physical_runs"] if r["condition"]=="fixed40"}
    assert len(original_cases)==84
    for row in original_cases.values():
        evaluate_one(row,data_dir,master_sha);print("FIXED40_TEST_COMPLETE",json.dumps(key(row)),flush=True)
    for row in master["physical_runs"]:
        if row["condition"]!="author_limit":continue
        reused=reuse_prediction(original_cases[key(row)],row,data_dir,master_sha)
        if not reused:evaluate_one(row,data_dir,master_sha)
        print("AUTHOR_LIMIT_TEST_COMPLETE",json.dumps({"key":key(row),"reused_actual_prediction":reused}),flush=True)
    old.write_json(original/"JOINT_TEST_COMPLETE.json",{"created_unix":time.time(),"joint_freeze_sha256":master_sha,"physical_runs":len(master["physical_runs"]),
                    "unique_TEST_forward_passes":sum(read(Path(r["path"])/"metrics.json")["test_forward_passes"] for r in master["physical_runs"])})
    print("ALL_JOINT_TEST_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["freeze","test"],required=True);p.add_argument("--root",required=True);p.add_argument("--author-root",required=True);p.add_argument("--data-dir",required=True)
    a=p.parse_args();freeze(a.root,a.author_root) if a.phase=="freeze" else test(a.root,a.author_root,a.data_dir)
