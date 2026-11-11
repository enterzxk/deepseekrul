"""New DSPT-matched batch: training is isolated from globally gated official TEST."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..open_source.adapters import UpstreamAdapter, module
from .common import ROOT, REFERENCES, Guard, backend_flags, configuration, config_hash, digest, metrics, predictions, research_task_start, seed_everything, state_hash, validation_record, verify_sources, verify_training_data, write_json
from .data import prepare
from .reference_wrap import train_reference, predict_reference, load_reference
from .reference.training import data as reference_data

def predict(net, data, seed, batch=256):
    net.eval()
    outputs = []
    # FFVE uses the actual variational sample, never a latent-mean shortcut.
    # Fixed evaluation seed gives repeatable single-draw validation/test.
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for start in range(0, len(data["y"]), batch):
            x = torch.from_numpy(data["x"][start:start + batch]).cuda()
            age = torch.from_numpy(data["age"][start:start + batch]).cuda()
            outputs.append(net(x, age).cpu().numpy())
    return np.concatenate(outputs)


def pretrain_two_phase(net, cfg, data, data_dir, out):
    paired = prepare(cfg["paired_pretraining_subset"], data_dir, cfg)
    write_json(out / "paired_split.json", {k: paired[k] for k in ["train_units", "val_units", "scaler"]})
    x = np.concatenate([data["train"]["x"], paired["train"]["x"]])
    y = np.concatenate([data["train"]["y"], paired["train"]["y"]])
    domain = np.concatenate([np.ones(len(data["train"]["y"]), np.int64), np.zeros(len(paired["train"]["y"]), np.int64)])
    # Upstream corruption is fixed once before training, with random seed 529.
    import random
    rng = random.Random(529)
    mask_p, random_p = (0.6, 0.3) if cfg["pretraining_epochs"] == 20 else (0.3, 0.6)
    for window in x:
        value = rng.random()
        if value < mask_p:
            window[-1] = 0
        elif value < mask_p + random_p:
            window[-1] = [rng.random() for _ in range(x.shape[-1])]
            window[-1, 1] = 0
    layers, dropout = (1, 0.1) if cfg["pretraining_epochs"] == 20 else (2, 0.0)
    pre = net.pre_module.Transformer_pre(17, 16, 45, 32, 32, layers, 2, dropout).cuda()
    opt = torch.optim.Adam(pre.parameters(), lr=0.001)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(domain))
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    history = []
    for epoch in range(1, cfg["pretraining_epochs"] + 1):
        pre.train()
        loss_sum = 0.0
        for bx, by, bd in loader:
            bx, by, bd = bx.cuda(), by.cuda(), bd.cuda()
            opt.zero_grad(set_to_none=True)
            regression, classification = pre(bx)
            loss = torch.sqrt(torch.mean((regression.reshape(-1) - by) ** 2)) + torch.nn.functional.cross_entropy(classification, bd)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
        history.append({"epoch": epoch, "loss": loss_sum / len(loader)})
        print(f"PRETRAIN ep={epoch} loss={history[-1]['loss']:.6f}", flush=True)
    torch.save(pre.state_dict(), out / "pretrain_checkpoint.pt")
    write_json(out / "pretrain_history.json", history)
    transfer = net.net.encoder.load_state_dict(pre.encoder.state_dict(), strict=True)
    assert not transfer.missing_keys and not transfer.unexpected_keys
    for parameter in net.net.encoder.parameters():
        parameter.requires_grad_(False)
    del pre
    torch.cuda.empty_cache()


def folder_for(root, model, subset, seed):
    return Path(root).resolve() / "matched" / model / subset / f"s{seed}"


def split_metadata(data):
    if "scaler" in data and isinstance(data["scaler"], dict):
        return {k: data[k] for k in ["train_units", "val_units", "scaler"]}
    s = data["scaler"]
    return {"train_units": data["train_units"], "val_units": data["val_units"],
            "scaler": {"kmeans_centers": s.kmeans.cluster_centers_.tolist(),
                       "sensor_mean": s.means.tolist(), "sensor_scale": s.scales.tolist(),
                       "fit_engine_ids": data["train_units"]}}


def train(model, subset, seed, data_dir, root):
    cfg = configuration(model, subset, seed, data_dir)
    out = folder_for(root, model, subset, seed)
    research_task_start(out, cfg)
    started = time.time()
    seed_everything(seed)
    guard = Guard("train")
    try:
        with guard:
            write_json(out / "environment.json", {"python":sys.version,"torch":torch.__version__,
                       "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"precision":"float32", "autocast_enabled":False})
            if model in REFERENCES:
                net, data, va, pv, history, best_epoch = train_reference(cfg, data_dir, out)
                meta = split_metadata(data)
                n_train = len(data["train"]["y"])
                best = history[best_epoch-1]["val_engine_mean_rmse"]
            else:
                data = prepare(subset, Path(data_dir), cfg)
                meta = split_metadata(data)
                net = UpstreamAdapter(cfg).cuda()
                pretrain_info = None
                if model == "two_phase":
                    t0 = time.time()
                    pretrain_two_phase(net, cfg, data, Path(data_dir), out)
                    paired = prepare(cfg["paired_pretraining_subset"], Path(data_dir), cfg)
                    pretrain_info = {"epochs":cfg["pretraining_epochs"], "training_time_s":time.time()-t0,
                                     "n_target_train_windows":len(data["train"]["y"]),
                                     "n_paired_train_windows":len(paired["train"]["y"]),
                                     "encoder_before_finetune_sha256":state_hash(net.net.encoder.state_dict())}
                tr, va = data["train"], data["val"]
                n_train = len(tr["y"])
                if model == "dual_mixer":
                    cmapss = module("dual_mixer", "dataset/cmapss.py", "upstream_cmapss")
                    dataset = cmapss.Cmapss(tr["x"], tr["unit"], tr["y"] / 125.0)
                    cmapss.CmapssGaussianNegativeSampler(dataset, neg_num=5, thresh=0.5, std=0.3)
                else:
                    dataset = TensorDataset(torch.from_numpy(tr["x"]),torch.from_numpy(tr["y"]),torch.from_numpy(tr["age"]))
                loader = DataLoader(dataset,batch_size=cfg["batch"],shuffle=True,num_workers=0)
                optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=cfg["lr"],weight_decay=cfg["weight_decay"])
                scheduler = None
                if model == "ffve": scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
                if model == "aidgn": scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.3,patience=10)
                best,best_epoch,history = float("inf"),0,[]
                for epoch in range(1,cfg["epochs"]+1):
                    net.train();total=0.0
                    for batch in loader:
                        x,y=batch[0].cuda().float(),batch[1].cuda().float()
                        age=batch[2].cuda() if model!="dual_mixer" else None
                        optimizer.zero_grad(set_to_none=True)
                        loss=net.loss(x,y,age)
                        if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
                        loss.backward();optimizer.step();total+=float(loss.detach())
                    pv=predict(net,va,seed+100000)
                    check=validation_record(va,pv);score=check["val_engine_mean_rmse"]
                    accepted=score<best-cfg["min_delta"]
                    if accepted:
                        best,best_epoch=score,epoch
                        torch.save(net.state_dict(),out/"checkpoint.pt")
                    row={**check,"epoch":epoch,"train_loss":total/len(loader),"accepted":accepted,
                         "selected_epoch_so_far":best_epoch,"state_content_sha256":state_hash(net.state_dict()),
                         "lr":optimizer.param_groups[0]["lr"],"elapsed_s":time.time()-started}
                    history.append(row)
                    print("TRAIN",json.dumps({k:row[k] for k in ["epoch","train_loss","val_engine_mean_rmse","val_pooled_rmse","selected_epoch_so_far","elapsed_s"]}),flush=True)
                    if scheduler is not None:
                        scheduler.step(score/net.output_scale**2) if model=="aidgn" else scheduler.step()
                    if epoch-best_epoch>=cfg["patience"]:break
                net.load_state_dict(torch.load(out/"checkpoint.pt",map_location="cuda",weights_only=True))
                pv=predict(net,va,seed+100000)
                if pretrain_info is not None:
                    pretrain_info["encoder_after_finetune_sha256"]=state_hash(net.net.encoder.state_dict())
                    assert pretrain_info["encoder_before_finetune_sha256"]==pretrain_info["encoder_after_finetune_sha256"]
                    write_json(out/"pretraining_metrics.json",pretrain_info)
            write_json(out/"split.json",{k:meta[k] for k in ["train_units","val_units"]})
            write_json(out/"scaler.json",meta["scaler"])
            raw=reference_data._prepare_split(subset,Path(data_dir),cfg["split_seed"],cfg["fold"],cfg["val_frac"],cfg["n_cond"])
            assert n_train==len(raw["tr"])
            expected=reference_data.make_val_windows(raw["va"],raw["va_s"],raw["lv"],raw["pop"],cfg["window"],cuts=tuple(cfg["cuts"]),max_rul=cfg["rul_cap"])
            for key in ["unit","cut","y"]:assert np.array_equal(va[key],expected[key]),key
            assert not set(meta["train_units"])&set(meta["val_units"])
            val=validation_record(va,pv)
            assert abs(best-val["val_engine_mean_rmse"])<1e-7
            predictions(out/"validation_predictions.csv",va["y"],pv,va["unit"],va["cut"])
            write_json(out/"validation_metrics.json",{**metrics(va["y"],pv),**val})
            write_json(out/"history.json",history)
            content=state_hash(net.state_dict())
            assert content==history[best_epoch-1]["state_content_sha256"]
            assert all(p.dtype==torch.float32 for p in net.parameters())
            runtime=backend_flags()
            assert not runtime["matmul_tf32"] and not runtime["cudnn_tf32"]
            result={"model":model,"subset":subset,"seed":seed,"best_epoch":best_epoch,"epochs_run":len(history),
                    "validation_engine_mean_rmse":val["val_engine_mean_rmse"],"validation_pooled_rmse":val["val_pooled_rmse"],
                    "params":sum(p.numel() for p in net.parameters()),"trainable_params":sum(p.numel() for p in net.parameters() if p.requires_grad),
                    "training_time_s":time.time()-started,"n_train_windows":n_train,"n_validation_cuts":len(va["y"]),
                    "config_sha256":config_hash(cfg),"checkpoint_sha256":digest(out/"checkpoint.pt"),
                    "state_content_sha256":content,"runtime_flags":runtime}
            write_json(out/"training_metrics.json",result)
            verify_training_data(cfg, data_dir)
            freeze={"frozen":True,"created_unix":time.time(),"selection":"val_engine_mean_rmse", "best_epoch":best_epoch,
                    "config_sha256":config_hash(cfg),"checkpoint_sha256":result["checkpoint_sha256"],
                    "state_content_sha256":content,"execution_manifest_sha256":cfg["execution_manifest_sha256"],
                    "protocol_sha256":cfg["protocol_sha256"],"test_evaluation_limit":1}
            write_json(out/"FREEZE.json",freeze)
            (out/"RUNNING.json").rename(out/"TRAINED.json")
            print("TRAINED",json.dumps(result),flush=True)
    except Exception:
        (out/"error.txt").write_text(traceback.format_exc())
        if (out/"RUNNING.json").exists():(out/"RUNNING.json").rename(out/"FAILED_TRAIN.json")
        raise
    finally:write_json(out/"train_data_access_audit.json",guard.record)


def test(model,subset,seed,data_dir,root):
    root=Path(root).resolve();out=folder_for(root,model,subset,seed)
    cfg=json.loads((out/"config.json").read_text());expected=cfg.pop("config_sha256")
    assert config_hash(cfg)==expected
    assert verify_sources()==cfg["execution_manifest_sha256"]
    verify_training_data(cfg, data_dir)
    batch=json.loads((root/"BATCH_FREEZE.json").read_text())
    assert batch["frozen"] and len(batch["runs"])==84
    entry=next(r for r in batch["runs"] if (r["model"],r["subset"],r["seed"])==(model,subset,seed))
    assert entry["config_sha256"]==expected and entry["checkpoint_sha256"]==digest(out/"checkpoint.pt")
    assert entry["training_data_sha256"]==cfg["training_data_sha256"] and entry["data_dir"]==cfg["data_dir"]
    assert (out/"TRAINED.json").exists() and not (out/"metrics.json").exists()
    assert not any((out/name).exists() for name in ["TEST_ATTEMPT.json","TEST_CLAIM.json","FAILED_TEST.json","COMPLETED.json"]), "official TEST attempt already consumed"
    with (out/"TEST_ATTEMPT.json").open("x") as f:json.dump({"started_unix":time.time(),"limit":1,"batch_freeze_sha256":digest(root/"BATCH_FREEZE.json")},f)
    with (out/"TEST_CLAIM.json").open("x") as f:json.dump({"started_unix":time.time(),"limit":1},f)
    seed_everything(seed);guard=Guard("test",unlocked=True)
    try:
        with guard:
            dataset_expected=json.loads((ROOT/"DATASET_EXPECTED_SHA256.json").read_text())
            for name in [f"test_{subset}.txt",f"RUL_{subset}.txt"]:
                assert digest(Path(data_dir)/name)==dataset_expected[name], "official data bytes differ from registered dataset"
            state=torch.load(out/"checkpoint.pt",map_location="cpu",weights_only=True)
            if model in REFERENCES:
                torch.backends.cudnn.benchmark=True
                net=load_reference(cfg,out/"checkpoint.pt")
                data=reference_data.prepare_test(subset,Path(data_dir),span=cfg["window"],split_seed=cfg["split_seed"],fold=cfg["fold"],val_frac=cfg["val_frac"],rul_cap=cfg["rul_cap"],n_cond=cfg["n_cond"])["test"]
                p=predict_reference(net,data,cfg)
            else:
                net=UpstreamAdapter(cfg).cuda();net.load_state_dict(state)
                data=prepare(subset,Path(data_dir),cfg,official_test=True)["test"]
                p=predict(net,data,seed+200000)
            trained=json.loads((out/"training_metrics.json").read_text())
            assert state_hash(net.state_dict())==trained["state_content_sha256"]
            assert len(data["y"])=={"FD001":100,"FD002":259,"FD003":100,"FD004":248}[subset]
            predictions(out/"test_predictions.csv",data["y"],p,data["unit"],data.get("cut"))
            m={**trained,**metrics(data["y"],p),"protocol":"author_limits_20261112","evaluation_unit":"official_test_engine", "test_forward_passes":1,
               "batch_freeze_sha256":digest(root/"BATCH_FREEZE.json"),"test_runtime_flags":backend_flags(),
               "official_test_data_sha256":{name:digest(Path(data_dir)/f"{name}_{subset}.txt") for name in ["test","RUL"]}}
            write_json(out/"metrics.json",m)
            (out/"TEST_CLAIM.json").rename(out/"COMPLETED.json")
            print("TEST_DONE",json.dumps(m),flush=True)
    except Exception:
        (out/"test_error.txt").write_text(traceback.format_exc())
        if (out/"TEST_CLAIM.json").exists():(out/"TEST_CLAIM.json").rename(out/"FAILED_TEST.json")
        raise
    finally:write_json(out/"test_data_access_audit.json",guard.record)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=["train","test"],required=True)
    ap.add_argument("--model",required=True);ap.add_argument("--subset",required=True)
    ap.add_argument("--seed",type=int,required=True);ap.add_argument("--data-dir",required=True);ap.add_argument("--root",required=True)
    a=ap.parse_args();(train if a.phase=="train" else test)(a.model,a.subset,a.seed,a.data_dir,a.root)
