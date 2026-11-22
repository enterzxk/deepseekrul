"""Execute pinned author programs, keeping runtime fixes separate and auditable.

No author architectures, losses, epochs or preprocessing are rewritten. Copies
are necessary because upstream scripts write into their own source directory.
"""
from __future__ import annotations
import argparse
import ast
import difflib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback

from .common import ROOT, config_hash, digest, metrics, predictions, research_task_start, seed_everything, source_record, write_json


def save_hparams(path, value):
    # Upstream argparse namespaces may contain torch.device / Path objects;
    # preserve their textual value without relaxing numeric metric validation.
    write_json(path, json.loads(json.dumps(value, default=str)))


def replace_file(path, old, new, changes):
    text = path.read_text(encoding="utf-8-sig")
    if old not in text:
        raise ValueError(f"patch context absent: {path}: {old[:80]}")
    revised = text.replace(old, new)
    path.write_text(revised, encoding="utf-8")
    changes.extend(difflib.unified_diff(text.splitlines(True), revised.splitlines(True),
                                     fromfile=str(path.name) + ".upstream", tofile=str(path.name) + ".runtime"))


class MainConstants(ast.NodeTransformer):
    def __init__(self, replacements):
        self.replacements = replacements

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.replacements:
                node.value = ast.parse(self.replacements[target.id], mode="eval").body
        return node


def execute(path, replacements, changes):
    original = path.read_text(encoding="utf-8-sig")
    tree = MainConstants(replacements).visit(ast.parse(original))
    revised = ast.unparse(ast.fix_missing_locations(tree))
    # Record only semantic constant replacements, not a misleading formatting diff.
    changes.append("\nAST main-variable overrides: " + json.dumps(replacements, sort_keys=True) + "\n")
    namespace = {"__name__": "__main__", "__file__": str(path), "__package__": None}
    exec(compile(revised, str(path), "exec"), namespace)
    return namespace


def link_data(work, data):
    for relative in ("CMAPSS", "raw_data", "dataset/CMAPSS", "data", "FC_STGNN/CMAPSSData"):
        folder = work / relative
        folder.mkdir(parents=True, exist_ok=True)
        for source in data.glob("*.txt"):
            target = folder / source.name
            if not target.exists():
                target.symlink_to(source)


def run(model, subset, seed, data_dir, out_root):
    import numpy as np
    import torch
    out = (Path(out_root) / "author" / model / subset / f"s{seed}").resolve()
    cfg = {"protocol": "author", "model": model, "subset": subset, "seed": seed,
           "source": source_record(model), "data_dir": str(Path(data_dir).resolve()),
           "runner_sha256": digest(__file__), "test_order": "sequential (evaluation ordering only)"}
    cfg["official_data_sha256"] = {name: digest(Path(data_dir) / f"{name}_{subset}.txt") for name in ["train", "test", "RUL"]}
    research_task_start(out, cfg)
    started = time.time()
    work = out / "runtime_source"
    shutil.copytree(ROOT / "upstream" / model, work, ignore=shutil.ignore_patterns("__pycache__"))
    link_data(work, Path(data_dir).resolve())
    changes = []
    # Keep all error evidence even when execution cannot reach its final block.
    write_json(out / "invocation.json", {"argv": sys.argv, "python": sys.executable,
                                        "torch": torch.__version__, "cuda": torch.version.cuda,
                                        "device": torch.cuda.get_device_name(0)})
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(work))
    seed_everything(seed)
    captured = {}
    states = []

    def profile(frame, event, arg):
        if event != "return":
            return
        name = frame.f_code.co_name
        if model == "aidgn" and name == "test" and "enc_pred" in frame.f_locals:
            loc = frame.f_locals
            captured.update(y=np.array(loc["gt"]), p=np.array(loc["enc_pred"]),
                            net=loc["self"].model, evaluation_unit="official_test_engine")
        if model == "fcstgnn" and name == "Prediction" and arg is not None:
            score = float(np.asarray(arg[0]))
            if score < captured.get("best", float("inf")):
                captured.update(best=score, y=np.array(arg[3]) * 125, p=np.array(arg[2]) * 125,
                                net=frame.f_locals["self"].net, evaluation_unit="official_test_engine")
                torch.save(captured["net"].state_dict(), out / "checkpoint.pt")

    try:
        if model == "aidgn":
            # Eager import of unrelated AIGCN is broken upstream (file absent).
            # Load only the requested AIDGN architecture; its code is untouched.
            init = work / "Model/__init__.py"
            old = init.read_text(encoding="utf-8")
            init.write_text("from Model.AIDGN import AIDGN\n", encoding="utf-8")
            changes.extend(difflib.unified_diff(old.splitlines(True), init.read_text().splitlines(True),
                                             fromfile="Model/__init__.py.upstream", tofile="Model/__init__.py.runtime"))
            entry = work / ("FD001_3_train.py" if subset in ("FD001", "FD003") else "FD002_4_train.py")
            # Remove imports of unselected task implementations; no task logic changes.
            for line in ("from Experiment.Experiment_PINN import Exp_PINN", "from Experiment.Experiment_DA import Exp_DA"):
                replace_file(entry, line, "# optional unselected task import omitted: " + line, changes)
            replace_file(work / "Experiment/Experiment.py",
                         "from N_CMAPSS_Related.N_CMAPSS_load_data import get_n_cmapss_data_, N_CMAPSSData",
                         "# unselected N-CMAPSS import omitted; C-MAPSS implementation unchanged", changes)
            sys.argv = [str(entry), "--Data_id_CMAPSS", subset, "--Data_id_CMAPSS_test", subset,
                        "--seed", str(seed), "--save_path", "paper_b_reproduction"]
            os.chdir(work)
            sys.setprofile(profile)
            ns = execute(entry, {}, changes)
            save_hparams(out / "effective_hparams.json", vars(ns["args"]))
            sys.setprofile(None)
        elif model == "two_phase":
            os.chdir(work)
            sys.argv = ["Transformer_pre.py"]
            replace_file(work / "Transformer_ft_serialatten.py",
                         "test_loader = DataLoader(test_dataset, **kwargs)",
                         "test_loader = DataLoader(test_dataset, **{**kwargs, 'shuffle': False})", changes)
            pre = execute(work / "Transformer_pre.py", {"SEED": str(seed), "sub_dataset": repr(subset)}, changes)
            names = ["SEED", "sub_dataset", "lr", "batch_size", "num_hidden", "ffn_hidden", "mlp_size", "n_heads", "seq_len", "mask_p", "random_p", "dropout", "epochs", "encoder_layers", "pre_model", "input_size", "max_rul", "n_type"]
            write_json(out / "pretrain_hparams.json", {key: pre[key] for key in names})
            torch.save(pre["model"].state_dict(), out / "pretrain_checkpoint.pt")
            ft = execute(work / "Transformer_ft_serialatten.py", {"SEED": str(seed), "sub_dataset": repr(subset)}, changes)
            write_json(out / "effective_hparams.json", {key: ft[key] for key in names if key in ft})
            captured.update(y=ft["Y_numpy"], p=ft["prediction_numpy"], net=ft["model"],
                            unit=ft["test_dataset"].df["id"].unique(), evaluation_unit="official_test_engine")
        elif model == "dual_mixer":
            sys.path.insert(0, str(work / "models/RULPrediction"))
            # TSNE is a visualization callback, unused by forward/loss/optimizer.
            entry = work / "models/RULPrediction/experiments.py"
            replace_file(entry, "net.set_visual_samples(visual_sample)", "net.set_visual_samples(None)", changes)
            replace_file(work / "train/trainable.py", "self.train_end()  # callback function",
                         "# plotting-only train_end callback omitted", changes)
            os.chdir(work)
            ns = execute(entry, {"dataset": f"cmapss.Subset.{subset}", "contra_training": "True"}, changes)
            write_json(out / "effective_hparams.json", {**{key: ns[key] for key in ["length", "negs", "bs", "mixer_layer_num", "hidden_dim", "dropout", "label_norm", "contra_training"]},
                        "subset": subset, "lr": 0.001, "epochs": 100, "patience": 5, "negative_thresh": 0.5, "negative_std": 0.3})
            net = ns["net"]
            folder = Path(net.get_model_result_path())
            norm_y = np.load(folder / "model_test_labels_part1.npy")
            norm_p = np.load(folder / "model_test_output_part1.npy")
            unit = net.test_loader.dataset.ids
            predictions(out / "author_normalized_predictions.csv", norm_y, norm_p, unit)
            from paper_b.baselines.open_source.reconcile_author import dual_cycles
            cy, cp, denoms = dual_cycles(norm_y, norm_p, unit, cfg["data_dir"], subset)
            captured.update(y=cy, p=cp, net=net, unit=unit,
                            evaluation_unit="all_test_windows_author_label_inverse", inverse_denominators=denoms,
                            normalized_author_rmse=metrics(norm_y, norm_p)["rmse"])
        elif model == "ffve":
            replace_file(work / "models/model_FFVE.py", "seed = 99", f"seed = {seed}", changes)
            os.chdir(work)
            sys.argv = ["main.py", "--dataset", subset, "--itr", "1", "--save_dir", str(out / "author_output"),
                        "--lr", "0.005" if subset == "FD001" else "0.001",
                        "--e_layers", "2" if subset in ("FD002", "FD004") else "1",
                        "--long_res", "1" if subset == "FD004" else "0"]
            ns = execute(work / "main.py", {}, changes)
            write_json(out / "effective_hparams.json", {**vars(ns["args"]), "window": ns["sequence_length"], "smoothing_alpha": ns["alpha"], "max_epochs": 100000, "epochs_run": ns["epoch"] + 1})
            torch.save({"encoder": ns["encoder"].state_dict(), "decoder": ns["decoder"].state_dict()}, out / "checkpoint.pt")
            captured.update(y=np.asarray(ns["y_test"]), p=ns["y_test_hat"].detach().cpu().numpy(),
                            evaluation_unit="official_test_engine_stochastic_score_draw")
            captured["author_reported_rmse"] = float(ns["test_rmse"])
            captured["params"] = sum(p.numel() for net in [ns["encoder"], ns["decoder"]] for p in net.parameters())
        elif model == "fcstgnn":
            work = work / "FC_STGNN"
            os.chdir(work)
            sys.path.insert(0, str(work))
            (work / "experiment").mkdir(exist_ok=True)
            entry = work / "main_RUL.py"
            replace_file(entry, "args = args_config(1, args)", f"args = args_config({int(subset[-1])}, args)", changes)
            sys.argv = ["main_RUL.py", "--seed", str(seed)]
            sys.setprofile(profile)
            ns = execute(entry, {}, changes)
            save_hparams(out / "effective_hparams.json", vars(ns["args"]))
            sys.setprofile(None)
        else:
            raise ValueError(model)
        if "p" not in captured:
            raise RuntimeError("author script completed without capturable predictions")
        if "net" in captured:
            net = captured["net"]
            if model != "fcstgnn":
                torch.save(net.state_dict(), out / "checkpoint.pt")
            captured["params"] = sum(p.numel() for p in net.parameters())
        predictions(out / "predictions.csv", captured["y"], captured["p"], captured.get("unit"))
        result = {**metrics(captured["y"], captured["p"]), "model": model, "subset": subset,
                  "seed": seed, "protocol": "author", "evaluation_unit": captured["evaluation_unit"],
                  "params": captured["params"], "elapsed_s": time.time() - started,
                  "config_sha256": config_hash(cfg), "checkpoint_sha256": digest(out / "checkpoint.pt")}
        if "author_reported_rmse" in captured:
            result["author_reported_rmse_separate_stochastic_draw"] = captured["author_reported_rmse"]
        if model == "two_phase":
            result["engine_order_verified"] = True
        if "inverse_denominators" in captured:
            result["inverse_denominators"] = captured["inverse_denominators"]
            result["oracle_denominator_engines"] = [u for u, d in captured["inverse_denominators"].items() if d != 126]
            result["normalized_author_rmse"] = captured["normalized_author_rmse"]
            result["true_window_labels_verified"] = True
            result["missing_test_engine_ids"] = sorted(set(range(1, len(captured["inverse_denominators"]) + 1)) - set(map(int, captured["unit"])))
        write_json(out / "metrics.json", result)
        (out / "RUNNING.json").rename(out / "COMPLETED.json")
        print("AUTHOR_DONE", json.dumps(result), flush=True)
    except Exception:
        sys.setprofile(None)
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (out / "RUNNING.json").rename(out / "FAILED.json")
        raise
    finally:
        (out / "runtime_changes.patch").write_text("".join(changes), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    run(a.model, a.subset, a.seed, a.data_dir, a.out_root)
