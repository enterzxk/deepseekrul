"""Post-failure exploratory gradients on TRAINING engines; zero updates.

This does not change or rescue the registered common module. Evaluation-mode
gradients describe selected saved states, not the stochastic training history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser()
    for name in ("runtime", "cache", "runs", "output", "protocol"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())
    runtime = args.runtime.resolve()
    output = args.output.resolve()
    for protected in (args.runtime, args.cache, args.runs, args.protocol):
        protected = protected.resolve()
        assert not output.is_relative_to(protected) and not protected.is_relative_to(output), "output overlaps protected inputs"
    assert not output.exists(), "diagnostic output exists; preserve prior evidence"
    for name in tuple(sys.modules):
        if name == "paper_b" or name.startswith("paper_b."):
            del sys.modules[name]
    sys.path.insert(0, str(args.runtime.resolve()))
    import numpy as np
    import torch
    from paper_b.ncmapss_response import run as runner
    from paper_b.ncmapss_response.prepare import digest, dump_new
    from paper_b.ncmapss_response.models import ridge_code
    for name, module in tuple(sys.modules.items()):
        if name == "paper_b" or name.startswith("paper_b."):
            file = getattr(module, "__file__", None)
            assert file is not None and Path(file).resolve().is_relative_to(runtime), "mixed frozen-runtime imports"
    assert runner.execution_hash() == spec["frozen_execution_sha256"]
    runtime_manifest = json.loads((args.runtime/"GIT_SOURCE_MANIFEST.json").read_text())
    assert all(digest(args.runtime/n) == h for n, h in runtime_manifest["files_sha256"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    dump_new(args.output/"DIAGNOSTIC_CLAIM.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "protocol": spec, "protocol_sha256": digest(args.protocol), "script_sha256": digest(__file__)})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows, representations = [], []
    started = time.monotonic()
    groups = ("raw_encoder", "response_body", "background", "basis", "aux_context_head", "embedding", "head")
    for subset in spec["subsets"]:
        data, provenance = runner.load_cache(args.cache, subset)
        outer = min(map(int, np.unique(data["unit"])))
        units = sorted(u for u in map(int, np.unique(data["unit"])) if u != outer)
        # Outer-engine arrays never reach the model, target, scaler fit, or query.
        keep = np.isin(data["unit"], units)
        train_data = {key: value[keep] for key, value in data.items()}
        for mode in spec["modes"]:
            folder = args.runs/f"{subset}_{mode}_u{outer}_s101"
            seal = json.loads((folder/"SUCCESS.json").read_text())
            assert all(digest(folder/n) == h for n, h in seal["files_sha256"].items())
            config = json.loads((folder/"config.json").read_text())
            assert config["train_units"] == units and config["seed"] == 101 and config["mode"] == mode
            assert config["held_out_unit"] == outer and config["cell"] == folder.name == f"{subset}_{mode}_u{outer}_s101"
            assert config["execution_sha256"] == spec["frozen_execution_sha256"]
            assert config["source_sha256"] == provenance["source_sha256"] and config["cache_sha256"] == provenance["cache_sha256"]
            protocol = config["protocol"]
            assert protocol == json.loads((runtime/"paper_b/ncmapss_response/PROTOCOL.json").read_text())
            scalers = runner.fit_scalers(train_data, units, protocol["label_cap"])
            assert scalers == json.loads((folder/"scalers.json").read_text())
            training = runner.bundle(train_data, units, scalers, protocol, "cuda:0")
            query = torch.as_tensor(np.linspace(0, len(training["y"])-1, spec["queries"]).round().astype(np.int64), device="cuda:0")
            for state_name, filename, hash_key in (("initial", "initial.pt", "initial_state_sha256"),
                                                  ("final", "checkpoint.pt", "final_state_sha256")):
                checkpoint = torch.load(folder/filename, map_location="cpu", weights_only=True)
                assert checkpoint["config"] == config and checkpoint["scalers"] == scalers
                model = runner.Model(mode, protocol, 101).to("cuda:0")
                model.load_state_dict(checkpoint["model"], strict=True)
                assert runner.state_hash(model) == checkpoint[hash_key]
                model.eval()
                named = list(model.named_parameters())
                params = [p for _, p in named]
                for block in spec["blocks"]:
                    captured = []
                    handle = model.encoder.embedding.register_forward_pre_hook(lambda module, values: captured.append(values[0]))
                    try:
                        pred, auxiliary, diagnostics = model(training, query, block=block)
                    finally:
                        handle.remove()
                    task = (pred-training["target"][query]).square().mean()
                    task_grads = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
                    auxiliary_grads = torch.autograd.grad(auxiliary, params, retain_graph=True, allow_unused=True)
                    input_grads = torch.autograd.grad(task, captured, allow_unused=True)
                    for group in groups:
                        indices = [i for i, (name, _) in enumerate(named) if name.startswith("head." if group == "head" else "encoder."+group+".")]
                        tq = aq = dot = 0.; task_paths = aux_paths = 0
                        for i in indices:
                            gt, ga = task_grads[i], auxiliary_grads[i]
                            if gt is not None:
                                assert bool(torch.isfinite(gt).all())
                                tq += float(gt.double().square().sum()); task_paths += 1
                            if ga is not None:
                                assert bool(torch.isfinite(ga).all())
                                aq += float(ga.double().square().sum()); aux_paths += 1
                            if gt is not None and ga is not None:
                                dot += float((gt.double()*ga.double()).sum())
                        tn, an = tq**.5, aq**.5
                        rows.append({"subset": subset, "mode": mode, "state": state_name, "block": block,
                                     "group": group, "task_gradient_l2": tn, "aux_gradient_l2": an,
                                     "weighted_aux_gradient_l2": protocol["auxiliary_weight"]*an,
                                     "weighted_aux_task_ratio": protocol["auxiliary_weight"]*an/tn if tn > 0 else None,
                                     "cosine_task_aux": dot/(tn*an) if tn > 0 and an > 0 else None,
                                     "task_parameter_paths": task_paths, "aux_parameter_paths": aux_paths,
                                     "task_loss": float(task.detach()), "auxiliary_loss": float(auxiliary.detach()),
                                     "checkpoint_sha256": digest(folder/filename), "training_engines": units,
                                     "outer_engine_excluded": outer, "queries": query.cpu().tolist()})
                    sensitivity = {}
                    for label, lo, hi in (("raw_embedding", 0, 64), ("level", 64, 68), ("innovation", 68, 72), ("coverage", 72, 73)):
                        sensitivity[label] = sum(float(g[:, lo:hi].double().square().sum()) for g in input_grads if g is not None)**.5
                    representations.append({"subset": subset, "mode": mode, "state": state_name, "block": block,
                                            "gradient_wrt_embedding_input_l2": sensitivity,
                                            "level_code_rms": float(diagnostics["level_code"].double().square().mean().sqrt()),
                                            "innovation_rms": float(diagnostics["innovation"].double().square().mean().sqrt())})
                    del pred, auxiliary, diagnostics, task, task_grads, auxiliary_grads, input_grads, captured
                with torch.no_grad():
                    raw = training["raw"]
                    hidden = model.encoder.response_body(raw[..., :4])
                    background = model.encoder.background(hidden)
                    a = torch.tanh(model.encoder.basis(hidden)).reshape(len(raw), 512, 14, 4)/(14.**.5)
                    h, q, _ = ridge_code(a, raw[..., 4:]-background, torch.ones((len(raw), 512), device=raw.device, dtype=torch.bool))
                    eigen = torch.linalg.eigvalsh(q).clamp_min(0).cpu().numpy()
                    dump_new(args.output/f"{subset}_{mode}_{state_name}_representation.json",
                             {"background_residual_rms": float((raw[..., 4:]-background).square().mean().sqrt()),
                              "ridge_code_rms": float(h.square().mean().sqrt()), "gram_eigenvalues": eigen.tolist(),
                              "interpretation": "empirical learned-loading Gram; not W-only excitation or calibrated noise precision"})
                del model, checkpoint, raw, hidden, background, a, h, q
                torch.cuda.empty_cache()
                assert time.monotonic()-started <= spec["maximum_wall_hours"]*3600
            del training
            print(json.dumps({"subset": subset, "mode": mode, "states": 2, "optimizer_updates": 0}), flush=True)
    report = {"status": "PASS" if len(rows) == 4*5*2*4*7 else "PARTIAL", "purpose": "post_failure_exploration_not_module_success",
              "rows": rows, "representations": representations, "protocol": spec, "protocol_sha256": digest(args.protocol),
              "source_sha256": digest(__file__), "seconds": time.monotonic()-started,
              "optimizer_updates": 0, "official_test_read": False, "outer_metrics_computed": False,
              "limitations": ["eval-mode finite-batch gradients, not causal attribution or past stochastic optimizer history",
                              "only seed101 and smallest outer-engine ID, fixed 16 training queries and four blocks",
                              "relative gradients depend on parameterization; correlation or a large norm is not proof of conflict",
                              "raw and FE auxiliary context inference paths differ; do not attribute their whole difference to ridge scale"]}
    dump_new(args.output/"GRADIENT_DIAGNOSTICS.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "representations", "protocol")}), flush=True)


if __name__ == "__main__":
    main()
