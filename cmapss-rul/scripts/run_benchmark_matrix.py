"""Resumable Stage 3 benchmark matrix runner (validation only)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("cnn1d", "bilstm", "gru_attention", "transformer", "tcn")
ABLATIONS = ("wo_agcn", "wo_ssm", "wo_delta", "wo_physics")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def condition_aware(subset: str) -> bool:
    return subset in {"FD002", "FD004"}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_specs(base: dict, gamma: float, tau: float) -> list[tuple[str, str, dict]]:
    common = dict(base)
    common.update(
        {
            "n_splits": 5, "split_seed": 42, "model_seed": 42,
            "eta": 2.0, "tau": tau, "gamma": gamma,
            "base_weight": 1.0, "asymmetric_weight": 0.01,
            "selection_metric": "composite", "lambda_selection": 0.5,
            "output_activation": "linear", "degradation_weighting": "on",
            "num_workers": 4, "device": "cuda",
        }
    )
    specs: list[tuple[str, str, dict]] = []
    for subset in SUBSETS:
        subset_common = {
            **common, "subset": subset,
            "condition_aware": condition_aware(subset),
            "scaler": "standard" if condition_aware(subset) else "minmax",
            "n_conditions": 6,
        }
        full = dict(subset_common)
        full.update(
            {"model": "lite", "epochs": 60, "patience": 15,
             "use_graph": "on", "use_delta": "on",
             "temporal_mode": "ssm", "bidirectional_ssm": "on"}
        )
        specs.append((subset, "full", full))
        for baseline in BASELINES:
            cfg = dict(subset_common)
            cfg.update({"model": baseline, "epochs": 40, "patience": 10})
            specs.append((subset, baseline, cfg))
        for ablation in ABLATIONS:
            cfg = dict(full)
            cfg.update({"epochs": 40, "patience": 10})
            if ablation == "wo_agcn":
                cfg["use_graph"] = "off"
            elif ablation == "wo_ssm":
                cfg["temporal_mode"] = "lstm"
            elif ablation == "wo_delta":
                cfg["use_delta"] = "off"
            elif ablation == "wo_physics":
                cfg.update({"gamma": 1.0, "eta": 0.0,
                            "base_weight": 1.0, "asymmetric_weight": 0.0})
            specs.append((subset, ablation, cfg))
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()

    os.environ["LD_LIBRARY_PATH"] = (
        "/home/yndx/miniconda3/envs/a40_env/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    specs = build_specs(base, args.gamma, args.tau)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    for subset, name, config in specs:
        config_path = args.config_dir / f"{subset}_{name}.json"
        write_json(config_path, config)
        run_root = args.output_root / subset / name
        if (run_root / "summary.json").is_file():
            print(f"SKIP {subset}/{name}", flush=True)
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_5fold_dev.py"),
            "--config", str(config_path),
            "--output-root", str(run_root),
        ]
        started = time.time()
        try:
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            status = "ok"
        except subprocess.CalledProcessError as error:
            status = f"failed:{error.returncode}"
        record = {
            "subset": subset, "name": name, "status": status,
            "elapsed_seconds": time.time() - started,
            "timestamp": time.time(),
            "run_root": str(run_root),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
