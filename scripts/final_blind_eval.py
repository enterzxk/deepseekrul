"""Final blind test evaluation for a trained 5-fold development run.

Run only after Stage 1--4 are frozen and test files are explicitly unsealed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402
from dataset import prepare_development_fold, prepare_test_dataset_with_scaler  # noqa: E402
from loss import risk_metrics  # noqa: E402
from model import Lite_STG_Mamba  # noqa: E402


def build_model_from_config(config: dict) -> torch.nn.Module:
    model_name = config.get("model", "lite")
    if model_name == "lite":
        return Lite_STG_Mamba(
            num_sensors=14,
            window_size=int(config["window_size"]),
            d_model=int(config["d_model"]),
            graph_embed_dim=int(config["graph_embed_dim"]),
            num_gcn_layers=int(config["num_gcn_layers"]),
            state_dim=int(config["state_dim"]),
            kernels=list(config["kernels"]),
            dropout=float(config["dropout"]),
            target_scale=125.0,
            output_activation=config.get("output_activation", "linear"),
            bidirectional_ssm=config.get("bidirectional_ssm", "on") == "on",
            use_graph=config.get("use_graph", "on") == "on",
            use_delta=config.get("use_delta", "on") == "on",
            temporal_mode=config.get("temporal_mode", "ssm"),
        )
    if model_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}")
    return build_baseline(
        model_name,
        num_sensors=14,
        d_model=int(config["d_model"]),
        dropout=float(config["dropout"]),
    )


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for x, _ in loader:
        x = x.to(device)
        outputs.append(model(x).detach().cpu().reshape(-1).numpy())
    return np.concatenate(outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CMAPSS")
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    fold_dirs = sorted(
        [path for path in args.run_root.glob("fold_*") if (path / "metrics.json").is_file()],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if len(fold_dirs) != 5:
        raise ValueError(f"Expected exactly 5 folds, found {len(fold_dirs)} in {args.run_root}.")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    predictions = []
    reference_dataset = None
    configs = []
    for fold_dir in fold_dirs:
        config = json.loads((fold_dir / "config.json").read_text(encoding="utf-8"))
        fold_index = int(config["fold"])
        dev = prepare_development_fold(
            data_dir=args.data_dir,
            subset=args.subset,
            fold_index=fold_index,
            n_splits=int(config.get("n_splits", 5)),
            seed=int(config.get("split_seed", 42)),
            scaler_type=config.get("scaler", "minmax"),
            window_size=int(config["window_size"]),
            stride=int(config.get("stride", 1)),
            target_scale=125.0,
            condition_aware=bool(config.get("condition_aware", False)),
            n_conditions=int(config.get("n_conditions", 6)),
        )
        test_dataset = prepare_test_dataset_with_scaler(
            args.data_dir,
            args.subset,
            dev.scaler,
            window_size=int(config["window_size"]),
            target_scale=125.0,
        )
        if reference_dataset is None:
            reference_dataset = test_dataset
        else:
            if not np.array_equal(reference_dataset.unit_ids, test_dataset.unit_ids):
                raise RuntimeError("Test engine order differs between folds.")
            if not np.allclose(reference_dataset.targets, test_dataset.targets):
                raise RuntimeError("Test targets differ between folds.")

        model = build_model_from_config(config).to(device)
        state = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        fold_prediction = predict(model, loader, device) * 125.0
        predictions.append(np.clip(fold_prediction, 0.0, 125.0))
        configs.append(config)

    predictions_array = np.stack(predictions, axis=0)  # (5, N)
    ensemble = predictions_array.mean(axis=0)
    true = reference_dataset.targets.astype(np.float64) * 125.0
    metrics = risk_metrics(true, ensemble)
    fold_metrics = [risk_metrics(true, row) for row in predictions_array]

    np.savez_compressed(
        args.output_dir / "fold_predictions.npz",
        fold_predictions=predictions_array,
        ensemble=ensemble,
        y_true=true,
        unit_ids=reference_dataset.unit_ids,
        end_cycles=reference_dataset.end_cycles,
    )
    pd.DataFrame(
        {
            "unit_id": reference_dataset.unit_ids,
            "end_cycle": reference_dataset.end_cycles,
            "y_true": true,
            "y_ensemble": ensemble,
            "signed_error": ensemble - true,
        }
    ).to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(args.output_dir / "per_fold_test_metrics.csv", index=False)
    payload = {
        "subset": args.subset,
        "run_root": str(args.run_root),
        "n_folds": len(fold_dirs),
        "ensemble_metrics": metrics,
        "fold_metrics": fold_metrics,
        "test_files_accessed": True,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
