"""Reproducible FD001--FD004 ablation runner for STG-MambaNet.

The script trains four variants under one protocol:

- wo_agcn:    spatial branch without adaptive graph coupling
- wo_ssm:     LSTM replaces the multi-scale selective SSM temporal block
- wo_physics: standard MSE replaces PhysicsGuidedLoss
- full:       complete model and PhysicsGuidedLoss

Scientific protocol
-------------------
1. Official training engines are split into train/validation engines.
2. Min-Max statistics are fitted ONLY on the training-engine subset.
3. Model selection and early stopping use validation engines only.
4. The official test set is evaluated once after training.
5. Every run stores config, engine split, scaler statistics, history,
   best checkpoint, test predictions, and metrics.
6. No performance value is hard-coded or fabricated.

Example:
    python scripts/run_ablation.py --subset FD001 --seeds 42 67 123
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import (  # noqa: E402
    CMAPSSDataset,
    OperatingConditionScaler,
    SELECTED_SENSOR_COLUMNS,
    SensorMinMaxScaler,
    build_piecewise_linear_rul,
    compute_nasa_score,
    compute_rmse,
    load_cmapss_files,
)
from baselines import build_baseline  # noqa: E402
from loss import PhysicsGuidedLoss  # noqa: E402
from model import CRGraphTransformer, STG_MambaNet  # noqa: E402


VARIANT_ORDER = (
    "wo_agcn",
    "identity_graph",
    "wo_ssm",
    "wo_physics",
    "wo_mono",
    "full",
    "bilstm",
    "tcn",
    "transformer",
    "p8_wo_graph",
    "p8_wo_phys_mono",
    "p8_conv_ssm",
    "p8_softplus",
    "p8_no_fusion",
    "p8_full",
    "p9_wo_crgcn",
    "p9_wo_synergy_mono",
    "p9_full",
    "p10_wo_crgcn",
    "p10_wo_synergy_mono",
    "p10_full",
)
VARIANT_LABELS = {
    "wo_agcn": "w/o A-GCN",
    "identity_graph": "Identity-Graph (A=I)",
    "wo_ssm": "w/o SSM-Block (matched LSTM)",
    "wo_physics": "w/o Physics Loss",
    "wo_mono": "w/o Mono Head",
    "full": "Full Model (Ours)",
    "bilstm": "Baseline BiLSTM",
    "tcn": "Baseline TCN",
    "transformer": "Baseline Transformer",
    "p8_wo_graph": "Ours w/o Graph (Conv-Transformer)",
    "p8_wo_phys_mono": "Ours w/o Physics+Mono (Pure MSE)",
    "p8_conv_ssm": "Conv-SSM + Lossless Projection",
    "p8_softplus": "Softplus-Mono-Head (P7)",
    "p8_no_fusion": "Ours w/o Cross-Attn Fusion",
    "p8_full": "Ours Full (PA-GraphTransformer)",
    "p9_wo_crgcn": "P9 w/o CR-GCN",
    "p9_wo_synergy_mono": "P9 w/o Synergy+Mono (Pure MSE)",
    "p9_full": "Ours Full (CR-GraphTransformer)",
    "p10_wo_crgcn": "P10 w/o Sensor-Graph",
    "p10_wo_synergy_mono": "P10 w/o Adaptive Tail+Mono",
    "p10_full": "Ours Full (Native Sensor-Graph Transformer)",
}
STG_VARIANTS = {
    "wo_agcn",
    "identity_graph",
    "wo_ssm",
    "wo_physics",
    "wo_mono",
    "full",
    "p8_wo_graph",
    "p8_wo_phys_mono",
    "p8_conv_ssm",
    "p8_softplus",
    "p8_no_fusion",
    "p8_full",
    "p9_wo_crgcn",
    "p9_wo_synergy_mono",
    "p9_full",
    "p10_wo_crgcn",
    "p10_wo_synergy_mono",
    "p10_full",
}
BASELINE_VARIANTS = {"bilstm", "tcn", "transformer"}
VARIANT_CONFIG = {
    "wo_agcn": {
        "spatial_mode": "independent",
        "graph_mode": "identity",
        "temporal_mode": "bilstm_attn",
        "loss_mode": "full",
        "monotonic_head": True,
        "use_delta_features": False,
        "condition_film": True,
    },
    "identity_graph": {
        "spatial_mode": "graph",
        "graph_mode": "identity",
        "temporal_mode": "bilstm_attn",
        "loss_mode": "full",
        "monotonic_head": True,
        "use_delta_features": False,
        "condition_film": True,
    },
    "wo_ssm": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "ssm",
        "loss_mode": "full",
        "monotonic_head": True,
        "use_delta_features": False,
        "condition_film": True,
    },
    "wo_physics": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "bilstm_attn",
        "loss_mode": "mse",
        "monotonic_head": False,
        "use_delta_features": False,
        "condition_film": True,
    },
    "wo_mono": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "bilstm_attn",
        "loss_mode": "asym",
        "monotonic_head": False,
        "use_delta_features": False,
        "condition_film": True,
    },
    "full": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "bilstm_attn",
        "loss_mode": "full",
        "monotonic_head": True,
        "use_delta_features": False,
        "condition_film": True,
    },
    "bilstm": {"loss_mode": "mse", "model_type": "bilstm", "d_model": 100},
    "tcn": {"loss_mode": "mse", "model_type": "tcn", "d_model": 84},
    "transformer": {"loss_mode": "mse", "model_type": "transformer", "d_model": 88},
    # ---- P8: PA-GraphTransformer with lossless terminal-anchored envelope ----
    # ---- P10: native 14x14 sensor-space graph + adaptive tail slope ----
    "p10_full": {
        "model_class": "crgraph",
        "d_model": 88,
        "ff_mult": 1.6,
        "use_crgcn": True,
        "use_synergy": True,
        "native_graph": True,
        "tail_mode": "adaptive",
        "loss_mode": "full",
    },
    "p10_wo_crgcn": {
        "model_class": "crgraph",
        "d_model": 88,
        "ff_mult": 1.6,
        "use_crgcn": False,
        "use_spatial": False,
        "use_synergy": True,
        "native_graph": False,
        "tail_mode": "adaptive",
        "loss_mode": "full",
    },
    "p10_wo_synergy_mono": {
        "model_class": "crgraph",
        "d_model": 88,
        "ff_mult": 1.6,
        "use_crgcn": True,
        "use_synergy": False,
        "native_graph": True,
        "tail_mode": "adaptive",
        "loss_mode": "mse",
    },
    # ---- P9: sequential CR-GCN -> ConvTransformer -> synergized envelope ----
    "p9_full": {
        "model_class": "crgraph",
        "d_model": 80,
        "use_crgcn": True,
        "use_synergy": True,
        "loss_mode": "full",
    },
    "p9_wo_crgcn": {
        "model_class": "crgraph",
        "d_model": 80,
        "use_crgcn": False,
        "use_synergy": True,
        "loss_mode": "full",
    },
    "p9_wo_synergy_mono": {
        "model_class": "crgraph",
        "d_model": 80,
        "use_crgcn": True,
        "use_synergy": False,
        "loss_mode": "mse",
    },
    "p8_full": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "conv_transformer",
        "loss_mode": "full",
        "monotonic_head": False,
        "monotonic_mode": "projection",
        "use_delta_features": False,
        "condition_film": True,
    },
    "p8_wo_graph": {
        "spatial_mode": "independent",
        "graph_mode": "identity",
        "temporal_mode": "conv_transformer",
        "loss_mode": "full",
        "monotonic_head": False,
        "monotonic_mode": "projection",
        "use_delta_features": False,
        "condition_film": True,
    },
    "p8_wo_phys_mono": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "conv_transformer",
        "loss_mode": "mse",
        "monotonic_head": False,
        "monotonic_mode": "free",
        "use_delta_features": False,
        "condition_film": True,
    },
    "p8_conv_ssm": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "ssm",
        "loss_mode": "full",
        "monotonic_head": False,
        "monotonic_mode": "projection",
        "use_delta_features": False,
        "condition_film": True,
    },
    "p8_no_fusion": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "conv_transformer",
        "loss_mode": "full",
        "monotonic_head": False,
        "monotonic_mode": "projection",
        "use_delta_features": False,
        "condition_film": True,
        "fusion_mode": "gated",
    },
    "p8_softplus": {
        "spatial_mode": "graph",
        "graph_mode": "hollow",
        "temporal_mode": "conv_transformer",
        "loss_mode": "full",
        "monotonic_head": True,
        "monotonic_mode": "softplus",
        "use_delta_features": False,
        "condition_film": True,
    },
}


@dataclass
class PreparedSplit:
    train: CMAPSSDataset
    validation: CMAPSSDataset
    test: CMAPSSDataset
    scaler: SensorMinMaxScaler
    train_units: list[int]
    validation_units: list[int]
    test_units: list[int]
    test_raw_targets: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_engines(
    unit_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    units = np.sort(np.asarray(unit_ids, dtype=np.int64))
    if units.size < 2:
        raise ValueError("At least two training engines are needed for a split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")

    n_validation = int(round(units.size * validation_fraction))
    n_validation = min(max(n_validation, 1), units.size - 1)
    rng = np.random.default_rng(seed + 20261020)
    shuffled = rng.permutation(units)
    validation_units = sorted(int(x) for x in shuffled[:n_validation])
    train_units = sorted(int(x) for x in shuffled[n_validation:])
    return train_units, validation_units


def build_windows_for_frame(
    frame: pd.DataFrame,
    sensor_values: np.ndarray,
    window_size: int,
    stride: int,
    max_rul: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rul_per_row = build_piecewise_linear_rul(frame, max_rul=max_rul)
    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []

    for unit_id, group in frame.groupby("unit", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        n_rows = positions.size
        if n_rows < window_size:
            raise ValueError(
                f"Engine {unit_id} has {n_rows} rows, fewer than window_size={window_size}."
            )
        unit_sensors = sensor_values[positions]
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        for start in range(0, n_rows - window_size + 1, stride):
            end = start + window_size - 1
            windows.append(np.ascontiguousarray(unit_sensors[start : end + 1]))
            targets.append(float(rul_per_row[positions[end]]))
            unit_ids.append(int(unit_id))
            end_cycles.append(int(cycles[end]))

    if not windows:
        raise ValueError("No windows were generated.")
    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int64),
        np.asarray(end_cycles, dtype=np.int64),
    )


def build_test_windows(
    frame: pd.DataFrame,
    sensor_values: np.ndarray,
    test_rul: np.ndarray,
    window_size: int,
    max_rul: float,
    clip_test_rul: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    units = sorted(int(x) for x in frame["unit"].unique())
    if len(units) != test_rul.size:
        raise ValueError(
            f"Test engines ({len(units)}) and RUL labels ({test_rul.size}) differ."
        )

    windows: list[np.ndarray] = []
    raw_targets: list[float] = []
    end_cycles: list[int] = []
    observed_lengths: list[int] = []
    for unit_id, rul_value in zip(units, test_rul):
        group = frame.loc[frame["unit"] == unit_id]
        positions = group.index.to_numpy(dtype=np.int64)
        sequence = np.asarray(sensor_values[positions], dtype=np.float32)
        observed_lengths.append(int(sequence.shape[0]))
        if sequence.shape[0] < window_size:
            pad = np.repeat(sequence[:1], window_size - sequence.shape[0], axis=0)
            sequence = np.concatenate([pad, sequence], axis=0)
        windows.append(np.ascontiguousarray(sequence[-window_size:]))
        raw_targets.append(float(rul_value))
        end_cycles.append(int(group["cycle"].iloc[-1]))

    raw = np.asarray(raw_targets, dtype=np.float32)
    targets = np.minimum(raw, float(max_rul)) if clip_test_rul else raw
    return (
        np.stack(windows).astype(np.float32, copy=False),
        targets.astype(np.float32, copy=False),
        raw,
        np.asarray(units, dtype=np.int64),
        np.asarray(end_cycles, dtype=np.int64),
        np.asarray(observed_lengths, dtype=np.int64),
    )


def build_condition_windows(
    frame: pd.DataFrame,
    values: np.ndarray,
    window_size: int,
    stride: int,
    *,
    test_mode: bool = False,
) -> np.ndarray:
    windows: list[np.ndarray] = []
    for _, group in frame.groupby("unit", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        unit_values = np.asarray(values[positions], dtype=np.float32)
        if test_mode:
            if unit_values.shape[0] < window_size:
                pad = np.repeat(unit_values[:1], window_size - unit_values.shape[0], axis=0)
                unit_values = np.concatenate([pad, unit_values], axis=0)
            windows.append(np.ascontiguousarray(unit_values[-window_size:]))
        else:
            for start in range(0, unit_values.shape[0] - window_size + 1, stride):
                windows.append(
                    np.ascontiguousarray(unit_values[start : start + window_size])
                )
    return np.stack(windows).astype(np.float32, copy=False)


def prepare_engine_split(
    data_dir: Path,
    subset: str,
    seed: int,
    validation_fraction: float,
    window_size: int,
    stride: int,
    max_rul: float,
    clip_test_rul: bool,
    condition_aware: bool = False,
    n_conditions: int = 6,
    full_train: bool = False,
) -> PreparedSplit:
    raw_train, test_frame, test_rul = load_cmapss_files(data_dir, subset)
    if full_train:
        # 100% of training engines; a single training engine is still scored
        # every epoch for monitoring, but it is never used for selection.
        train_units = sorted(int(x) for x in raw_train["unit"].unique())
        validation_units = train_units[:1]
    else:
        train_units, validation_units = split_engines(
            raw_train["unit"].unique(), validation_fraction, seed
        )
    train_set = set(train_units)
    validation_set = set(validation_units)

    train_frame = (
        raw_train.loc[raw_train["unit"].isin(train_set)]
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )
    validation_frame = (
        raw_train.loc[raw_train["unit"].isin(validation_set)]
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )
    test_frame = test_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)

    columns = list(SELECTED_SENSOR_COLUMNS)
    condition_values: dict[str, np.ndarray] | None = None
    if condition_aware:
        op_columns = ["op_1", "op_2", "op_3"]
        op_train = train_frame.loc[:, op_columns].to_numpy(dtype=np.float64)
        op_mean = op_train.mean(axis=0)
        op_std = op_train.std(axis=0)
        op_std = np.where(op_std > 1e-8, op_std, 1.0)
        condition_values = {
            "train": ((train_frame.loc[:, op_columns].to_numpy(dtype=np.float64) - op_mean) / op_std).astype(np.float32),
            "validation": ((validation_frame.loc[:, op_columns].to_numpy(dtype=np.float64) - op_mean) / op_std).astype(np.float32),
            "test": ((test_frame.loc[:, op_columns].to_numpy(dtype=np.float64) - op_mean) / op_std).astype(np.float32),
        }
    if condition_aware:
        scaler = OperatingConditionScaler(
            columns, n_clusters=n_conditions, random_state=seed
        )
        train_sensors = scaler.fit_transform(train_frame)
        validation_sensors = scaler.transform(validation_frame)
        test_sensors = scaler.transform(test_frame)
    else:
        scaler = SensorMinMaxScaler(columns)
        train_sensors = scaler.fit_transform(
            train_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )
        validation_sensors = scaler.transform(
            validation_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )
        test_sensors = scaler.transform(
            test_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )

    train_windows = build_windows_for_frame(
        train_frame, train_sensors, window_size, stride, max_rul
    )
    validation_windows = build_windows_for_frame(
        validation_frame, validation_sensors, window_size, stride, max_rul
    )
    test_windows = build_test_windows(
        test_frame,
        test_sensors,
        test_rul,
        window_size,
        max_rul,
        clip_test_rul,
    )

    train_conditions = (
        build_condition_windows(
            train_frame, condition_values["train"], window_size, stride
        )
        if condition_values is not None
        else None
    )
    validation_conditions = (
        build_condition_windows(
            validation_frame, condition_values["validation"], window_size, stride
        )
        if condition_values is not None
        else None
    )
    test_conditions = (
        build_condition_windows(
            test_frame,
            condition_values["test"],
            window_size,
            stride,
            test_mode=True,
        )
        if condition_values is not None
        else None
    )
    train_dataset = CMAPSSDataset(
        train_windows[0],
        train_windows[1],
        train_windows[2],
        train_windows[3],
        conditions=train_conditions,
    )
    validation_dataset = CMAPSSDataset(
        validation_windows[0],
        validation_windows[1],
        validation_windows[2],
        validation_windows[3],
        conditions=validation_conditions,
    )
    test_dataset = CMAPSSDataset(
        test_windows[0],
        test_windows[1],
        test_windows[3],
        test_windows[4],
        observed_lengths=test_windows[5],
        conditions=test_conditions,
    )
    return PreparedSplit(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
        scaler=scaler,
        train_units=train_units,
        validation_units=validation_units,
        test_units=sorted(int(x) for x in test_frame["unit"].unique()),
        test_raw_targets=test_windows[2],
    )


def make_loader(
    dataset: CMAPSSDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 3:
            x, y, cond = batch
            cond = cond.to(device, non_blocking=True)
        else:
            x, y = batch
            cond = None
        x = x.to(device, non_blocking=True)
        try:
            output = model(x, cond=cond, return_trajectory=True)
        except TypeError:
            output = model(x)
        if isinstance(output, tuple):
            prediction, trajectory = output
        else:
            prediction, trajectory = output, None
        predictions.append(prediction.detach().cpu().reshape(-1).numpy())
        targets.append(y.detach().cpu().reshape(-1).numpy())
        if trajectory is not None:
            trajectories.append(trajectory.detach().cpu().numpy())
    trajectory_array = (
        np.concatenate(trajectories) if trajectories else np.empty((0, 0))
    )
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        trajectory_array,
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trajectory: np.ndarray | None = None,
) -> dict[str, float]:
    error = y_pred - y_true
    metrics = {
        "rmse": float(compute_rmse(y_true, y_pred)),
        "nasa_score": float(compute_nasa_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(error))),
        "mean_bias": float(np.mean(error)),
        "late_fraction": float(np.mean(error > 0.0)),
        "severe_late_count_gt10": int(np.sum(error > 10.0)),
        "severe_late_fraction_gt10": float(np.mean(error > 10.0)),
        "severe_late_fraction_gt20": float(np.mean(error > 20.0)),
        "n_test": int(y_true.size),
    }
    if trajectory is not None and trajectory.size > 0 and trajectory.shape[1] > 1:
        increments = np.diff(trajectory, axis=1)
        metrics["mono_violation_rate"] = float(np.mean(increments > 0.0))
    else:
        metrics["mono_violation_rate"] = 0.0
    return metrics


def model_config_from_args(args: argparse.Namespace, variant: str) -> dict:
    variant_cfg = VARIANT_CONFIG[variant]
    if variant_cfg.get("model_class") == "crgraph":
        return {
            "num_sensors": 14,
            "window_size": args.window_size,
            "d_model": variant_cfg.get("d_model", args.d_model),
            "graph_embed_dim": args.graph_embed_dim,
            "num_bases": 3,
            "num_gcn_layers": 1,
            "num_heads": 4,
            "num_transformer_layers": 2,
            "dropout": args.dropout,
            "max_alpha": 0.20,
            "cond_dim": 3,
            "use_crgcn": variant_cfg["use_crgcn"],
            "use_synergy": variant_cfg["use_synergy"],
            "tail_window": 5,
            "ff_mult": variant_cfg.get("ff_mult", 1.8),
            "native_graph": variant_cfg.get("native_graph", False),
            "use_spatial": variant_cfg.get("use_spatial", True),
            "tail_mode": variant_cfg.get("tail_mode", "fixed"),
        }
    if variant in BASELINE_VARIANTS:
        return {
            "num_sensors": 14,
            "d_model": variant_cfg.get("d_model", args.d_model),
            "dropout": args.dropout,
        }
    return {
        "num_sensors": 14,
        "window_size": args.window_size,
        "d_model": args.d_model,
        "graph_embed_dim": args.graph_embed_dim,
        "num_gcn_layers": args.num_gcn_layers,
        "state_dim": args.state_dim,
        "kernels": list(args.kernels),
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "spatial_mode": variant_cfg["spatial_mode"],
        "graph_mode": variant_cfg["graph_mode"],
        "identity_prior_alpha": args.identity_prior_alpha,
        "hollow_alpha": args.hollow_alpha,
        "monotonic_head": variant_cfg["monotonic_head"],
        "monotonic_mode": variant_cfg.get("monotonic_mode", "none"),
        "use_delta_features": variant_cfg["use_delta_features"],
        "condition_film": variant_cfg.get("condition_film", False),
        "conv_transformer_layers": args.conv_transformer_layers,
        "fusion_mode": variant_cfg.get("fusion_mode", "cross"),
        "temporal_mode": variant_cfg["temporal_mode"],
        "lstm_num_layers": args.lstm_num_layers,
        "lstm_hidden_size": args.lstm_hidden_size,
    }


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(
    args: argparse.Namespace,
    prepared: PreparedSplit,
    subset: str,
    variant: str,
    seed: int,
    device: torch.device,
    run_root: Path,
) -> dict:
    set_seed(seed)
    # The dilated-convolution TCN falls back to a ~72x slower deterministic
    # cuDNN kernel; keep bitwise determinism for every other variant and use
    # the fast kernel only for this external baseline (documented in config).
    if variant == "tcn":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    run_dir = run_root / f"seed_{seed}" / variant
    run_dir.mkdir(parents=True, exist_ok=False)

    model_config = model_config_from_args(args, variant)
    if VARIANT_CONFIG[variant].get("model_class") == "crgraph":
        model = CRGraphTransformer(**model_config).to(device)
    elif variant in BASELINE_VARIANTS:
        model = build_baseline(
            VARIANT_CONFIG[variant]["model_type"],
            num_sensors=14,
            d_model=model_config.get("d_model", args.d_model),
            dropout=args.dropout,
        ).to(device)
    else:
        model = STG_MambaNet(**model_config).to(device)
    num_parameters = (
        model.num_parameters
        if hasattr(model, "num_parameters")
        else int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    loss_mode = args.loss_mode or VARIANT_CONFIG[variant]["loss_mode"]
    if loss_mode not in {"mse", "asym", "mono", "full"}:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    use_physics = loss_mode != "mse"
    criterion = PhysicsGuidedLoss(
        mse_weight=args.mse_weight,
        asymmetric_weight=args.asymmetric_weight,
        late_weight=args.late_weight,
        early_weight=args.early_weight,
        use_physics=use_physics,
        use_asym=loss_mode in {"asym", "full"},
        use_mono=loss_mode in {"mono", "full"},
        lambda_mono=args.lambda_mono,
    ).to(device)

    train_loader = make_loader(
        prepared.train,
        args.batch_size,
        True,
        args.num_workers,
        device.type == "cuda",
        seed,
    )
    validation_loader = make_loader(
        prepared.validation,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
        seed,
    )
    test_loader = make_loader(
        prepared.test,
        args.batch_size,
        False,
        args.num_workers,
        device.type == "cuda",
        seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    config = {
        "subset": subset,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "seed": seed,
        "model_config": model_config,
        "loss": criterion.__class__.__name__,
        "loss_mode": loss_mode,
        "use_physics_loss": use_physics,
        "lambda_mono": args.lambda_mono,
        "clip_test_rul": args.clip_test_rul,
        "condition_aware": args.condition_aware,
        "n_conditions": args.n_conditions,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mse_weight": args.mse_weight,
        "asymmetric_weight": args.asymmetric_weight,
        "late_weight": args.late_weight,
        "early_weight": args.early_weight,
        "train_units": prepared.train_units,
        "validation_units": prepared.validation_units,
        "test_units_count": len(prepared.test_units),
        "train_windows": len(prepared.train),
        "validation_windows": len(prepared.validation),
        "test_windows": len(prepared.test),
        "num_parameters": num_parameters,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "full_train": bool(args.full_train),
        "selection_metric": "none (last epoch)" if args.full_train else "validation_rmse",
        "test_usage": "single final evaluation after training",
    }
    write_json(run_dir / "config.json", config)
    if hasattr(prepared.scaler, "state_dict"):
        scaler_record = prepared.scaler.state_dict()
    else:
        scaler_record = prepared.scaler.feature_range()
    write_json(run_dir / "scaler_stats.json", scaler_record)

    history: list[dict] = []
    best_rmse = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()
        for batch in train_loader:
            if len(batch) == 3:
                x, y, cond = batch
                cond = cond.to(device, non_blocking=True)
            else:
                x, y = batch
                cond = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if variant in BASELINE_VARIANTS:
                prediction = model(x)
                trajectory = None
            else:
                prediction, trajectory = model(x, cond=cond, return_trajectory=True)
            loss = criterion(prediction, y, y_pred_traj=trajectory)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = x.shape[0]
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size
        scheduler.step()

        validation_prediction, validation_target, validation_trajectory = predict(
            model, validation_loader, device
        )
        validation_metrics = regression_metrics(
            validation_target, validation_prediction, validation_trajectory
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "validation_rmse": validation_metrics["rmse"],
            "validation_nasa_score": validation_metrics["nasa_score"],
            "validation_mae": validation_metrics["mae"],
            "validation_mean_bias": validation_metrics["mean_bias"],
            "validation_late_fraction": validation_metrics["late_fraction"],
            "validation_severe_late_count_gt10": validation_metrics[
                "severe_late_count_gt10"
            ],
            "validation_mono_violation_rate": validation_metrics[
                "mono_violation_rate"
            ],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"[{subset} {variant} seed={seed}] epoch {epoch:03d} "
            f"loss={row['train_loss']:.4f} "
            f"val_rmse={row['validation_rmse']:.4f} "
            f"val_nasa={row['validation_nasa_score']:.3f} "
            f"time={row['epoch_seconds']:.1f}s",
            flush=True,
        )

        if args.full_train:
            # Fixed budget: monitor only, never select, never early-stop.
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
        elif validation_metrics["rmse"] < best_rmse - args.min_delta:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            torch.save(best_state, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"[{subset} {variant} seed={seed}] early stop at epoch {epoch}",
                    flush=True,
                )
                break

    if args.full_train:
        # Last-epoch weights are the evaluation weights.
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        torch.save(best_state, run_dir / "final_model.pt")
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected.")
    model.load_state_dict(best_state)

    test_prediction, _, test_trajectory = predict(model, test_loader, device)
    raw_target = np.asarray(prepared.test_raw_targets, dtype=np.float32)
    if raw_target.size != test_prediction.size:
        raise ValueError("Raw test targets and predictions are misaligned.")
    clip_target = np.minimum(raw_target, float(args.max_rul))
    metrics_clip = regression_metrics(
        clip_target, test_prediction, test_trajectory
    )
    metrics_raw = regression_metrics(
        raw_target, test_prediction, test_trajectory
    )
    test_metrics = metrics_clip if args.clip_test_rul else metrics_raw
    test_target = clip_target if args.clip_test_rul else raw_target

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    predictions_frame = pd.DataFrame(
        {
            "unit_id": prepared.test.unit_ids,
            "end_cycle": prepared.test.end_cycles,
            "y_true": test_target,
            "y_true_raw": raw_target,
            "y_pred": test_prediction,
            "signed_error": test_prediction - test_target,
            "mono_violation_rate": float(metrics_clip["mono_violation_rate"]),
        }
    )
    predictions_frame.to_csv(run_dir / "test_predictions.csv", index=False)

    result = {
        "subset": subset,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_rmse": best_rmse,
        "num_parameters": num_parameters,
        "clip_test_rul": bool(args.clip_test_rul),
        "loss_mode": loss_mode,
        "test_metrics": test_metrics,
        "test_metrics_clip125": metrics_clip,
        "appendix_raw": metrics_raw,
        "elapsed_seconds": time.time() - start_time,
    }
    write_json(run_dir / "metrics.json", result)
    return result


def summarize(results: list[dict], run_root: Path) -> None:
    rows: list[dict] = []
    for variant in VARIANT_ORDER:
        variant_results = [r for r in results if r["variant"] == variant]
        if not variant_results:
            continue
        rmses = np.asarray([r["test_metrics"]["rmse"] for r in variant_results])
        nasa = np.asarray([r["test_metrics"]["nasa_score"] for r in variant_results])
        maes = np.asarray([r["test_metrics"]["mae"] for r in variant_results])
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "seeds": len(variant_results),
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=0)),
                "nasa_mean": float(nasa.mean()),
                "nasa_std": float(nasa.std(ddof=0)),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(ddof=0)),
                "params": int(variant_results[0]["num_parameters"]),
            }
        )

    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Variant | RMSE | NASA Score | MAE | Params | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant_label']} | "
            f"{row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f} | "
            f"{row['nasa_mean']:.3f} +/- {row['nasa_std']:.3f} | "
            f"{row['mae_mean']:.3f} +/- {row['mae_std']:.3f} | "
            f"{row['params']:,} | {row['seeds']} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run_root / "summary.json", {"rows": rows})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CMAPSS"
    )
    parser.add_argument(
        "--subset", choices=("FD001", "FD002", "FD003", "FD004"), default="FD001"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANT_ORDER, default=list(VARIANT_ORDER)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-rul", type=float, default=125.0)
    parser.add_argument("--condition-aware", action="store_true", default=False)
    parser.add_argument("--n-conditions", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--graph-embed-dim", type=int, default=16)
    parser.add_argument("--num-gcn-layers", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--lstm-hidden-size", type=int, default=70)
    parser.add_argument("--conv-transformer-layers", type=int, default=2)
    parser.add_argument("--identity-prior-alpha", type=float, default=0.1)
    parser.add_argument("--hollow-alpha", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--asymmetric-weight", type=float, default=1.0)
    parser.add_argument("--late-weight", type=float, default=2.0)
    parser.add_argument("--early-weight", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=0.1)
    parser.add_argument(
        "--loss-mode", choices=("mse", "asym", "mono", "full"), default=None
    )
    parser.add_argument(
        "--clip-test-rul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip test RUL at max_rul for the main table (default: True).",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--full-train",
        action="store_true",
        default=False,
        help="Train on 100% of training engines for a fixed epoch budget; "
        "the last-epoch weights are evaluated once on the test set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_dir or (
        PROJECT_ROOT
        / "experiments"
        / "ablations"
        / f"{args.subset}_{timestamp}"
    )
    if run_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_root}. Use --overwrite explicitly."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    run_config["device_resolved"] = str(device)
    write_json(run_root / "run_config.json", run_config)

    all_results: list[dict] = []
    for seed in args.seeds:
        print(f"Preparing engine-level split for seed={seed}", flush=True)
        prepared = prepare_engine_split(
            data_dir=args.data_dir,
            subset=args.subset,
            seed=seed,
            validation_fraction=args.validation_fraction,
            window_size=args.window_size,
            stride=args.stride,
            max_rul=args.max_rul,
            clip_test_rul=args.clip_test_rul,
            condition_aware=args.condition_aware,
            n_conditions=args.n_conditions,
            full_train=args.full_train,
        )
        write_json(
            run_root / f"split_seed_{seed}.json",
            {
                "train_units": prepared.train_units,
                "validation_units": prepared.validation_units,
                "test_units": prepared.test_units,
                "train_windows": len(prepared.train),
                "validation_windows": len(prepared.validation),
                "test_windows": len(prepared.test),
            },
        )
        for variant in args.variants:
            result = run_one(
                args=args,
                prepared=prepared,
                subset=args.subset,
                variant=variant,
                seed=seed,
                device=device,
                run_root=run_root,
            )
            all_results.append(result)

    summarize(all_results, run_root)
    write_json(run_root / "all_results.json", {"results": all_results})
    print(f"Completed. Results: {run_root / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
