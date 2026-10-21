"""C-MAPSS (FD001--FD004) preprocessing and PyTorch dataset construction.

This module implements the standard sensor-only C-MAPSS preprocessing pipeline:

1. Read train/test/RUL files with the canonical 26-column schema.
2. Keep the standard 14 informative sensors (2, 3, 4, 7, 8, 9, 11,
   12, 13, 14, 15, 17, 20, 21).
3. Fit Min-Max statistics on *training sensors only* and transform both
   train and test sets with those statistics.
4. Build piecewise-linear training RUL targets with an early-life plateau
   at ``max_rul`` (default 125).
5. Build all sliding training windows (window_size=30, stride=1) and one
   last-available test window per engine. Short test sequences are padded
   at the beginning by replicating their first frame.
6. Provide RMSE and the NASA asymmetric scoring function.

Important protocol notes
------------------------
* Operating-setting columns are deliberately not used: the model input is
  the standard 14-sensor signal, shape ``(window_size, 14)``.
* Unit/engine identity is never used as a feature. Windows are constructed
  independently within each engine.
* Main protocol: test targets from ``RUL_FDxxx.txt`` are clipped at
  ``max_rul=125`` by default (clip125).  The raw, unclipped labels are kept
  available with ``clip_test_rul=False`` for the appendix only.
* Short *training* engines shorter than ``window_size`` raise an error rather
  than silently changing the training distribution. This situation does not
  occur for the standard FD001--FD004 training files.

Only NumPy, pandas, and PyTorch are required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Canonical C-MAPSS layout: unit, cycle, 3 operating settings, 21 sensors.
CMAPSS_COLUMNS: tuple[str, ...] = (
    "unit",
    "cycle",
    "op_1",
    "op_2",
    "op_3",
    *(f"sensor_{i}" for i in range(1, 22)),
)

# The standard 14 informative sensor channels used by most C-MAPSS studies.
SELECTED_SENSOR_IDS: tuple[int, ...] = (
    2,
    3,
    4,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    17,
    20,
    21,
)
SELECTED_SENSOR_COLUMNS: tuple[str, ...] = tuple(
    f"sensor_{sensor_id}" for sensor_id in SELECTED_SENSOR_IDS
)

VALID_SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_WINDOW_SIZE = 30
DEFAULT_STRIDE = 1
DEFAULT_MAX_RUL = 125.0


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required C-MAPSS file not found: {path}")


def _read_whitespace_table(path: Path) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS table and validate its shape."""
    _require_file(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"{path} has {frame.shape[1]} columns after cleaning; "
            f"expected {len(CMAPSS_COLUMNS)}."
        )

    frame.columns = list(CMAPSS_COLUMNS)
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing or non-numeric values.")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains NaN or infinite values.")

    unit = frame["unit"].to_numpy(dtype=np.float64)
    cycle = frame["cycle"].to_numpy(dtype=np.float64)
    if not np.allclose(unit, np.rint(unit)):
        raise ValueError(f"{path}: unit identifiers must be integers.")
    if not np.allclose(cycle, np.rint(cycle)):
        raise ValueError(f"{path}: cycle values must be integers.")
    if np.any(cycle <= 0):
        raise ValueError(f"{path}: cycle values must be positive.")

    frame["unit"] = frame["unit"].astype(np.int64)
    frame["cycle"] = frame["cycle"].astype(np.int64)

    duplicated = frame.duplicated(subset=["unit", "cycle"], keep=False)
    if duplicated.any():
        bad_units = sorted(frame.loc[duplicated, "unit"].unique().tolist())
        raise ValueError(f"{path}: duplicate (unit, cycle) rows in units {bad_units}.")

    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


def _read_rul_file(path: Path) -> np.ndarray:
    """Read a one-column C-MAPSS test RUL file in engine order."""
    _require_file(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.shape[1] != 1:
        raise ValueError(f"{path} must contain exactly one RUL column.")
    values = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{path} contains no finite RUL values.")
    if np.any(values < 0):
        raise ValueError(f"{path} contains negative RUL values.")
    return values.astype(np.float32)


def load_cmapss_files(
    data_dir: str | Path,
    subset: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and test-RUL files for one C-MAPSS subset."""
    subset = subset.upper()
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset!r}.")

    root = Path(data_dir)
    train = _read_whitespace_table(root / f"train_{subset}.txt")
    test = _read_whitespace_table(root / f"test_{subset}.txt")
    test_rul = _read_rul_file(root / f"RUL_{subset}.txt")

    n_test_units = int(test["unit"].nunique())
    if n_test_units != int(test_rul.size):
        raise ValueError(
            f"{subset}: test file has {n_test_units} engines but RUL file has "
            f"{test_rul.size} values."
        )
    return train, test, test_rul


class SensorMinMaxScaler:
    """Per-sensor Min-Max scaler fitted only on training data.

    A zero-range training sensor is mapped to zero with unit scale. This
    avoids division by zero while remaining deterministic.
    """

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.data_min_ is not None

    def fit(self, values: np.ndarray) -> "SensorMinMaxScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Sensor values must have shape (n_rows, n_features).")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {array.shape[1]}."
            )
        if array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError("Cannot fit scaler on empty or non-finite values.")

        self.data_min_ = array.min(axis=0)
        self.data_max_ = array.max(axis=0)
        span = self.data_max_ - self.data_min_
        self._scale = np.where(span > 0.0, span, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SensorMinMaxScaler must be fitted before transform().")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected shape (n_rows, {len(self.feature_names)}), "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Cannot transform non-finite sensor values.")
        assert self.data_min_ is not None and self._scale is not None
        transformed = (array - self.data_min_) / self._scale
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def feature_range(self) -> dict[str, tuple[float, float]]:
        """Return fitted ranges as an auditable preprocessing record."""
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")
        assert self.data_min_ is not None and self.data_max_ is not None
        return {
            name: (float(lo), float(hi))
            for name, lo, hi in zip(
                self.feature_names, self.data_min_, self.data_max_
            )
        }


def build_piecewise_linear_rul(
    frame: pd.DataFrame,
    max_rul: float = DEFAULT_MAX_RUL,
) -> np.ndarray:
    """Return per-row training RUL = min(max_rul, last_cycle - cycle)."""
    if max_rul <= 0:
        raise ValueError("max_rul must be positive.")
    required = {"unit", "cycle"}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame must contain columns {sorted(required)}.")

    last_cycle = frame.groupby("unit", sort=False)["cycle"].transform("max")
    rul = (last_cycle.to_numpy(dtype=np.float64) - frame["cycle"].to_numpy(dtype=np.float64))
    if np.any(rul < 0):
        raise ValueError("Computed negative training RUL; check cycle ordering.")
    return np.minimum(rul, float(max_rul)).astype(np.float32)


def _last_window_with_edge_padding(
    sequence: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Take the final window, left-padding a short sequence with frame 0."""
    array = np.asarray(sequence)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("A test sequence must be a non-empty 2-D array.")
    if array.shape[0] < window_size:
        pad = np.repeat(array[:1], window_size - array.shape[0], axis=0)
        array = np.concatenate([pad, array], axis=0)
    return np.ascontiguousarray(array[-window_size:], dtype=np.float32)


def _build_train_windows(
    train_frame: pd.DataFrame,
    train_sensors: np.ndarray,
    window_size: int,
    stride: int,
    max_rul: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")

    rul_per_row = build_piecewise_linear_rul(train_frame, max_rul=max_rul)
    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []

    for unit_id, group in train_frame.groupby("unit", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        n_rows = positions.size
        if n_rows < window_size:
            raise ValueError(
                f"Training engine {unit_id} has only {n_rows} cycles, fewer than "
                f"window_size={window_size}; refusing to alter training data."
            )

        sensors = train_sensors[positions]
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        for start in range(0, n_rows - window_size + 1, stride):
            end = start + window_size - 1
            windows.append(np.ascontiguousarray(sensors[start : end + 1]))
            targets.append(float(rul_per_row[positions[end]]))
            unit_ids.append(int(unit_id))
            end_cycles.append(int(cycles[end]))

    if not windows:
        raise ValueError("No training windows were generated.")
    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int64),
        np.asarray(end_cycles, dtype=np.int64),
    )


def _build_test_windows(
    test_frame: pd.DataFrame,
    test_sensors: np.ndarray,
    test_rul: np.ndarray,
    window_size: int,
    clip_test_rul: bool,
    max_rul: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unit_ids_sorted = sorted(int(x) for x in test_frame["unit"].unique())
    if len(unit_ids_sorted) != test_rul.size:
        raise ValueError("Mismatch between test engines and RUL labels.")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    end_cycles: list[int] = []
    observed_lengths: list[int] = []

    for unit_id, rul_value in zip(unit_ids_sorted, test_rul):
        group = test_frame.loc[test_frame["unit"] == unit_id]
        positions = group.index.to_numpy(dtype=np.int64)
        sequence = test_sensors[positions]
        windows.append(_last_window_with_edge_padding(sequence, window_size))
        target = float(rul_value)
        if clip_test_rul:
            target = min(target, float(max_rul))
        targets.append(target)
        end_cycles.append(int(group["cycle"].iloc[-1]))
        observed_lengths.append(int(positions.size))

    return (
        np.stack(windows).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.float32),
        np.asarray(unit_ids_sorted, dtype=np.int64),
        np.asarray(end_cycles, dtype=np.int64),
        np.asarray(observed_lengths, dtype=np.int64),
    )


class CMAPSSDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tensor dataset with input ``(window_size, 14)`` and target ``(1,)``.

    The target is intentionally shaped ``(1,)`` rather than scalar. With a
    model output ``(batch, 1)``, DataLoader targets become ``(batch, 1)`` and
    avoid accidental ``(batch, batch)`` broadcasting during loss computation.
    """

    def __init__(
        self,
        windows: np.ndarray,
        targets: np.ndarray,
        unit_ids: np.ndarray,
        end_cycles: np.ndarray,
        *,
        feature_names: Sequence[str] = SELECTED_SENSOR_COLUMNS,
        observed_lengths: np.ndarray | None = None,
        target_scale: float = 1.0,
        conditions: np.ndarray | None = None,
    ) -> None:
        windows_array = np.ascontiguousarray(windows, dtype=np.float32)
        targets_array = np.asarray(targets, dtype=np.float32).reshape(-1)
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive.")
        targets_array = targets_array / float(target_scale)
        unit_ids_array = np.asarray(unit_ids, dtype=np.int64).reshape(-1)
        end_cycles_array = np.asarray(end_cycles, dtype=np.int64).reshape(-1)

        if windows_array.ndim != 3:
            raise ValueError("windows must have shape (n, window_size, n_features).")
        n = windows_array.shape[0]
        if n == 0:
            raise ValueError("Dataset cannot be empty.")
        if targets_array.size != n or unit_ids_array.size != n or end_cycles_array.size != n:
            raise ValueError("Windows and metadata arrays must have equal length.")
        if windows_array.shape[2] != len(feature_names):
            raise ValueError("Feature dimension does not match feature_names.")
        if not np.isfinite(windows_array).all() or not np.isfinite(targets_array).all():
            raise ValueError("Dataset contains non-finite values.")

        self.windows = windows_array
        self.targets = targets_array
        self.unit_ids = unit_ids_array
        self.end_cycles = end_cycles_array
        self.feature_names = tuple(feature_names)
        self.target_scale = float(target_scale)
        self.observed_lengths = (
            None
            if observed_lengths is None
            else np.asarray(observed_lengths, dtype=np.int64).reshape(-1)
        )
        if conditions is None:
            self.conditions = None
        else:
            conditions_array = np.asarray(conditions, dtype=np.float32)
            if conditions_array.ndim != 3 or conditions_array.shape[0] != n:
                raise ValueError("conditions must have shape (n, window_size, 3).")
            self.conditions = np.ascontiguousarray(conditions_array)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        x = torch.from_numpy(self.windows[index])
        y = torch.from_numpy(np.asarray([self.targets[index]], dtype=np.float32))
        if self.conditions is None:
            return x, y
        cond = torch.from_numpy(self.conditions[index])
        return x, y, cond

    @property
    def window_size(self) -> int:
        return int(self.windows.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.windows.shape[2])


@dataclass
class CMAPSSDataBundle:
    """Prepared datasets plus auditable preprocessing state."""

    train: CMAPSSDataset
    test: CMAPSSDataset
    scaler: SensorMinMaxScaler
    subset: str


def prepare_cmapss_datasets(
    data_dir: str | Path,
    subset: str = "FD001",
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    max_rul: float = DEFAULT_MAX_RUL,
    clip_test_rul: bool = True,
) -> CMAPSSDataBundle:
    """Build leak-aware train/test datasets for one C-MAPSS subset.

    Parameters
    ----------
    data_dir:
        Directory containing ``train_FDxxx.txt``, ``test_FDxxx.txt``, and
        ``RUL_FDxxx.txt``.
    subset:
        One of FD001, FD002, FD003, FD004.
    window_size:
        Number of consecutive cycles in each model input.
    stride:
        Start-index step for training windows. Test always uses one final
        window per engine.
    max_rul:
        Training-target plateau threshold.
    clip_test_rul:
        If True (main protocol default), clip test ground-truth RUL at
        ``max_rul``.  Set False only for the appendix raw-label re-evaluation.
    """
    subset = subset.upper()
    train_frame, test_frame, test_rul = load_cmapss_files(data_dir, subset)

    missing = [c for c in SELECTED_SENSOR_COLUMNS if c not in train_frame.columns]
    if missing:
        raise ValueError(f"Missing selected sensor columns: {missing}")

    # Fit statistics ONLY on training rows, then reuse for test.
    scaler = SensorMinMaxScaler(SELECTED_SENSOR_COLUMNS)
    train_sensors = scaler.fit_transform(
        train_frame.loc[:, SELECTED_SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    )
    test_sensors = scaler.transform(
        test_frame.loc[:, SELECTED_SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    )

    train_windows, train_targets, train_units, train_end_cycles = _build_train_windows(
        train_frame,
        train_sensors,
        window_size=window_size,
        stride=stride,
        max_rul=max_rul,
    )
    (
        test_windows,
        test_targets,
        test_units,
        test_end_cycles,
        observed_lengths,
    ) = _build_test_windows(
        test_frame,
        test_sensors,
        test_rul,
        window_size=window_size,
        clip_test_rul=clip_test_rul,
        max_rul=max_rul,
    )

    train_dataset = CMAPSSDataset(
        train_windows,
        train_targets,
        train_units,
        train_end_cycles,
        feature_names=SELECTED_SENSOR_COLUMNS,
    )
    test_dataset = CMAPSSDataset(
        test_windows,
        test_targets,
        test_units,
        test_end_cycles,
        feature_names=SELECTED_SENSOR_COLUMNS,
        observed_lengths=observed_lengths,
    )
    return CMAPSSDataBundle(
        train=train_dataset,
        test=test_dataset,
        scaler=scaler,
        subset=subset,
    )




def prepare_test_dataset_with_scaler(
    data_dir: str | Path,
    subset: str,
    scaler: object,
    window_size: int = 30,
    target_scale: float = DEFAULT_MAX_RUL,
) -> CMAPSSDataset:
    """Build the official test dataset using an already-fitted train scaler.

    This function is intentionally reserved for the final blind evaluation. It
    must not be called during Stage 1--4 development.
    """
    _, test_frame, test_rul = load_cmapss_files(data_dir, subset)
    test_frame = test_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    if isinstance(scaler, OperatingConditionScaler):
        test_sensors = scaler.transform(test_frame)
    else:
        test_sensors = scaler.transform(
            test_frame.loc[:, list(SELECTED_SENSOR_COLUMNS)].to_numpy(dtype=np.float64)
        )
    windows, targets, units, end_cycles, lengths = _build_test_windows(
        test_frame,
        test_sensors,
        test_rul,
        window_size=window_size,
        clip_test_rul=False,
        max_rul=target_scale,
    )
    return CMAPSSDataset(
        windows,
        targets,
        units,
        end_cycles,
        target_scale=target_scale,
        observed_lengths=lengths,
    )


def _as_flat_float64(values: torch.Tensor | Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    result = np.asarray(array, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError("Metric input cannot be empty.")
    if not np.isfinite(result).all():
        raise ValueError("Metric input contains NaN or infinite values.")
    return result


def _paired_flat_arrays(
    y_true: torch.Tensor | Iterable[float] | np.ndarray,
    y_pred: torch.Tensor | Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = _as_flat_float64(y_true)
    pred = _as_flat_float64(y_pred)
    if true.size != pred.size:
        raise ValueError(
            f"y_true and y_pred must contain the same number of values; "
            f"got {true.size} and {pred.size}."
        )
    return true, pred


def compute_rmse(
    y_true: torch.Tensor | Iterable[float] | np.ndarray,
    y_pred: torch.Tensor | Iterable[float] | np.ndarray,
) -> float:
    """Root mean squared error over all samples."""
    true, pred = _paired_flat_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(pred - true))))


def compute_nasa_score(
    y_true: torch.Tensor | Iterable[float] | np.ndarray,
    y_pred: torch.Tensor | Iterable[float] | np.ndarray,
) -> float:
    """NASA C-MAPSS asymmetric score (lower is better).

    With d = y_pred - y_true:
        score = exp(-d / 13) - 1,  if d < 0  (early prediction)
        score = exp( d / 10) - 1,  if d >= 0 (late prediction)
    The returned value is the sum over samples, following the standard
    C-MAPSS evaluation convention.
    """
    true, pred = _paired_flat_arrays(y_true, y_pred)
    delta = pred - true
    with np.errstate(over="ignore"):
        early = np.expm1(-delta / 13.0)
        late = np.expm1(delta / 10.0)
    scores = np.where(delta < 0.0, early, late)
    return float(np.sum(scores))




def load_train_only(
    data_dir: str | Path,
    subset: str = "FD001",
) -> pd.DataFrame:
    """Load only the official training file.

    This function intentionally has no test/RUL-file access path. Development
    scripts use it during the blind-test-freeze phase.
    """
    subset = subset.upper()
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset!r}.")
    return _read_whitespace_table(Path(data_dir) / f"train_{subset}.txt")


def make_engine_folds(
    unit_ids: np.ndarray | Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Return deterministic engine-level (train, validation) fold pairs."""
    units = np.sort(np.asarray(unit_ids, dtype=np.int64))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if units.size < n_splits:
        raise ValueError("Not enough engines for the requested number of folds.")
    if np.unique(units).size != units.size:
        raise ValueError("unit_ids must be unique.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    fold_assignments = np.array_split(shuffled, n_splits)
    folds: list[tuple[list[int], list[int]]] = []
    for validation in fold_assignments:
        validation_set = set(int(x) for x in validation)
        train = [int(x) for x in units if int(x) not in validation_set]
        validation_ids = [int(x) for x in sorted(validation_set)]
        folds.append((sorted(train), validation_ids))
    return folds


class SensorStandardScaler:
    """Per-sensor Z-score scaler fitted on training rows only."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, values: np.ndarray) -> "SensorStandardScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("Unexpected sensor matrix shape.")
        if array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError("Cannot fit scaler on empty or non-finite values.")
        self.mean_ = array.mean(axis=0)
        std = array.std(axis=0)
        self.scale_ = np.where(std > 0.0, std, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform().")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("Unexpected sensor matrix shape.")
        if not np.isfinite(array).all():
            raise ValueError("Cannot transform non-finite values.")
        assert self.mean_ is not None and self.scale_ is not None
        return ((array - self.mean_) / self.scale_).astype(np.float32, copy=False)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def state_dict(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")
        return {
            "type": "standard",
            "feature_names": list(self.feature_names),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }


class OperatingConditionScaler:
    """Six-condition KMeans + cluster-wise sensor Z-score normalization.

    KMeans is fitted only on the three operating-setting columns of training
    engines. Sensor means/stds are then fitted separately within every
    discovered operating cluster.
    """

    def __init__(
        self,
        feature_names: Sequence[str] = SELECTED_SENSOR_COLUMNS,
        n_clusters: int = 6,
        random_state: int = 42,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.kmeans = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.cluster_counts_: np.ndarray | None = None
        self.global_mean_: np.ndarray | None = None
        self.global_scale_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.means_ is not None

    def fit(self, frame: pd.DataFrame) -> "OperatingConditionScaler":
        from sklearn.cluster import KMeans

        required = {"op_1", "op_2", "op_3", *self.feature_names}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")

        operating = frame.loc[:, ["op_1", "op_2", "op_3"]].to_numpy(dtype=np.float64)
        sensors = frame.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        if sensors.shape[0] == 0 or not np.isfinite(sensors).all():
            raise ValueError("Cannot fit operating-condition scaler on invalid data.")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
        ).fit(operating)
        labels = self.kmeans.labels_
        n_features = sensors.shape[1]
        self.means_ = np.zeros((self.n_clusters, n_features), dtype=np.float64)
        self.scales_ = np.ones((self.n_clusters, n_features), dtype=np.float64)
        self.cluster_counts_ = np.bincount(labels, minlength=self.n_clusters)

        self.global_mean_ = sensors.mean(axis=0)
        global_std = sensors.std(axis=0)
        self.global_scale_ = np.where(global_std > 0.0, global_std, 1.0)

        for cluster in range(self.n_clusters):
            mask = labels == cluster
            if not mask.any():
                self.means_[cluster] = self.global_mean_
                self.scales_[cluster] = self.global_scale_
                continue
            cluster_values = sensors[mask]
            self.means_[cluster] = cluster_values.mean(axis=0)
            std = cluster_values.std(axis=0)
            self.scales_[cluster] = np.where(std > 0.0, std, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.kmeans is None:
            raise RuntimeError("Scaler must be fitted before transform().")
        operating = frame.loc[:, ["op_1", "op_2", "op_3"]].to_numpy(dtype=np.float64)
        sensors = frame.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        labels = self.kmeans.predict(operating)
        assert self.means_ is not None and self.scales_ is not None
        normalized = (sensors - self.means_[labels]) / self.scales_[labels]
        return normalized.astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def state_dict(self) -> dict[str, object]:
        if not self.fitted or self.kmeans is None:
            raise RuntimeError("Scaler has not been fitted.")
        return {
            "type": "operating_condition_zscore",
            "feature_names": list(self.feature_names),
            "n_clusters": self.n_clusters,
            "cluster_centers": self.kmeans.cluster_centers_.tolist(),
            "means": self.means_.tolist(),
            "scales": self.scales_.tolist(),
            "cluster_counts": self.cluster_counts_.tolist(),
        }


def compute_degradation_weights(
    targets: np.ndarray,
    num_bins: int = 5,
) -> np.ndarray:
    """Inverse-frequency sample weights across RUL degradation bins."""
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("targets must be non-empty and finite.")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    bins = np.linspace(float(values.min()), float(values.max()) + 1e-8, num_bins + 1)
    bin_ids = np.clip(np.digitize(values, bins[1:-1]), 0, num_bins - 1)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)
    inverse = np.zeros(num_bins, dtype=np.float64)
    present = counts > 0
    inverse[present] = values.size / (present.sum() * counts[present])
    weights = inverse[bin_ids]
    return (weights / weights.mean()).astype(np.float32)


@dataclass
class DevelopmentFoldBundle:
    train: CMAPSSDataset
    validation: CMAPSSDataset
    scaler: object
    train_units: list[int]
    validation_units: list[int]
    fold_index: int
    n_splits: int


def prepare_development_fold(
    data_dir: str | Path,
    subset: str = "FD001",
    fold_index: int = 0,
    n_splits: int = 5,
    seed: int = 42,
    scaler_type: str = "minmax",
    window_size: int = 30,
    stride: int = 1,
    max_rul: float = DEFAULT_MAX_RUL,
    target_scale: float = DEFAULT_MAX_RUL,
    condition_aware: bool = False,
    n_conditions: int = 6,
) -> DevelopmentFoldBundle:
    """Prepare one blind-development fold using only train_FDxxx.txt.

    No test or RUL file is read. Validation targets are also scaled by
    target_scale when requested.
    """
    scaler_type = scaler_type.lower()
    if scaler_type not in {"minmax", "standard"}:
        raise ValueError("scaler_type must be 'minmax' or 'standard'.")
    raw_train = load_train_only(data_dir, subset)
    folds = make_engine_folds(raw_train["unit"].unique(), n_splits=n_splits, seed=seed)
    if not 0 <= fold_index < len(folds):
        raise IndexError(f"fold_index must be in [0, {len(folds) - 1}].")

    train_units, validation_units = folds[fold_index]
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
    columns = list(SELECTED_SENSOR_COLUMNS)

    if condition_aware:
        scaler: object = OperatingConditionScaler(
            columns, n_clusters=n_conditions, random_state=seed
        )
        train_sensors = scaler.fit_transform(train_frame)
        validation_sensors = scaler.transform(validation_frame)
    elif scaler_type == "minmax":
        scaler = SensorMinMaxScaler(columns)
        train_sensors = scaler.fit_transform(
            train_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )
        validation_sensors = scaler.transform(
            validation_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )
    else:
        scaler = SensorStandardScaler(columns)
        train_sensors = scaler.fit_transform(
            train_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )
        validation_sensors = scaler.transform(
            validation_frame.loc[:, columns].to_numpy(dtype=np.float64)
        )

    train_windows = _build_train_windows(
        train_frame, train_sensors, window_size, stride, max_rul
    )
    validation_windows = _build_train_windows(
        validation_frame, validation_sensors, window_size, stride, max_rul
    )
    train_dataset = CMAPSSDataset(
        train_windows[0],
        train_windows[1],
        train_windows[2],
        train_windows[3],
        target_scale=target_scale,
    )
    validation_dataset = CMAPSSDataset(
        validation_windows[0],
        validation_windows[1],
        validation_windows[2],
        validation_windows[3],
        target_scale=target_scale,
    )
    return DevelopmentFoldBundle(
        train=train_dataset,
        validation=validation_dataset,
        scaler=scaler,
        train_units=train_units,
        validation_units=validation_units,
        fold_index=fold_index,
        n_splits=n_splits,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C-MAPSS preprocessing smoke check and dataset inspection."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=VALID_SUBSETS, default="FD001")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-rul", type=float, default=DEFAULT_MAX_RUL)
    parser.add_argument(
        "--clip-test-rul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip test ground-truth RUL at max_rul (default: True).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    bundle = prepare_cmapss_datasets(
        data_dir=args.data_dir,
        subset=args.subset,
        window_size=args.window_size,
        stride=args.stride,
        max_rul=args.max_rul,
        clip_test_rul=args.clip_test_rul,
    )

    train, test = bundle.train, bundle.test
    print(f"Subset: {bundle.subset}")
    print(f"Selected sensors: {SELECTED_SENSOR_IDS}")
    print(f"Train windows: {len(train)}, shape={tuple(train.windows.shape)}")
    print(f"Test windows:  {len(test)}, shape={tuple(test.windows.shape)}")
    print(f"Train units: {np.unique(train.unit_ids).size}")
    print(f"Test units:  {np.unique(test.unit_ids).size}")
    print(
        "Train target range: "
        f"[{float(train.targets.min()):.3f}, {float(train.targets.max()):.3f}]"
    )
    print(
        "Test target range:  "
        f"[{float(test.targets.min()):.3f}, {float(test.targets.max()):.3f}]"
    )
    print(
        "Transformed train sensor range: "
        f"[{float(train.windows.min()):.4f}, {float(train.windows.max()):.4f}]"
    )

    loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
    x_batch, y_batch = next(iter(loader))
    print(f"One train batch: x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
    print(f"Dtypes: x={x_batch.dtype}, y={y_batch.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())