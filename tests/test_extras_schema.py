"""Extras schema must be identical across train / val / pseudo-test paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_candidate import build_windows, load_cmapss_train  # noqa: E402
from dataset import SELECTED_SENSOR_COLUMNS, SensorMinMaxScaler  # noqa: E402


def main() -> None:
    frame = load_cmapss_train(ROOT / "data" / "CMAPSS", "FD001")
    cols = list(SELECTED_SENSOR_COLUMNS)
    scaler = SensorMinMaxScaler(cols)
    values = scaler.fit_transform(frame[cols].to_numpy(dtype=np.float64))
    widths = {}
    _, _, e_train = build_windows(frame, values, 50, 1, 125.0)
    widths["train"] = e_train.shape[1]
    for cutoff in (0.5, 0.7, 0.9):
        _, _, e_cut = build_windows(frame, values, 50, 1, 125.0, cutoff=cutoff)
        widths[f"cutoff_{cutoff}"] = e_cut.shape[1]
    _, _, e_test = build_windows(frame, values, 50, 1, 125.0, test_mode=True)
    widths["test_mode"] = e_test.shape[1]
    print(widths)
    assert len(set(widths.values())) == 1, widths
    # 14 baseline + cycle + 6 reliability evidence dims (ORDER_02 spec)
    assert widths["train"] == 14 + 1 + 6, widths
    print("extras schema test passed")


if __name__ == "__main__":
    main()
