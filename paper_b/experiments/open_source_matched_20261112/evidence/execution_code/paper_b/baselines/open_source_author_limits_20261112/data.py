"""Training-only Paper B splits/scaling; no test access on the training path."""
from __future__ import annotations
import numpy as np
from .reference.training.data import _prepare_split, load_test


def windows(frame, values, length, average_life, cuts=None, rul_map=None):
    xs, ys, units, ends, ages = [], [], [], [], []
    for unit, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(np.int64)
        seq = values[pos]
        cycles = group.cycle.to_numpy(np.float32)
        n = len(seq)
        if rul_map is not None:
            points = [n]
        elif cuts is not None:
            points = sorted(set(min(n, max(1, round(n * c))) for c in cuts))
        else:
            points = range(1, n + 1)
        for end in points:
            count = min(end, length)
            x = np.zeros((length, values.shape[1]), np.float32)
            age = np.zeros((length, 1), np.float32)
            x[-count:] = seq[end - count:end]
            age[-count:, 0] = cycles[end - count:end] / average_life * 1e-3
            xs.append(x)
            ages.append(age)
            ys.append(min(125.0, float(rul_map[int(unit)]) if rul_map is not None else n - end))
            units.append(int(unit))
            ends.append(int(cycles[end - 1]))
    return {"x": np.stack(xs), "age": np.stack(ages), "y": np.asarray(ys, np.float32),
            "unit": np.asarray(units), "cut": np.asarray(ends)}


def prepare(subset, data_dir, cfg, official_test=False):
    s = _prepare_split(subset, data_dir, split_seed=7, fold=0, val_frac=0.2, n_cond=6)
    tr, va = s["tr"], s["va"]
    average_life = float(tr.groupby("unit").cycle.max().mean())
    tr_values, va_values = s["tr_s"].copy(), s["va_s"].copy()
    opcols = ["op_1", "op_2", "op_3"]
    tr_ops, va_ops = tr[opcols].to_numpy(), va[opcols].to_numpy()
    opmean, opscale = tr_ops.mean(0), tr_ops.std(0)
    opscale = np.where(opscale > 1e-12, opscale, 1.0)

    def transform(frame, sensors):
        if cfg["model"] == "ffve":
            # Preserve the author's causal per-engine exponential smoothing.
            sensors = sensors.copy()
            for _, group in frame.groupby("unit", sort=True):
                indexes = group.index.to_numpy()
                from pandas import DataFrame
                sensors[indexes] = DataFrame(sensors[indexes]).ewm(alpha=0.1).mean().to_numpy()
        if cfg["model"] == "two_phase":
            ops = (frame[opcols].to_numpy() - opmean) / opscale
            return np.concatenate([ops, sensors], axis=-1).astype(np.float32)
        return sensors

    tr_values, va_values = transform(tr, tr_values), transform(va, va_values)
    common = {"train_units": s["train_units"], "val_units": s["val_units"],
              "scaler": {"kmeans_centers": s["scaler"].kmeans.cluster_centers_.tolist(),
                         "sensor_mean": s["scaler"].means.tolist(), "sensor_scale": s["scaler"].scales.tolist(),
                         "operating_mean": opmean.tolist(), "operating_scale": opscale.tolist(),
                         "fit_engine_ids": s["train_units"], "mean_training_lifetime": average_life}}
    assert not set(s["train_units"]) & set(s["val_units"])
    if official_test:
        test, rul_map = load_test(data_dir, subset)
        test = test.sort_values(["unit", "cycle"]).reset_index(drop=True)
        values = transform(test, s["scaler"].transform(test))
        return {"test": windows(test, values, cfg["window"], average_life, rul_map=rul_map), **common}
    return {"train": windows(tr, tr_values, cfg["window"], average_life),
            "val": windows(va, va_values, cfg["window"], average_life,
                           cuts=tuple(cfg["cuts"])), **common}
