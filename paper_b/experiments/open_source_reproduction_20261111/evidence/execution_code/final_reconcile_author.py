"""Recover explicit engine identities/scales without retraining or test selection.

The first author runs exposed two upstream issues: shuffled test export and
nonstandard target normalization. Retain their raw files and add reconciled
evidence rather than silently treating them as engine/cycle predictions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .common import ROOT, digest, metrics, predictions, seed_everything, write_json
from .adapters import module


def dual_cycles(y, p, unit, data_dir, subset):
    raw = np.loadtxt(Path(data_dir) / f"test_{subset}.txt")
    true_rul = np.loadtxt(Path(data_dir) / f"RUL_{subset}.txt").reshape(-1)
    denoms = {}
    for uid in np.unique(raw[:, 0]).astype(int):
        lifetime = raw[raw[:, 0] == uid, 1].max() + true_rul[uid - 1]
        denoms[int(uid)] = 126.0 if lifetime > 125 else float(lifetime)
    denominators = np.array([denoms[int(uid)] for uid in unit])
    cy, cp = np.asarray(y).reshape(-1) * denominators - 1, np.asarray(p).reshape(-1) * denominators - 1
    expected = []
    # Author samples every complete window of length 30, in engine/cycle order.
    for uid in np.unique(raw[:, 0]).astype(int):
        engine = raw[raw[:, 0] == uid]
        cycles = engine[29:, 1]
        expected.extend(np.minimum(125, engine[:, 1].max() + true_rul[uid - 1] - cycles))
    assert len(expected) == len(cy) and np.allclose(cy, expected, atol=1e-5, rtol=1e-6), "author label/cycle reconciliation failed"
    return cy, cp, denoms


def reconcile(folder, data_dir):
    folder = Path(folder).resolve()
    cfg = json.loads((folder / "config.json").read_text())
    model, subset, seed = cfg["model"], cfg["subset"], cfg["seed"]
    result = json.loads((folder / "metrics.json").read_text())
    if model == "dual_mixer":
        source = next((folder / "runtime_source/train/model_result").iterdir())
        y = np.load(source / "model_test_labels_part1.npy").reshape(-1)
        p = np.load(source / "model_test_output_part1.npy").reshape(-1)
        import pandas as pd
        old = pd.read_csv(folder / "predictions.csv")
        unit = old["unit"].to_numpy()
        predictions(folder / "author_normalized_predictions.csv", y, p, unit)
        cy, cp, denoms = dual_cycles(y, p, unit, data_dir, subset)
        predictions(folder / "cycles_all_window_predictions.csv", cy, cp, unit)
        _, last = np.unique(unit[::-1], return_index=True)
        last = np.sort(len(unit) - 1 - last)
        predictions(folder / "cycles_last_window_predictions.csv", cy[last], cp[last], unit[last])
        corrected = {**result, **metrics(cy, cp), "evaluation_unit": "all_test_windows_author_label_inverse",
                     "normalized_author_rmse": metrics(y, p)["rmse"], "inverse_denominators": denoms,
                     "oracle_denominator_engines": [u for u, d in denoms.items() if d != 126],
                     "prediction_file": "cycles_all_window_predictions.csv",
                     "endpoint_metrics": metrics(cy[last], cp[last]),
                     "endpoint_evaluation_unit": "author_retained_test_engines",
                     "missing_test_engine_ids": sorted(set(range(1, len(denoms) + 1)) - set(map(int, unit))),
                     "true_window_labels_verified": True,
                     "endpoint_prediction_file": "cycles_last_window_predictions.csv",
                     "reconciliation_sha256": digest(__file__)}
    elif model == "two_phase":
        if result.get("engine_order_verified"):
            return result
        seed_everything(seed)
        upstream = ROOT / "upstream/two_phase"
        sys.path.insert(0, str(upstream))
        data_module = module(model, "dataset/CMAPSS/CMAPSSDataset_ft.py", "reconcile_two_phase_data")
        _, dataset = data_module.CMAPSSDataset_ft.get_datasets(str(data_dir), subset, max_rul=125,
                            sequence_len=45, norm_type="z-score", use_max_rul_on_test=True)
        mod = module(model, "Transformer_ft_serialatten_model.py", "reconcile_two_phase_model")
        layers = 1 if subset in ["FD001", "FD003"] else 2
        net = mod.Transformer_ft(17, 16, 45, 32, 32, layers, 2, 0.1 if layers == 1 else 0).cuda()
        net.load_state_dict(torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cuda"))
        net.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False):
                ys.append(y.numpy().reshape(-1))
                ps.append(net(x.cuda()).clamp_min(0).cpu().numpy().reshape(-1))
        y, p = np.concatenate(ys), np.concatenate(ps)
        import pandas as pd
        old = pd.read_csv(folder / "predictions.csv")
        old_pairs = sorted(zip(old.y_true.to_numpy(), old.y_pred.to_numpy()))
        new_pairs = sorted(zip(y, p))
        # Rounding in old CSV is <=1e-7, parallel MHA may add tiny batch error.
        assert np.allclose(np.asarray(old_pairs), np.asarray(new_pairs), atol=2e-4, rtol=1e-5)
        unit = dataset.df.id.unique()
        predictions(folder / "official_engine_predictions.csv", y, p, unit)
        corrected = {**result, **metrics(y, p), "evaluation_unit": "official_test_engine",
                     "prediction_file": "official_engine_predictions.csv", "raw_export_order": "shuffled; row indices were not engine IDs",
                     "multiset_reconciliation_passed": True, "reconciliation_sha256": digest(__file__)}
    else:
        return result
    # Leave the initial export and metric record as explicit withdrawn evidence.
    write_json(folder / "RECONCILED_metrics.json", corrected)
    return corrected


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    print(json.dumps(reconcile(a.run_dir, a.data_dir), indent=2), flush=True)
