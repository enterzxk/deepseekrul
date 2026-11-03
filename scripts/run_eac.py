"""EAC-RUL gate runner (spec: paper/LSC_DESIGN.md companion task book, 2026-11-03).

Pre-registered success criteria (written before any V1/V6 training run; not to
be modified afterwards):

  Primary (G1): V1 vs V0 on the validation split, FD001+FD002, single model,
  seeds 101/202/303:  mean dRMSE >= 0.3, paired engine-cluster bootstrap 95% CI
  excludes 0, and V6's gain is less than half of V1's.
  Secondary: the short-life group's |bias| must not increase; late and severe
  fractions must not worsen; FD002 must not be worse than V0.

Deviations from the task-book text, recorded here on purpose:
  * late weight: the task book says 2.0; the frozen protocol that produced the
    recorded V0 numbers (experiments/e10/PE_*) used run_candidate's default 1.8.
    The recorded protocol is authoritative (the V0-reproduction test demands it),
    so 1.8 is used everywhere and V0/V1/V6 always share it.
  * "FiLM parameters L2 (1e-4)" is implemented as AdamW weight_decay=1e-4, which
    already covers the FiLM head; no second penalty term is added.

The official test files are never opened.  Validation split only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS, OperatingConditionScaler, SensorMinMaxScaler,
    load_cmapss_train, compute_rmse,
)
from candidates.model import CandidateModel  # noqa: E402
from eac import SensorInFiLM, derangement, install_eac  # noqa: E402
import run_candidate as RC  # noqa: E402

WINDOW_DEFAULT = 50
K_DEFAULT = 30
LATE_WEIGHT = 1.8          # recorded frozen protocol (see docstring)
SEEDS_GATE = (101, 202, 303)

GATE_CRITERIA = {
    "primary": "V1 vs V0 mean dRMSE >= 0.3 on validation, paired engine-cluster "
               "bootstrap 95% CI excludes 0, and V6 gain < half of V1 gain",
    "secondary": "short-life |bias| not increased; late/severe not worsened; "
                 "FD002 not worse than V0",
    "negative": "if not met, report as a negative result; criteria are frozen",
}


class EACWindowSet(Dataset):
    """WindowSet + per-window early-history tensor E (n, k, N)."""

    def __init__(self, x, y, cond, extras, e):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.cond = torch.from_numpy(cond) if cond is not None else None
        self.extras = torch.from_numpy(extras)
        self.e = torch.from_numpy(e)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        cond = self.cond[index] if self.cond is not None else torch.zeros(50, 3)
        return self.x[index], self.y[index], cond, self.extras[index], self.e[index]


def rebuild_split(subset: str, split_seed: int, fold: int):
    """Replicate prepare()'s engine split exactly (asserted downstream)."""
    if "," in subset:
        parts = [x.strip() for x in subset.split(",") if x.strip()]
        frames = []
        for i, sub in enumerate(parts):
            f = load_cmapss_train(ROOT / "data" / "CMAPSS", sub).copy()
            f["unit"] = f["unit"].astype(int) + i * 10000
            f["source_subset"] = sub
            frames.append(f)
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)
    units = np.sort(raw["unit"].unique())
    rng = np.random.default_rng(split_seed + 20261020)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    if fold >= 0:
        start = fold * n_val
        val_units = sorted(int(x) for x in shuffled[start:start + n_val])
    else:
        val_units = sorted(int(x) for x in shuffled[:n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    train_frame = raw[raw.unit.isin(train_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    val_frame = raw[raw.unit.isin(val_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    return train_frame, val_frame


def scaled_values(train_frame, val_frame, subset: str, split_seed: int):
    """Replicate prepare()'s scaler exactly (same args, same seed)."""
    cols = list(SELECTED_SENSOR_COLUMNS)
    if any(x in subset for x in ("FD002", "FD004")):
        scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=split_seed)
        tr_s = scaler.fit_transform(train_frame)
        va_s = scaler.transform(val_frame)
    else:
        scaler = SensorMinMaxScaler(cols)
        tr_s = scaler.fit_transform(train_frame[cols].to_numpy(np.float64))
        va_s = scaler.transform(val_frame[cols].to_numpy(np.float64))
    return np.asarray(tr_s, dtype=np.float32), np.asarray(va_s, dtype=np.float32)


def window_counts(lengths: np.ndarray, window: int) -> np.ndarray:
    """Both window builders emit exactly (length - window + 1) windows per unit
    for the full-window train/val path; asserted against the real sets."""
    return np.maximum(0, lengths - window + 1)


def build_unit_early(frame, values: np.ndarray, k: int):
    """E per unit: the first min(k, len) rows of that unit's scaled values.

    Units shorter than k are padded by repeating their FIRST row and logged;
    the count of padded units is returned so the caller can record it (spec
    section 2 requires the log).  No RUL or total-life information is read.
    """
    e_by_unit, padded = {}, 0
    for unit_id, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = values[pos]
        n = min(k, seq.shape[0])
        rows = seq[:n]
        if n < k:
            pad = np.repeat(rows[:1], k - n, axis=0)
            rows = np.concatenate([rows, pad], axis=0)
            padded += 1
        e_by_unit[int(unit_id)] = rows.astype(np.float32)
    return e_by_unit, padded


def attach_e(frame, values, window: int, k: int, set_size: int, name: str):
    """Per-window E aligned with the frozen builder's unit-major ordering."""
    lengths = frame.groupby("unit", sort=True).size().to_numpy()
    counts = window_counts(lengths, window)
    e_by_unit, padded = build_unit_early(frame, values, k)
    unit_ids, per_unit = [], []
    for unit_id, group in frame.groupby("unit", sort=True):
        c = int(window_counts(np.array([len(group)]), window)[0])
        if c > 0:
            unit_ids.append(int(unit_id))
            per_unit.append(c)
    total = int(sum(per_unit))
    assert total == set_size, f"{name}: window count {total} != set size {set_size}"
    unit_of_window = np.repeat(np.array(unit_ids, dtype=np.int64), per_unit)
    e = np.stack([e_by_unit[int(u)] for u in unit_of_window]).astype(np.float32)
    print(f"[{name}] units={len(unit_ids)} windows={total} padded_units={padded} "
          f"min_len={int(lengths.min())}", flush=True)
    return e, unit_of_window


def evaluate_eac(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y, cond, extras, e in loader:
        if getattr(model, "sensor_in", None) is not None and \
                isinstance(model.sensor_in, SensorInFiLM):
            model.sensor_in.current_E = e.to(device)
        out = model(x.to(device), cond=cond.to(device), extras=extras.to(device))
        preds.append(out["prediction"].detach().reshape(-1).cpu().numpy())
        targets.append(y.numpy().reshape(-1))
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return float(compute_rmse(t, p)), p, t


def train_one(subset: str, variant: str, seed: int, epochs: int, patience: int,
              fold: int, split_seed: int, window: int, k: int, out_root: str,
              device_str: str = "cuda"):
    assert variant in {"V0", "V1", "V2", "V3", "V4", "V6"}
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    _prep = RC.prepare(
        subset, split_seed, window, 125.0,
        condition_aware=any(x in subset for x in ("FD002", "FD004")), n_cond=6,
        ref_mode="cond", fold=fold, scaler_mode="default",
    )
    train_set, val_set = _prep[0], _prep[1]
    val_unit_ids = _prep[5]

    use_baseline = variant not in {"V2", "V3"}      # V2/V3 drop the handcrafted feature
    use_eac = variant in {"V1", "V2", "V6"}

    tr_frame, va_frame = rebuild_split(subset, split_seed, fold)
    tr_s, va_s = scaled_values(tr_frame, va_frame, subset, split_seed)
    e_tr, unit_tr = attach_e(tr_frame, tr_s, window, k, len(train_set), f"train {subset}")
    e_va, unit_va = attach_e(va_frame, va_s, window, k, len(val_set), f"val {subset}")

    if variant == "V6":
        units_present = sorted(set(unit_tr.tolist()) | set(unit_va.tolist()))
        idx = {u: i for i, u in enumerate(units_present)}
        e_all = {u: (e_tr[unit_tr == u][0] if (unit_tr == u).any()
                     else e_va[unit_va == u][0]) for u in units_present}
        perm = derangement(len(units_present), np.random.default_rng(seed + 777))
        e_all_perm = {u: e_all[units_present[perm[i]]] for i, u in enumerate(units_present)}
        e_tr = np.stack([e_all_perm[int(u)] for u in unit_tr]).astype(np.float32)
        e_va = np.stack([e_all_perm[int(u)] for u in unit_va]).astype(np.float32)

    train_set = EACWindowSet(train_set.x.numpy(), train_set.y.numpy(),
                             train_set.cond.numpy() if train_set.cond is not None else None,
                             train_set.extras.numpy(), e_tr)
    val_set = EACWindowSet(val_set.x.numpy(), val_set.y.numpy(),
                           val_set.cond.numpy() if val_set.cond is not None else None,
                           val_set.extras.numpy(), e_va)

    if variant == "V4":
        # G2 gate: modern backbone (patch + RoPE/SwiGLU), V0 readout kept,
        # no EAC.  Params within +-5% of the PE V0 body (asserted in tests).
        from modern_backbone import PatchRopeModel
        model = PatchRopeModel(
            num_sensors=14, window_size=window, d_model=64, dropout=0.15,
        ).to(device)
        body_params = sum(p.numel() for p in model.parameters())
        film = None
    else:
        model = CandidateModel(
            num_sensors=14, window_size=window, d_model=64, num_layers=1, dropout=0.15,
            use_baseline=use_baseline, use_ema_features=True, use_cycle=True,
            use_pe=True, use_graph=False, use_dual_head=False,
        ).to(device)
        body_params = sum(p.numel() for p in model.parameters())
        film = install_eac(model, 14, 64) if use_eac else None
        if film is not None:
            film.current_E = None
    film_params = sum(p.numel() for p in model.parameters()) - body_params
    model = model.to(device)          # install_eac params must move with the body

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs,
                                                           eta_min=1e-5)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2)

    rng_k = np.random.default_rng(seed + 101)
    best, best_epoch, best_state = float("inf"), -1, None
    history, start = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y, cond, extras, e in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras, e = cond.to(device), extras.to(device), e.to(device)
            if film is not None:
                kp = int(rng_k.integers(20, k + 1))     # training-time k in [20, 30]
                film.current_E = e[:, :kp]
            optimiser.zero_grad(set_to_none=True)
            out = model(x, cond=cond, extras=extras)
            pred = out["prediction"]
            y = y.reshape(-1, 1)
            delta = pred - y
            loss = nn.functional.mse_loss(pred, y)
            late = torch.where(delta > 0, LATE_WEIGHT * delta.square(), delta.square())
            loss = loss + 0.5 * late.mean()
            if film is not None:
                # spec section 2: FiLM parameters carry an extra L2 (1e-4) on
                # top of the shared AdamW weight decay -- the generator can
                # otherwise overfit unit identity on the 100-unit subsets.
                loss = loss + 1e-4 * sum(
                    p.square().sum() for p in film.gen.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()
        if film is not None:
            film.current_E = None
        model.eval()
        val_rmse, _, _ = evaluate_eac(model, val_loader, device)
        eval_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1),
                        "val_rmse": val_rmse})
        print(f"[{subset} {variant} s{seed}] ep{epoch:03d} "
              f"loss={total / max(seen, 1):.3f} val={val_rmse:.4f}", flush=True)
        if val_rmse < best - 1e-4:
            best, best_epoch, best_state = val_rmse, epoch, eval_state
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_rmse, preds, trues = evaluate_eac(model, val_loader, device)
    assert abs(val_rmse - best) < 1e-3 or best_state is None

    out_dir = ROOT / out_root / subset.replace(",", "+") / f"{variant}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": subset, "variant": variant, "seed": seed,
        "use_baseline": use_baseline, "use_eac": use_eac, "eac_k": k,
        "params_body": body_params, "params_film": film_params,
        "best_val_rmse": best, "best_epoch": best_epoch,
        "epochs_ran": len(history), "fold": fold, "split_seed": split_seed,
        "late_weight": LATE_WEIGHT,
        "padded_units_logged": True,
        "elapsed_seconds": time.time() - start,
        "gate_criteria": GATE_CRITERIA,
        "provenance": {
            "code_sha256": {p: _sha(ROOT / p) for p in
                            ("scripts/run_eac.py", "src/eac.py",
                             "src/candidates/model.py", "src/dataset.py",
                             "scripts/run_candidate.py")},
            "command": " ".join([sys.executable] + sys.argv),
            "argv": list(sys.argv),
        },
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(best_state, out_dir / "best_model.pt")
    pd.DataFrame({"unit_id": unit_va, "y_true": trues, "y_pred": preds,
                  "signed_error": preds - trues}).to_csv(
        out_dir / "val_predictions.csv", index=False)
    pd.DataFrame(sorted(set(unit_va.tolist())), columns=["unit_id"]).to_csv(
        out_dir / "val_units.csv", index=False)
    print(f"BEST {variant} {subset} s{seed} val={best:.4f} "
          f"params={body_params + film_params:,} (film {film_params:,})", flush=True)
    return best


def _sha(path: Path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> int:
    import torch.nn as nn  # noqa: F401  (used via nn.functional in train_one)
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--variant", required=True,
                    choices=("V0", "V1", "V2", "V3", "V4", "V6"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/eac")
    ap.add_argument("--repro-v0", action="store_true",
                    help="unit test 5: FD002 PE s2027 fold1 40ep/pat15 must "
                         "reproduce the recorded 12.203 within 0.3")
    args = ap.parse_args()

    if args.repro_v0:
        best = train_one("FD002", "V0", 2027, epochs=40, patience=15, fold=1,
                         split_seed=7, window=50, k=args.k, out_root="results/eac/repro",
                         device_str=args.device)
        recorded = 12.203456924693972
        ok = abs(best - recorded) <= 0.3
        print(f"REPRO V0 FD002 s2027 f1: got {best:.4f} recorded {recorded:.4f} "
              f"-> {'PASS' if ok else 'FAIL'} (tolerance 0.3)")
        return 0 if ok else 1
    train_one(args.subset, args.variant, args.seed, epochs=args.epochs,
              patience=args.patience, fold=args.fold, split_seed=args.split_seed,
              window=args.window, k=args.k, out_root=args.out,
              device_str=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
