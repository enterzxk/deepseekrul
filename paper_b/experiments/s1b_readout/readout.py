"""S1-B readouts on the frozen F_full_basic_hi1 predictions.

Members for one evaluation point (subset, unit, cut):
  * the three seeds at the frozen window (j = 0), and
  * for the multi-window family, the same seeds at windows ending j cycles
    earlier, whose estimate of the reference RUL is (pred_j - j).

Selection happens on validation engines only. The selection loop computes
validation metrics alone; the test split is scored afterwards, for every
variant, as sensitivity output. Note that this still means 17 variants' test
numbers get computed - the guarantee is that the chosen configuration was fixed
before that happened, not that the test set was read once.

Usage:
  python readout.py --preds preds_fp32.csv --out-json readout.json --out-csv readout.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAP = 125.0
QUANTILES = (0.5, 0.4, 0.3, 0.2)
KSD = (0.25, 0.5, 1.0, 1.5)
JWIN = (3, 5, 10)
DECAYS = (1.0, 0.9)


def nasa_penalty(d):
    d = np.asarray(d, float)
    return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)


def members_for(df_point, mode, j_max=1, decay=1.0):
    """Return (values, weights) estimating the RUL at the reference point."""
    d = df_point[df_point.j <= j_max - 1]
    if mode == 'seeds':
        vals = d[d.j == 0].sort_values('seed').y_pred.to_numpy(float)
        return vals, np.ones_like(vals)
    vals, w = [], []
    for _, r in d.sort_values(['seed', 'j']).iterrows():
        if r.y_pred >= CAP - 1e-9 and r.j != 0:
            continue  # capped predictions carry no information; keep the last window
        vals.append(float(r.y_pred) - float(r.j))
        w.append(float(decay) ** float(r.j))
    return np.asarray(vals), np.asarray(w)


def readout_value(vals, w, kind, param=None):
    if len(vals) == 0:
        return float('nan')
    if kind == 'mean':
        return float(np.average(vals, weights=w))
    if kind == 'median':
        return float(np.median(vals))
    if kind == 'quantile':
        return float(np.quantile(vals, param))
    if kind == 'mean_minus_ksd':
        mu = float(np.average(vals, weights=w))
        sd = float(np.sqrt(np.average((vals - mu) ** 2, weights=w))) if len(vals) > 1 else 0.0
        return mu - float(param) * sd
    if kind == 'nasa_bayes':
        grid = np.arange(0.0, CAP + 0.01, 0.5)
        loss = [np.average(nasa_penalty(g - vals), weights=w) for g in grid]
        return float(grid[int(np.argmin(loss))])
    raise ValueError(kind)


def configs():
    """The frozen variant list, in the order the specification gives them."""
    out = [('mean', 'seeds', 1, 1.0, None), ('median', 'seeds', 1, 1.0, None)]
    out += [('quantile', 'seeds', 1, 1.0, q) for q in QUANTILES]
    out += [('mean_minus_ksd', 'seeds', 1, 1.0, k) for k in KSD]
    out += [('nasa_bayes', 'seeds', 1, 1.0, None)]
    out += [('mean', 'windows', J, dec, None) for J in JWIN for dec in DECAYS]
    return out


def config_name(kind, mode, j_max, decay, param):
    if mode == 'seeds':
        if kind == 'quantile':
            return f'quantile_q{param}'
        if kind == 'mean_minus_ksd':
            return f'mean_minus_{param}sd'
        return kind
    return f'multiwindow_J{j_max}_decay{decay}'


def score_points(df, config, by_seed=False):
    """Apply one readout to every evaluation point, return per-point predictions."""
    kind, mode, j_max, decay, param = config
    keys = ['subset', 'split', 'unit', 'cut_idx'] + (['seed'] if by_seed else [])
    rows = []
    for key, g in df.groupby(keys):
        k = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        members = g if by_seed else g
        vals, w = members_for(members, mode, j_max, decay)
        rows.append(dict(**k,
                         y_true=float(g[g.j == 0].y_true_raw.iloc[0]),
                         y_pred=readout_value(vals, w, kind, param),
                         n_members=int(len(vals))))
    return pd.DataFrame(rows)


def evaluate_split(points, split):
    """Metrics for one split only.

    The split is a required argument so the selection loop can compute
    validation numbers without touching the test split. Scoring both at once
    would mean the test column exists before the choice is fixed, which is the
    thing this module is supposed to avoid.
    """
    p = points[points.split == split]
    if p.empty:
        return {}
    if split == 'val':
        per_engine = p.groupby(['subset', 'unit']).apply(
            lambda g: float(np.sqrt(np.mean((g.y_pred - g.y_true) ** 2))))
        note = 'mean of per-engine RMSE over five cut points'
    else:
        per_engine = (p.y_pred - p.y_true).abs().groupby([p.subset, p.unit]).mean()
        note = ('one prediction per engine, so the per-engine RMSE equals |error| and its '
                'mean equals the MAE; compare RMSE via rmse_pooled')
    err = (p.y_pred - p.y_true).to_numpy(float)
    return dict(
        engine_mean_rmse=float(per_engine.mean()),
        rmse_pooled=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        nasa_sum=float(nasa_penalty(err).sum()),
        nasa_mean=float(nasa_penalty(err).mean()),
        bias=float(err.mean()),
        n_points=int(len(p)),
        n_engines=int(p.groupby(['subset', 'unit']).ngroups),
        engine_metric_note=note,
    )


def per_seed_test(points_seed):
    """Per-seed official-test RMSE, used for the seed-spread part of the S1-B rule."""
    p = points_seed[points_seed.split == 'test']
    rows = []
    for (subset, seed), g in p.groupby(['subset', 'seed']):
        err = (g.y_pred - g.y_true).to_numpy(float)
        rows.append(dict(subset=subset, seed=int(seed),
                         rmse=float(np.sqrt(np.mean(err ** 2))),
                         nasa_sum=float(nasa_penalty(err).sum()),
                         n_engines=int(len(g))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.preds)
    rows, cache = [], {}
    best = None
    for cfg in configs():
        name = config_name(*cfg)
        pts = score_points(df, cfg)
        val = evaluate_split(pts, 'val')   # test is not computed in this loop
        cache[name] = (cfg, pts, val)
        row = dict(config=name,
                   val_engine_mean_rmse=val['engine_mean_rmse'],
                   val_rmse_pooled=val['rmse_pooled'],
                   val_nasa_sum=val['nasa_sum'],
                   val_mae=val['mae'],
                   val_bias=val['bias'])
        rows.append(row)
        if best is None or row['val_engine_mean_rmse'] < best[1]:
            best = (name, row['val_engine_mean_rmse'])
        print(f"{name:28s} val engine-RMSE {row['val_engine_mean_rmse']:.4f} "
              f"pooled {row['val_rmse_pooled']:.4f} NASA {row['val_nasa_sum']:.1f}", flush=True)

    table = pd.DataFrame(rows).sort_values('val_engine_mean_rmse').reset_index(drop=True)
    selected = best[0]

    # Sensitivity: test numbers for every variant, computed after the choice is
    # fixed and never used to change it.
    sens = []
    for name, (cfg, pts, val) in cache.items():
        t = evaluate_split(pts, 'test')
        sens.append(dict(config=name, test_rmse_pooled=t['rmse_pooled'],
                         test_engine_mean_abs=t['engine_mean_rmse'],
                         test_nasa_sum=t['nasa_sum'], test_nasa_mean=t['nasa_mean'],
                         test_mae=t['mae'], test_bias=t['bias']))
    sens = pd.DataFrame(sens)
    table = table.merge(sens, on='config', how='left')
    table['selected'] = table.config == selected
    table.to_csv(a.out_csv, index=False)

    base = 'mean'
    sel = cache[selected]
    base_val = cache[base][2]
    sel_val = sel[2]
    base_test = evaluate_split(cache[base][1], 'test')
    sel_test = evaluate_split(sel[1], 'test')

    # per-seed spread of the selected readout on test
    pts_seed = score_points(df, sel[0], by_seed=True)
    ps = per_seed_test(pts_seed)
    ps.to_csv(a.out_csv.replace('.csv', '_perseed.csv'), index=False)
    sd_by_subset = ps.groupby('subset').rmse.std(ddof=1).to_dict()

    d_nasa = sel_test['nasa_sum'] - base_test['nasa_sum']
    rel_nasa = d_nasa / base_test['nasa_sum']
    d_rmse = sel_test['rmse_pooled'] - base_test['rmse_pooled']
    max_sd = max(sd_by_subset.values()) if sd_by_subset else float('nan')

    payload = dict(
        selection_rule='minimum validation engine-level RMSE over the 17 frozen variants; '
                       'the selection loop computes validation metrics only, and the test '
                       'split is scored afterwards for every variant as sensitivity',
        selected=selected,
        selected_config=dict(kind=sel[0][0], member_mode=sel[0][1], j_max=sel[0][2],
                             decay=sel[0][3], param=sel[0][4]),
        baseline_config=base,
        validation=sel_val,
        baseline_validation=base_val,
        official_test=sel_test,
        baseline_official_test=base_test,
        s1b_criterion=dict(
            nasa_abs_change=d_nasa,
            nasa_relative_change=rel_nasa,
            nasa_reduction_at_least_10pct=bool(rel_nasa <= -0.10),
            rmse_change=d_rmse,
            seed_sd_by_subset=sd_by_subset,
            max_seed_sd=max_sd,
            rmse_within_seed_sd=bool(abs(d_rmse) <= max_sd) if max_sd == max_sd else None,
            met=bool(rel_nasa <= -0.10 and abs(d_rmse) <= max_sd) if max_sd == max_sd else None,
        ),
        n_variants=len(rows),
    )
    Path(a.out_json).write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    print(f'\nSELECTED {selected} (val engine-RMSE {sel_val["engine_mean_rmse"]:.4f})')
    print(f'baseline {base}: val {base_val["engine_mean_rmse"]:.4f} '
          f'test {base_test["rmse_pooled"]:.4f} NASA {base_test["nasa_sum"]:.1f}')
    print(f'selected       : val {sel_val["engine_mean_rmse"]:.4f} '
          f'test {sel_test["rmse_pooled"]:.4f} NASA {sel_test["nasa_sum"]:.1f}')
    print(f'NASA change {d_nasa:+.1f} ({rel_nasa:+.2%}); RMSE change {d_rmse:+.4f} '
          f'vs max seed sd {max_sd:.4f}')


if __name__ == '__main__':
    main()
