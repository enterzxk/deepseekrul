"""Aggregate P10 5-seed results into tables, stats, and LaTeX."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "tcn": "Baseline TCN",
    "bilstm": "Baseline BiLSTM",
    "transformer": "Baseline Transformer",
    "p10_wo_crgcn": "Ours w/o Sensor-Graph",
    "p10_wo_synergy_mono": "Ours w/o Adaptive Tail+Mono",
    "p10_full": "Ours Full (Native Sensor-Graph Transformer)",
}
ORDER = ["tcn", "bilstm", "transformer", "p10_wo_crgcn", "p10_wo_synergy_mono", "p10_full"]


def load_subset(subset: str, root: Path) -> dict:
    out = {}
    for path in sorted((root / subset).glob("*_s*/seed_*/*/metrics.json")):
        data = json.loads(path.read_text())
        out.setdefault(data["variant"], {})[data["seed"]] = data
    return out



def ensemble_rows(subset: str, root: Path, variants) -> list:
    """Average test predictions across seeds (same test order per subset)."""
    import csv as _csv
    out = []
    for variant in variants:
        paths = sorted((root / subset).glob(f"{variant}_s*/seed_*/{variant}/test_predictions.csv"))
        if len(paths) < 2:
            continue
        preds, truth = [], None
        for path in paths:
            with path.open() as fh:
                rows = list(_csv.DictReader(fh))
            preds.append(np.asarray([float(r["y_pred"]) for r in rows]))
            if truth is None:
                truth = np.asarray([float(r["y_true"]) for r in rows])
        mean_pred = np.mean(np.stack(preds), axis=0)
        err = mean_pred - truth
        out.append({
            "variant": variant,
            "label": LABELS.get(variant, variant) + " (5-seed ensemble)",
            "n": len(preds),
            "rmse_mean": float(np.sqrt(np.mean(err ** 2))),
            "nasa_mean": float(np.sum(np.where(err < 0, np.expm1(-err / 13.0), np.expm1(err / 10.0)))),
            "mae_mean": float(np.mean(np.abs(err))),
            "mono_mean": 0.0,
            "params": 0,
            "seeds": [],
            "rmse_std": 0.0,
            "nasa_std": 0.0,
            "per_seed_rmse": {},
        })
    return out


def paired_stats(a, b, rng, reps: int = 20000):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = a - b
    idx = rng.integers(0, delta.size, size=(reps, delta.size))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int((delta < 0).sum())
    n = int(delta.size)
    k = min(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wins": wins,
        "n": n,
        "sign_test_p": float(p),
        "per_seed": [float(x) for x in delta],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/p10/final")
    parser.add_argument("--out", default="results/p10")
    args = parser.parse_args()
    root = ROOT / args.root
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)

    summary = {}
    md = []
    for subset in ("FD001", "FD002"):
        data = load_subset(subset, root)
        if not data:
            continue
        rows = []
        for variant in ORDER:
            if variant not in data:
                continue
            seeds = sorted(data[variant])
            rmse = [data[variant][s]["test_metrics"]["rmse"] for s in seeds]
            nasa = [data[variant][s]["test_metrics"]["nasa_score"] for s in seeds]
            mae = [data[variant][s]["test_metrics"]["mae"] for s in seeds]
            mono = [data[variant][s]["test_metrics"]["mono_violation_rate"] for s in seeds]
            rows.append({
                "variant": variant,
                "label": LABELS.get(variant, variant),
                "seeds": seeds,
                "n": len(seeds),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_std": float(np.std(rmse)),
                "nasa_mean": float(np.mean(nasa)),
                "nasa_std": float(np.std(nasa)),
                "mae_mean": float(np.mean(mae)),
                "mono_mean": float(np.mean(mono)),
                "params": int(data[variant][seeds[0]]["num_parameters"]),
                "per_seed_rmse": {str(s): float(r) for s, r in zip(seeds, rmse)},
            })
        rows.extend(ensemble_rows(subset, root, [v for v in ORDER if v in data]))
        summary[subset] = rows
        md.append("## " + subset + " 5-seed test results (clip125)")
        md.append("")
        md.append("| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(
                "| " + r["label"] + " | "
                + f"{r['rmse_mean']:.3f} +/- {r['rmse_std']:.3f} | "
                + f"{r['nasa_mean']:.1f} +/- {r['nasa_std']:.1f} | "
                + f"{r['mae_mean']:.3f} | {r['mono_mean']:.3f} | "
                + f"{r['params']:,} | {r['n']} |"
            )
        md.append("")

        if "p10_full" in data:
            base_seeds = sorted(data["p10_full"])
            base = [data["p10_full"][s]["test_metrics"]["rmse"] for s in base_seeds]
            md.append("### Paired deltas vs Ours Full (" + subset + ", RMSE; negative = better)")
            md.append("")
            md.append("| Comparison | Delta | 95% CI | Wins | sign-test p |")
            md.append("|---|---:|---|---:|---:|")
            for variant in ORDER:
                if variant == "p10_full" or variant not in data:
                    continue
                other = [data[variant][s]["test_metrics"]["rmse"] for s in base_seeds if s in data[variant]]
                if len(other) != len(base):
                    continue
                st = paired_stats(other, base, rng)
                summary.setdefault("stats", {}).setdefault(subset, {})[variant] = st
                md.append(
                    "| " + LABELS.get(variant, variant) + " | "
                    + f"{st['mean_delta']:+.3f} | [{st['ci_low']:+.3f}, {st['ci_high']:+.3f}] | "
                    + f"{st['wins']}/{st['n']} | {st['sign_test_p']:.4f} |"
                )
            md.append("")

    (out_dir / "p10_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "p10_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pm = "$\\pm$"
    lines = [
        "% Auto-generated by scripts/p10_summarize.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{C-MAPSS test results (clip125), mean " + pm + " std over 5 seeds. Mono is the test-set monotone violation rate.}",
        "\\label{tab:p10_main}",
        "\\begin{tabular}{llccrrr}",
        "\\toprule",
        "Dataset & Method & RMSE & NASA & MAE & Mono & Params \\\\",
        "\\midrule",
    ]
    for subset in ("FD001", "FD002"):
        rows = [r for r in summary.get(subset, []) if r.get("params", 0) > 0]
        for i, r in enumerate(rows):
            name = r["label"].replace("_", "\\_")
            dataset = subset if i == 0 else ""
            lines.append(
                dataset + " & " + name + " & "
                + f"{r['rmse_mean']:.3f}" + pm + f"{r['rmse_std']:.3f} & "
                + f"{r['nasa_mean']:.1f}" + pm + f"{r['nasa_std']:.1f} & "
                + f"{r['mae_mean']:.3f} & {r['mono_mean']:.3f} & {r['params']:,} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    (out_dir / "p10_table_main.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote " + str(out_dir / "p10_summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
