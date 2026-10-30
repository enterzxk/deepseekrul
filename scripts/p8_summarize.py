"""Aggregate P8 5-seed results into tables, stats, and LaTeX."""

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
    "p8_wo_graph": "Ours w/o Graph",
    "p8_wo_phys_mono": "Ours w/o Physics+Mono",
    "p8_conv_ssm": "Conv-SSM + Envelope",
    "p8_softplus": "Softplus-Mono-Head",
    "p8_full": "Ours Full (PA-GraphTransformer)",
}
ORDER = ["tcn", "bilstm", "transformer", "p8_wo_graph", "p8_wo_phys_mono", "p8_conv_ssm", "p8_softplus", "p8_full"]


def load_subset(subset: str, root: Path) -> dict:
    out = {}
    for path in sorted((root / subset).glob("*_s*/seed_*/*/metrics.json")):
        data = json.loads(path.read_text())
        out.setdefault(data["variant"], {})[data["seed"]] = data
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
    parser.add_argument("--root", default="experiments/p8/final")
    parser.add_argument("--out", default="results/p8")
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

        if "p8_full" in data:
            base_seeds = sorted(data["p8_full"])
            base = [data["p8_full"][s]["test_metrics"]["rmse"] for s in base_seeds]
            md.append("### Paired deltas vs Ours Full (" + subset + ", RMSE; negative = better)")
            md.append("")
            md.append("| Comparison | Delta | 95% CI | Wins | sign-test p |")
            md.append("|---|---:|---|---:|---:|")
            for variant in ORDER:
                if variant == "p8_full" or variant not in data:
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

    (out_dir / "p8_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "p8_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pm = "$\\pm$"
    lines = [
        "% Auto-generated by scripts/p8_summarize.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{C-MAPSS test results (clip125), mean " + pm + " std over 5 seeds. Mono is the test-set monotone violation rate.}",
        "\\label{tab:p8_main}",
        "\\begin{tabular}{llccrrr}",
        "\\toprule",
        "Dataset & Method & RMSE & NASA & MAE & Mono & Params \\\\",
        "\\midrule",
    ]
    for subset in ("FD001", "FD002"):
        rows = summary.get(subset, [])
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
    (out_dir / "p8_table_main.tex").write_text("\n".join(lines), encoding="utf-8")
    print("wrote " + str(out_dir / "p8_summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
