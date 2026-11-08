from pathlib import Path
p=Path('PAPER_RESULTS.md'); s=p.read_text()
old = "## A. Main comparison (seed 101)"
new = """## A0. Cross-subset summary (seed 101, M0 vs FULL)

| Subset | Engines | RMSE M0 | RMSE FULL | RMSE rel | MAE rel | NASA rel | Bias M0 | Bias FULL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 100 | 10.274 | 10.235 | -0.38% | -0.70% | -4.34% | +0.801 | -0.329 |
| FD002 | 260 | 13.505 | 13.054 | -3.34% | -2.48% | -12.98% | +1.208 | +0.288 |
| FD003 | 100 | 10.863 | 10.158 | -6.49% | -5.63% | -13.58% | -1.724 | -1.125 |
| FD004 | 249 | 13.398 | 13.253 | -1.08% | -1.68% | **+25.29%** | -0.171 | -1.038 |

FULL improves RMSE on all four subsets, but the NASA score improves on three and
degrades by 25% on FD004. The single-condition subsets (FD001, FD003) show the guardrail
is satisfied with margin. Source: \`paper_tables/cross_subset.csv\`.

## A. Main comparison (seed 101)"""
if old in s and 'A0.' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
