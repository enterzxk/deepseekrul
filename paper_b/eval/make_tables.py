"""Generate LaTeX tables from frozen result JSONs (local)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

T = Path('tables')
T.mkdir(exist_ok=True)
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def write(name, s):
    (T / name).write_text(s)
    print('wrote', T / name)


# Table 1: test
d = json.loads(Path('tmp/claude_pkg/results/LTAG_test_eval.json').read_text())
g = defaultdict(list)
for k, v in d.items():
    g[k.split('_')[0]].append(v)
rows = []
for s in SUBSETS:
    k1 = np.mean([r['k1']['rmse'] for r in g[s]])
    lt = np.mean([r['learned']['rmse'] for r in g[s]])
    md = np.mean([r['median']['rmse'] for r in g[s]])
    wins = sum(r['learned']['rmse'] < r['k1']['rmse'] for r in g[s])
    rows.append((s, len(g[s]), k1, lt, md, k1 - lt, wins))
mean_d = np.mean([r[5] for r in rows])
body = '\\\\\n'.join(
    f'{s} & {n} & {k1:.3f} & \\textbf{{{lt:.3f}}} & {md:.3f} & +{dl:.3f} & {w}/{n}'
    for s, n, k1, lt, md, dl, w in rows)
s = r'''\begin{table*}[t]
\centering
\caption{Frozen B0 official test RMSE (one window per engine; lower is better).
LTAG improves over K1 on every subset and every seed (15/15).}
\label{tab:test}
\begin{tabular}{lcccccc}
\toprule
Subset & Seeds & K1 & LTAG & Median & $\Delta$(K1$-$LTAG) & LTAG wins \\
\midrule
''' + body + rf''' \\
\midrule
Mean & -- & -- & -- & -- & +{mean_d:.3f} & 15/15 \\
\bottomrule
\end{{tabular}}
\end{{table*}}
'''
write('tab1_test.tex', s)

# Table 2: validation
arms = [('K1', 'K1'), ('Median', 'Median'), ('LTAG', 'LTAG'),
        ('Global-g', 'Global-g'), ('K1-Calib', 'K1-Calib'),
        ('Linear-Stack', 'Linear-Stack'), ('LTAG-noMW', 'LTAG-noMW')]
tab = {}
for s in SUBSETS:
    d = json.loads(Path(f'results/ltagv2_p1_{s}.json').read_text())
    tab[s] = {r['arm']: r['engine_mean_rmse'] for r in d['table']}
lines = []
for name, _ in arms:
    vals = [tab[s].get(name, float('nan')) for s in SUBSETS]
    lines.append(f'{name} & ' + ' & '.join(f'{v:.3f}' for v in vals) + r' \\')
s = r'''\begin{table}[t]
\centering
\caption{Validation engine-mean RMSE (observable features; 5 cuts; seeds
101/202/303, FD001 adds 404/505/606). Lower is better.}
\label{tab:validation}
\begin{tabular}{lcccc}
\toprule
Arm & FD001 & FD002 & FD003 & FD004 \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
write('tab2_validation.tex', s)

# Table 3: cut-wise
cw = json.loads(Path('results/cutwise.json').read_text())
lines = []
for s in SUBSETS:
    for r in cw[s]:
        lines.append(f"{s} & {r['frac']:.2f} & {r['k1']:.3f} & {r['ltag']:.3f} & "
                     f"{r['calib']:.3f} & {r['ls']:.3f} & {r['ls'] - r['ltag']:+.3f} \\\\")
s = r'''\begin{table*}[t]
\centering
\caption{Cut-wise validation engine-mean RMSE. LS-ctx is stronger at early cuts;
LTAG is competitive or better at 0.85--0.95. The last column is LS$-$LTAG
(positive = LTAG better).}
\label{tab:cutwise}
\small
\begin{tabular}{llcccccc}
\toprule
Subset & Cut & K1 & LTAG & K1-Calib & LS-ctx & LS$-$LTAG \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
write('tab3_cutwise.tex', s)

# Table 4: Phase 4
p4 = json.loads(Path('results/phase4.json').read_text())
lines = []
for key in sorted(p4):
    subset, fault, onset = key.split('|')
    v = p4[key]
    r3, rb, rm = v.get('R3', float('nan')), v.get('R5b', float('nan')), v.get('R5m', float('nan'))
    mx = max(rb, rm)
    lines.append(f'{subset} & {fault} & {onset} & {r3:.3f} & {rb:.3f} & {rm:.3f} & {r3-mx:+.3f} \\\\')
s = r'''\begin{table*}[t]
\centering
\caption{Fair naive baselines for RHSE-TSH (3-seed mean recovery). R5b: backtracked
hold; R5m: backtracked median. R3 only passes the $\geq$0.10 gate on stuck.}
\label{tab:rhse}
\small
\begin{tabular}{llcccccc}
\toprule
Subset & Fault & Onset & R3 & R5b & R5m & R3$-$max \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
write('tab4_rhse.tex', s)

# Table 5: gate summary
s = r'''\begin{table}[t]
\centering
\caption{Gate summary. Gate 1 fails because simple calibration beats LTAG on
validation; RHSE-TSH is removed after fair temporal baselines match it on
bias/drift.}
\label{tab:gates}
\begin{tabular}{ll}
\toprule
Item & Result \\
\midrule
Gate 1 (LTAG vs controls) & FAIL \\
\quad G1.1 vs K1-Calib & 0/4, mean $-0.393$ \\
\quad G1.2 vs Linear-Stack & 0/4 \\
\quad G1.3 Ens3-LTAG vs Ens3-K1 & PASS (4/4) \\
\quad G1.4 noMW worse & 2/4 (FAIL) \\
Gate 2 (cross-backbone) & not started \\
RHSE-TSH & removed \\
Test & sealed; not rerun \\
\bottomrule
\end{tabular}
\end{table}
'''
write('tab5_gates.tex', s)
