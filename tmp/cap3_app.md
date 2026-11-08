
**Note on the ablation table.** Deltas in \`ABLATIONS.csv\` are computed against the plain
backbone M0 on the folds where both the variant and M0 exist, so they are not directly
comparable with the deltas in \`RESULTS.csv\`, which are computed against the locked
reference arm (FULL). For NORM on FD004 the ablation delta against M0 is +1.390
[+0.932, +1.870] while the delta against FULL is +1.535 [+0.989, +2.116]; both are correct
for their respective baselines.
