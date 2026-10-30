## FD001 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 16.923 +/- 0.380 | 497.9 +/- 23.0 | 13.289 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 14.141 +/- 0.600 | 345.4 +/- 31.7 | 10.117 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 13.525 +/- 0.272 | 419.2 +/- 66.5 | 9.381 | 0.000 | 144,146 | 5 |
| Ours w/o CR-GCN | 13.393 +/- 0.160 | 330.8 +/- 13.8 | 9.419 | 0.000 | 135,735 | 5 |
| Ours w/o Synergy+Mono | 13.734 +/- 0.378 | 374.9 +/- 22.2 | 9.965 | 0.533 | 143,415 | 5 |
| Ours Full (CR-GraphTransformer) | 13.209 +/- 0.318 | 317.7 +/- 34.6 | 9.522 | 0.000 | 143,415 | 5 |
| Baseline TCN (5-seed ensemble) | 16.502 +/- 0.000 | 457.8 +/- 0.0 | 13.050 | 0.000 | 0 | 5 |
| Baseline BiLSTM (5-seed ensemble) | 13.885 +/- 0.000 | 329.8 +/- 0.0 | 9.845 | 0.000 | 0 | 5 |
| Baseline Transformer (5-seed ensemble) | 13.137 +/- 0.000 | 381.0 +/- 0.0 | 9.096 | 0.000 | 0 | 5 |
| Ours w/o CR-GCN (5-seed ensemble) | 13.201 +/- 0.000 | 322.5 +/- 0.0 | 9.243 | 0.000 | 0 | 5 |
| Ours w/o Synergy+Mono (5-seed ensemble) | 13.394 +/- 0.000 | 353.8 +/- 0.0 | 9.687 | 0.000 | 0 | 5 |
| Ours Full (CR-GraphTransformer) (5-seed ensemble) | 12.868 +/- 0.000 | 296.3 +/- 0.0 | 9.244 | 0.000 | 0 | 5 |

### Paired deltas vs Ours Full (FD001, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +3.714 | [+3.177, +4.251] | 0/5 | 0.0625 |
| Baseline BiLSTM | +0.931 | [+0.399, +1.353] | 1/5 | 0.3750 |
| Baseline Transformer | +0.315 | [-0.022, +0.642] | 1/5 | 0.3750 |
| Ours w/o CR-GCN | +0.184 | [-0.160, +0.542] | 2/5 | 1.0000 |
| Ours w/o Synergy+Mono | +0.525 | [+0.120, +1.090] | 1/5 | 0.3750 |

## FD002 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 18.350 +/- 0.626 | 1587.3 +/- 70.2 | 13.924 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 17.601 +/- 0.418 | 2067.5 +/- 331.4 | 12.095 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 15.150 +/- 0.438 | 1608.6 +/- 259.5 | 10.341 | 0.000 | 144,146 | 5 |
| Ours w/o CR-GCN | 15.683 +/- 0.703 | 1888.4 +/- 364.2 | 10.218 | 0.000 | 135,735 | 5 |
| Ours w/o Synergy+Mono | 15.807 +/- 0.761 | 2185.8 +/- 390.9 | 10.527 | 0.517 | 143,415 | 5 |
| Ours Full (CR-GraphTransformer) | 15.577 +/- 0.508 | 1771.2 +/- 367.9 | 10.080 | 0.000 | 143,415 | 5 |
| Baseline TCN (5-seed ensemble) | 17.822 +/- 0.000 | 1409.2 +/- 0.0 | 13.654 | 0.000 | 0 | 5 |
| Baseline BiLSTM (5-seed ensemble) | 15.578 +/- 0.000 | 1335.2 +/- 0.0 | 10.896 | 0.000 | 0 | 5 |
| Baseline Transformer (5-seed ensemble) | 13.251 +/- 0.000 | 978.4 +/- 0.0 | 8.834 | 0.000 | 0 | 5 |
| Ours w/o CR-GCN (5-seed ensemble) | 13.902 +/- 0.000 | 1309.4 +/- 0.0 | 9.209 | 0.000 | 0 | 5 |
| Ours w/o Synergy+Mono (5-seed ensemble) | 13.484 +/- 0.000 | 1181.6 +/- 0.0 | 9.095 | 0.000 | 0 | 5 |
| Ours Full (CR-GraphTransformer) (5-seed ensemble) | 13.412 +/- 0.000 | 1094.4 +/- 0.0 | 8.861 | 0.000 | 0 | 5 |

### Paired deltas vs Ours Full (FD002, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +2.773 | [+2.272, +3.260] | 0/5 | 0.0625 |
| Baseline BiLSTM | +2.024 | [+1.371, +2.747] | 0/5 | 0.0625 |
| Baseline Transformer | -0.427 | [-1.052, +0.195] | 3/5 | 1.0000 |
| Ours w/o CR-GCN | +0.107 | [-0.437, +0.626] | 2/5 | 1.0000 |
| Ours w/o Synergy+Mono | +0.231 | [-0.245, +0.706] | 2/5 | 1.0000 |
