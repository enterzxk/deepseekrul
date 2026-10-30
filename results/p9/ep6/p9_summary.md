## FD002 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 19.336 +/- 1.145 | 1566.5 +/- 221.4 | 14.872 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 14.283 +/- 0.236 | 1013.3 +/- 96.4 | 10.021 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 12.978 +/- 0.121 | 916.0 +/- 69.6 | 8.789 | 0.000 | 144,146 | 5 |
| Ours w/o CR-GCN | 13.224 +/- 0.194 | 852.1 +/- 66.9 | 9.124 | 0.000 | 135,735 | 5 |
| Ours w/o Synergy+Mono | 13.445 +/- 0.132 | 1031.3 +/- 58.7 | 9.056 | 0.494 | 143,415 | 5 |
| Ours Full (CR-GraphTransformer) | 13.490 +/- 0.414 | 931.1 +/- 73.7 | 9.278 | 0.000 | 143,415 | 5 |
| Baseline TCN (5-seed ensemble) | 18.529 +/- 0.000 | 1306.4 +/- 0.0 | 14.459 | 0.000 | 0 | 5 |
| Baseline BiLSTM (5-seed ensemble) | 13.417 +/- 0.000 | 806.2 +/- 0.0 | 9.432 | 0.000 | 0 | 5 |
| Baseline Transformer (5-seed ensemble) | 12.294 +/- 0.000 | 794.6 +/- 0.0 | 8.332 | 0.000 | 0 | 5 |
| Ours w/o CR-GCN (5-seed ensemble) | 12.799 +/- 0.000 | 762.9 +/- 0.0 | 8.880 | 0.000 | 0 | 5 |
| Ours w/o Synergy+Mono (5-seed ensemble) | 12.925 +/- 0.000 | 939.8 +/- 0.0 | 8.725 | 0.000 | 0 | 5 |
| Ours Full (CR-GraphTransformer) (5-seed ensemble) | 12.904 +/- 0.000 | 820.4 +/- 0.0 | 8.923 | 0.000 | 0 | 5 |

### Paired deltas vs Ours Full (FD002, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +5.846 | [+4.901, +6.895] | 0/5 | 0.0625 |
| Baseline BiLSTM | +0.793 | [+0.495, +1.040] | 0/5 | 0.0625 |
| Baseline Transformer | -0.512 | [-0.855, -0.123] | 4/5 | 0.3750 |
| Ours w/o CR-GCN | -0.266 | [-0.525, -0.018] | 4/5 | 0.3750 |
| Ours w/o Synergy+Mono | -0.045 | [-0.309, +0.266] | 3/5 | 1.0000 |
