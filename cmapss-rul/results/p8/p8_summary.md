## FD001 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 18.103 +/- 0.491 | 621.8 +/- 127.0 | 14.068 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 14.266 +/- 0.565 | 331.9 +/- 28.8 | 10.289 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 13.548 +/- 0.234 | 357.9 +/- 30.0 | 9.775 | 0.000 | 144,146 | 5 |
| Ours w/o Graph | 13.809 +/- 0.327 | 326.6 +/- 13.5 | 9.733 | 0.000 | 141,318 | 5 |
| Ours w/o Physics+Mono | 13.793 +/- 0.375 | 323.5 +/- 13.6 | 9.818 | 0.501 | 141,766 | 5 |
| Ours Full (PA-GraphTransformer) | 13.593 +/- 0.365 | 315.0 +/- 10.8 | 9.766 | 0.000 | 141,766 | 5 |

### Paired deltas vs Ours Full (FD001, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +4.511 | [+3.883, +5.007] | 0/5 | 0.0625 |
| Baseline BiLSTM | +0.673 | [+0.396, +0.924] | 0/5 | 0.0625 |
| Baseline Transformer | -0.045 | [-0.522, +0.432] | 2/5 | 1.0000 |
| Ours w/o Graph | +0.216 | [-0.166, +0.647] | 2/5 | 1.0000 |
| Ours w/o Physics+Mono | +0.200 | [+0.009, +0.381] | 1/5 | 0.3750 |

## FD002 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 18.870 +/- 0.647 | 1716.4 +/- 178.1 | 14.459 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 13.554 +/- 0.220 | 857.2 +/- 76.6 | 9.992 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 12.833 +/- 0.355 | 781.8 +/- 68.6 | 9.244 | 0.000 | 144,146 | 5 |
| Ours w/o Graph | 13.680 +/- 0.534 | 942.0 +/- 126.9 | 9.817 | 0.000 | 141,318 | 5 |
| Ours w/o Physics+Mono | 14.084 +/- 0.843 | 1019.9 +/- 244.0 | 10.137 | 0.496 | 141,766 | 5 |
| Ours Full (PA-GraphTransformer) | 14.454 +/- 0.370 | 1056.0 +/- 193.7 | 10.423 | 0.000 | 141,766 | 5 |

### Paired deltas vs Ours Full (FD002, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +4.416 | [+3.583, +5.267] | 0/5 | 0.0625 |
| Baseline BiLSTM | -0.900 | [-1.399, -0.501] | 5/5 | 0.0625 |
| Baseline Transformer | -1.621 | [-2.021, -1.131] | 5/5 | 0.0625 |
| Ours w/o Graph | -0.774 | [-1.157, -0.396] | 5/5 | 0.0625 |
| Ours w/o Physics+Mono | -0.370 | [-1.335, +0.545] | 3/5 | 1.0000 |
