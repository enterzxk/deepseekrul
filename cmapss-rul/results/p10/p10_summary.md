## FD001 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 16.646 +/- 0.413 | 466.0 +/- 47.6 | 12.998 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 14.158 +/- 0.290 | 328.2 +/- 9.5 | 10.172 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 13.289 +/- 0.132 | 390.1 +/- 21.2 | 9.320 | 0.000 | 144,146 | 5 |
| Ours w/o Sensor-Graph | 14.390 +/- 1.024 | 391.0 +/- 60.3 | 10.050 | 0.000 | 141,093 | 5 |
| Ours w/o Adaptive Tail+Mono | 13.713 +/- 0.150 | 362.9 +/- 17.6 | 9.622 | 0.496 | 141,707 | 5 |
| Ours Full (Native Sensor-Graph Transformer) | 13.753 +/- 0.191 | 339.5 +/- 24.4 | 9.615 | 0.000 | 141,707 | 5 |
| Baseline TCN (5-seed ensemble) | 16.165 +/- 0.000 | 425.3 +/- 0.0 | 12.766 | 0.000 | 0 | 5 |
| Baseline BiLSTM (5-seed ensemble) | 13.985 +/- 0.000 | 318.6 +/- 0.0 | 9.976 | 0.000 | 0 | 5 |
| Baseline Transformer (5-seed ensemble) | 13.058 +/- 0.000 | 376.3 +/- 0.0 | 9.166 | 0.000 | 0 | 5 |
| Ours w/o Sensor-Graph (5-seed ensemble) | 13.808 +/- 0.000 | 350.0 +/- 0.0 | 9.667 | 0.000 | 0 | 5 |
| Ours w/o Adaptive Tail+Mono (5-seed ensemble) | 13.455 +/- 0.000 | 343.6 +/- 0.0 | 9.442 | 0.000 | 0 | 5 |
| Ours Full (Native Sensor-Graph Transformer) (5-seed ensemble) | 13.243 +/- 0.000 | 300.5 +/- 0.0 | 9.182 | 0.000 | 0 | 5 |

### Paired deltas vs Ours Full (FD001, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +2.893 | [+2.482, +3.276] | 0/5 | 0.0625 |
| Baseline BiLSTM | +0.405 | [+0.090, +0.724] | 0/5 | 0.0625 |
| Baseline Transformer | -0.463 | [-0.669, -0.295] | 5/5 | 0.0625 |
| Ours w/o Sensor-Graph | +0.637 | [-0.132, +1.720] | 2/5 | 1.0000 |
| Ours w/o Adaptive Tail+Mono | -0.040 | [-0.112, +0.032] | 3/5 | 1.0000 |

## FD002 5-seed test results (clip125)

| Variant | RMSE | NASA | MAE | Mono | Params | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline TCN | 19.958 +/- 1.542 | 1653.1 +/- 260.0 | 15.271 | 0.000 | 145,237 | 5 |
| Baseline BiLSTM | 14.027 +/- 0.112 | 901.8 +/- 72.7 | 10.038 | 0.000 | 144,701 | 5 |
| Baseline Transformer | 12.707 +/- 0.178 | 822.7 +/- 65.9 | 8.661 | 0.000 | 144,146 | 5 |
| Ours w/o Sensor-Graph | 13.966 +/- 0.367 | 1132.2 +/- 111.4 | 9.351 | 0.000 | 141,093 | 5 |
| Ours w/o Adaptive Tail+Mono | 13.840 +/- 0.497 | 1132.8 +/- 115.8 | 9.035 | 0.490 | 141,707 | 5 |
| Ours Full (Native Sensor-Graph Transformer) | 13.709 +/- 0.437 | 1044.4 +/- 140.7 | 9.094 | 0.000 | 141,707 | 5 |
| Baseline TCN (5-seed ensemble) | 19.025 +/- 0.000 | 1344.1 +/- 0.0 | 14.801 | 0.000 | 0 | 5 |
| Baseline BiLSTM (5-seed ensemble) | 13.223 +/- 0.000 | 739.1 +/- 0.0 | 9.375 | 0.000 | 0 | 5 |
| Baseline Transformer (5-seed ensemble) | 12.194 +/- 0.000 | 734.9 +/- 0.0 | 8.285 | 0.000 | 0 | 5 |
| Ours w/o Sensor-Graph (5-seed ensemble) | 13.290 +/- 0.000 | 976.5 +/- 0.0 | 8.965 | 0.000 | 0 | 5 |
| Ours w/o Adaptive Tail+Mono (5-seed ensemble) | 13.118 +/- 0.000 | 960.2 +/- 0.0 | 8.559 | 0.000 | 0 | 5 |
| Ours Full (Native Sensor-Graph Transformer) (5-seed ensemble) | 13.025 +/- 0.000 | 875.3 +/- 0.0 | 8.701 | 0.000 | 0 | 5 |

### Paired deltas vs Ours Full (FD002, RMSE; negative = better)

| Comparison | Delta | 95% CI | Wins | sign-test p |
|---|---:|---|---:|---:|
| Baseline TCN | +6.249 | [+5.182, +7.718] | 0/5 | 0.0625 |
| Baseline BiLSTM | +0.318 | [+0.013, +0.790] | 1/5 | 0.3750 |
| Baseline Transformer | -1.002 | [-1.287, -0.654] | 5/5 | 0.0625 |
| Ours w/o Sensor-Graph | +0.257 | [+0.052, +0.448] | 1/5 | 0.3750 |
| Ours w/o Adaptive Tail+Mono | +0.131 | [-0.243, +0.401] | 1/5 | 0.3750 |
