| subset | variant | folds | note | rmse_variant | rmse_M0 | delta | ci_lo | ci_hi | engines |
|---|---|---|---|---|---|---|---|---|---|
| FD002 | DIFF | 2 | window minus individual early reference | 13.82 | 13.85 | -0.038 | -0.534 | 0.472 | 104 |
| FD002 | PHASE | 2 | RUL = phi_hat * mu_T | 17.76 | 13.85 | 3.909 | 2.383 | 5.421 | 104 |
| FD002 | M0_eqbal | 1 | engine-balanced sample weights | 13.88 | 13.97 | -0.092 | -0.873 | 0.704 | 52 |
| FD002 | NORM | 5 | cycle coordinate replaced by c/mu_T | 14.59 | 13.51 | 1.085 | 0.7 | 1.473 | 260 |
| FD004 | NORM | 5 | cycle coordinate replaced by c/mu_T | 14.79 | 13.4 | 1.39 | 0.932 | 1.87 | 249 |