# Step 1 Validation Record

Date: 2026-10-20

## Scope

Standard C-MAPSS preprocessing and PyTorch Dataset construction for FD001--FD004.

## Environment

- Python 3.9.25
- NumPy 1.26.4
- pandas 2.0.3
- PyTorch 2.5.1+cu121
- NVIDIA A40 available

## Automated checks

Command:

```bash
cd /home/yndx/zxk/bearing-rul-pi
python -m unittest discover -s tests -v
```

Result: **12 tests passed**.

Verified properties:

- 26-column C-MAPSS schema parsing
- Correct standard 14-sensor selection
- Training-only Min-Max fit and test transform
- Piecewise RUL label construction and window-end alignment
- Training window length 30 and stride 1
- One final test window per engine
- Left-edge first-frame padding for short test histories
- Test ground-truth values match `RUL_FDxxx.txt`
- No NaN or infinite values
- RMSE and NASA asymmetric score formulas
- DataLoader target shape `(B, 1)` with input `(B, 30, 14)`

## Real-data summary

| Subset | Train windows | Test windows | Train engines | Test engines | Test RUL | Test lengths | Short tests |
|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 17731 | 100 | 100 | 100 | 7--145 | 31--303 | 0 |
| FD002 | 46219 | 259 | 260 | 259 | 6--194 | 21--367 | 6 |
| FD003 | 21820 | 100 | 100 | 100 | 6--145 | 38--475 | 0 |
| FD004 | 54028 | 248 | 249 | 248 | 6--195 | 19--486 | 11 |

## Artifact hashes

```text
ba32aa502fd9be5761206d47982368aac6c9f356f9f1781a1b47557ac381383e  src/dataset.py
d2699cffbdef0ff2efc43b845f48cb9c80781598c556c337f290f8075d1bd3b8  tests/test_dataset.py
```

## Notes

- The packaged readme reports FD004 train/test counts as 248/249, while the delivered files contain 249 training engines and 248 test engines/RUL labels. The code validates the actual files and passes.
- Official test data were not used for model selection or hyperparameter tuning.
- No model training or performance claims have been made at this milestone.