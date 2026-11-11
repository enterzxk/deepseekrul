# Paper B protocol audit (2026-11-11)

Every claim below was checked against the repository, the lab3090 working tree,
or an executed command. Numbers that could not be traced to a checkpoint and a
prediction file are marked untraceable.

## 0. Headline

The historical Paper B evidence is **not a single protocol**. Three different
evaluators, two window lengths, two feature sets and at least two uncommitted
code states were mixed. The 2026-11-11 `95e729e` commit claims to fix P0 bugs
but changed only a Markdown file; the code fixes lived on one server and were
never committed. The new `paper_b/` package is the first state in which code,
data split, model selection and test evaluation are one auditable pipeline.

## 1. Defects found

| id | defect | evidence | severity | resolution |
|---|---|---|---|---|
| P0-1 | final evaluator `ev()` ignored `--no-cycle` | repo `run_dspt.py` passes `cycle=` unconditionally; `git show --stat 95e729e` = 1 file (P0_FIXES.md) | fatal for every no-cycle result | evaluation honours `net.use_cycle`; no-cycle results invalidated |
| P0-2 | MS convolution mixed zero padding into valid frames | repo `dspt.py` `encode()`; server copy has a partial fix | moderate | replicate-pad **both** edges, zero output on pad positions |
| P0-3 | HI monotonic penalty applied to padding positions | `torch.mean(relu(hseq[:,1:]-hseq[:,:-1]))` over all L | low | adjacent-valid mask |
| P0-3b | HI monotonic **direction** | probe: per-engine Spearman(HI, cycle) = **-0.92 / -0.88 / -0.88 / -0.79** (FD001/2/3/4). HI is a decreasing *health* index, so the legacy `relu(h_t - h_{t-1})` direction was correct. | clarification | sign kept configurable; main grid uses the correct (decreasing) constraint |
| P0-4 | `windows_for` signature mismatch | repo: body uses `legacy=legacy_rul`, signature has no such argument -> NameError | fatal (repo copy cannot run) | new `training/data.py` |
| P0-5 | HI branch missing `else` | server snapshot | low | not applicable in new code |
| P0-6 | EMA checkpoint selection | logic was correct (best-state vs EMA compared on validation) | — | removed: a second validation-selection step adds optimism; one predetermined rule (best validation checkpoint) |
| P1-1 | causal EMA initialised at `x[:,0]` | for a left-padded window `x[:,0]=0`, so the first valid EMA is halved | moderate for short test/validation windows | initialise at the first valid frame |
| P1-2 | early stopping on window-level validation RMSE | `evaluate(net, VA)` over **all** windows of the 20% validation engines | high: optimistically biased, dominated by long engines | engine-level truncated validation, 5 cuts (0.40/0.55/0.70/0.85/0.95) |
| P1-3 | OOF truncation points sampled from the official test RUL distribution | `RESULTS_SUMMARY.md` §0 states this explicitly | fatal for OOF | OOF is INVALID, not used |
| P1-4 | test evaluator cannot load the checkpoints it is said to evaluate | `eval_dspt.py` builds `max_span=100`; `ABL_FD001_base/FD001/dspt_s101.pt` has `temporal_pe [1,70,64]`; load raises `RuntimeError: size mismatch` | fatal for provenance | unified model/data/evaluator; every test number regenerated |
| P1-5 | official test observed many times | `RESULTS_SUMMARY.md` §4; `VANILLA_ABLATION_REPORT.md` §10 | high | freeze manifest; test opened once per frozen run |
| P1-6 | reported parameter counts included dead modules | old base 115,716 params although SA/HI were off; new conditional base 77,058 | reporting | modules built conditionally |
| P1-7 | commit message and content disagree | `git show --stat 95e729e` | high | archived; server snapshot saved in `archive/legacy_2026_11/` |
| P1-8 | HI / scaler fitted on all engines | fixed in PHASE1 (train-only) | — | verified in new `fit_hi` / `OpScaler` |
| P1-9 | HI target is a noisy re-parameterisation of RUL | probe: R^2(HI~RUL) = 0.61-0.78; ~48% of adjacent steps move opposite to the global trend | mechanism | analysed, not a bug |

## 2. Verification commands and observed output

```text
$ git show --stat --oneline 95e729e
95e729e P0 reproducibility fixes: ...
 docs/paper_b_code/P0_FIXES.md | 105 ++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 105 insertions(+)

$ python scripts/eval_dspt.py --subset FD001 --fold 0 --seed 101 \
    --ckpt experiments/ABL_FD001_base/FD001/dspt_s101.pt --out /tmp/x.csv
RuntimeError: Error(s) in loading state_dict for DSPT:
  size mismatch for temporal_pe: copying a param with shape torch.Size([1, 70, 64])
  from checkpoint, the shape in current model is torch.Size([1, 100, 64]).

$ ls -l experiments/ABLTEST/FD001_base.csv experiments/ABL_FD001_base/FD001/dspt_s101.pt
-rw-rw-r-- ... 00:39 ... dspt_s101.pt
-rw-rw-r-- ... 12:20 ... FD001_base.csv
```

## 3. Frozen protocol used by `paper_b/`

| item | value |
|---|---|
| split | official training engines, 80/20 **by engine id**, `split_seed=7`, `fold=0` |
| model seeds | 101 / 202 / 303 |
| validation | 5 engine-level truncations at 40/55/70/85/95% of each engine's life; true RUL = n - cut |
| model selection | lowest engine-level validation RMSE; single best checkpoint, no EMA |
| test | last observed cycle per official test engine; RUL from `RUL_FDxxx.txt`, capped 125 |
| test access | only `evaluation/evaluate_test.py`, only for tags listed in `results/FREEZE.json` |
| window | L=70 for the structure comparison; L search only after the structure is frozen |
| loss | MSE + 0.5 x asymmetric MSE (late 1.8x); DSPT dual-span auxiliary 0.25 x (e_A^2 + e_B^2) |
| preprocessing | operating-condition KMeans (6) + per-cluster z-score, fitted on training engines only |
| HI | PCA-1 health index, fitted on training engines only, decreasing with life |

## 4. What this audit does **not** establish

- It does not prove the old numbers are numerically wrong; it proves they are
  **untraceable** (no evaluator/checkpoint pair that reproduces them) and
  **protocol-confounded** (different windows/features/validation).
- It does not remove the fact that the official test sets have been observed.
  The freeze manifest limits further selection, but the four C-MAPSS test sets
  are not pristine. This must be stated in any paper.
