# DSPT-Base functional audit

**Scope.** What `DSPT-Base` actually computes, whether each advertised function is
active, whether the pipeline leaks test information, and how it compares with an
independently defined Vanilla Transformer. This document does **not** decide the
novelty question (§6.5 is marked PENDING-L2).

**Repository / commit.** `enterzxk/deepseekrul`, branch `punet-interval`, baseline
commit `4bfba0b`. Audit written 2026-11-12.

**Evidence classes used below.**

| class | meaning |
|---|---|
| `code` | read from the file at the cited line |
| `runtime` | measured by `paper_b/experiments/audit_dspt_base.py` on a synthetic batch |
| `real-val` | measured on the frozen validation split with a saved/retrained checkpoint |
| `grid` | from the 96-cell A8 grid (`paper_b/results/a8_engine_level.csv`) |

Runtime evidence is structural: it proves a module is wired and active, not that it
helps. Performance claims come only from `real-val` and `grid`.

---

## 0. Bottom line

1. **DSPT-Base is not "DSPT with everything off".** It is a specific five-stage
   pipeline: 3-channel input → one depthwise k=3 conv → per-sensor 2-layer
   Transformer → learned per-timestep sensor pooling → fixed dual-span mix.
   77,058 parameters, of which 66,944 (86.9%) are the shared temporal Transformer.
2. **Three advertised functions are weaker than their names suggest.**
   The dual-span gate is not learned and is a fixed function of window length
   (`g=0.4` for every full 70-frame window); the two spans' embeddings differ by
   0.80%, and on real validation `dual` is 0.073 RMSE *worse* than `long` alone.
   The cycle encoding is added after the temporal Transformer and after sensor
   pooling, is constant over time, and therefore cannot modulate attention.
   The single-scale conv is followed by global self-attention, so it is a local
   feature mixer, not the temporal receptive field.
3. **The dominant feature is the absolute cycle count, not the architecture.**
   Shuffling `cycle` across validation engines (same marginal distribution) raises
   engine-level RMSE from 11.379 to 16.285 (+4.91). Zeroing it raises it to 22.736.
   In the Vanilla 2×2, adding the cycle encoding alone moves mean RMSE from 12.803
   to 11.758, while adding the 3-channel representation alone does not help
   (12.897) and has no detectable effect of its own (§5.4).
4. **No test-set leakage was found.** The training path cannot open the official
   test files (verified by monkeypatching `load_test` to raise); the operating-condition
   scaler is fitted on training engines only and its means reproduce the training-frame
   cluster means exactly; train/validation engines are disjoint; the baseline reference
   is causal. What *is* wrong is the **label definition**: `data.py:245` caps the test
   target at 125, which is a protocol defect, not contamination.
5. **No module has a stable independent gain.** Over the 96-cell grid, `sa` is the only
   switch with a consistent sign on FD001–FD003 (negative ≈ −0.2…−0.5, comparable to the
   seed spread); `ms` is mixed; `hi` is stably ≈ 0. `sa` on FD004 flips sign between the
   capped and raw scoring conventions, so it cannot be written as a mechanism.

---

## 1. What "DSPT-Base" is (and is not)

`DSPT-Base` is the all-off cell of the A8 grid, tag `A8_base_L70`:

```json
{"model":"dspt","span":70,"d_model":64,"layers":2,"dropout":0.2,
 "ms_mode":"off","sa_mode":"off","hi_source":"none","instnorm":false,
 "use_cycle":true,"use_pad_mask":true,"mode":"dual"}
```

The runtime build from `paper_b/training/train.py:_build` with this config has
**77,058 parameters**, which matches `metrics.json:params` of every `A8_base_L70`
cell exactly. That equality is the link between the code audited here and the
numbers reported in the A8 grid.

It is **not**:

- *"DSPT"* as the full model: `A8_full_L70` (ms+sa+hi) has 115,651 parameters.
- *"a Vanilla Transformer"*: it has a conv, learned sensor pooling, a learned
  distance-from-end positional table, and a cycle encoding; Vanilla has none of these.
- *"multi-scale"*: `ms_mode="off"` builds exactly one `Conv1d`.

---

## 2. Module-by-module audit

### 2.1 Operating-condition normalisation — ACTIVE (preprocessing, shared by all arms)

| | |
|---|---|
| code | `paper_b/training/data.py:47-86` (`OpScaler`), `:264-266` (fit on train, transform train/val), `:319` (test reuses the train scaler) |
| what it does | KMeans (`k=6`, `random_state=split_seed`) on `op_1..op_3`; each sensor is z-scored **within its cluster** |
| runtime | `audit_preprocessing.py`: `scaler.means` equals the training-frame cluster means to **0.0** absolute error on all four subsets; transformed training rows have per-cluster `max|mean| ≤ 1.8e-6` and `max|std−1| ≤ 2.4e-5` |
| leakage | train/validation engines disjoint on all subsets (80/20, 208/52, 80/20, 199/50); the scaler is never fitted on validation or test rows |
| verdict | **active**, and correctly fitted on training engines only. It is not a DSPT module — every arm, including Vanilla, receives the same normalised sensors. |

### 2.2 Three-channel input representation — ACTIVE

| | |
|---|---|
| code | `paper_b/models/dspt.py:152-174` (`_features`), `:97` (`sensor_in`), `:95` (`n_feat=3` when `instnorm=False`) |
| what it does | per sensor and step: `[x, x − b, e]` where `b` is the shrinkage baseline reference and `e` is a causal EMA with `α=0.5` |
| runtime | `sensor_in.in_features = 3`; channel 0 `== x`; channel 1 `== x − baseline`; channel 2 satisfies the EMA recursion on every valid step and is frozen on padding; `max|ch2−ch0| = 2.35` |
| P2 fix verified | EMA is initialised at the **first valid frame**, not `x[:,0]`: on a 22-frame window the pre-valid EMA equals `x[first_valid]` exactly |
| verdict | **active**. Note `instnorm=False`, so the optional 4th channel is *not* built (`n_feat=3`). |

### 2.3 Cycle encoding — ACTIVE, but structurally post-hoc

| | |
|---|---|
| code | `paper_b/models/dspt.py:239-243`; input `cycle = min(cycle/250, 1)` from `data.py:186` |
| what it does | Fourier features of the absolute cycle at periods (25, 50, 125, 250), `Linear(8→64)`, added to the pooled tensor as a `(B,1,D)` broadcast |
| runtime | `pe_proj.in_features = 8`; zeroing `cycle` moves the prediction by 0.39 on the synthetic batch |
| placement | added **after** the per-sensor Transformer and **after** sensor pooling, then passed through `LayerNorm`. It is identical at every time step, so it carries **zero temporal information** and cannot modulate attention. Its effect is a per-engine additive bias before the readout. |
| real-val | shuffling `cycle` across validation engines: 11.379 → **16.285** (+4.91); zeroing: → 22.736 (+11.36); mean absolute prediction shift under zeroing = 19.0 cycles |
| verdict | **active and highly influential**, but as a global engine-age feature rather than as a positional encoding. This is the single most important finding for "why is Base strong" (§6.1) and for comparability with papers that do not use absolute cycle (§6.3). |

### 2.4 Single-scale convolution — ACTIVE (one depthwise k=3 conv)

| | |
|---|---|
| code | `paper_b/models/dspt.py:101-102` (module), `:176-190` (`_ms`), `:202-210` (padding fill + mask) |
| what it does | `ms_mode="off"` → `GELU(conv3(h))`, depthwise (`groups=d_model`), kernel 3, `padding_mode="replicate"`; **no** `ms_proj` |
| runtime | exactly **1** `Conv1d`: kernel 3, groups 64, replicate padding; `has_ms_proj=False`; conv output differs from input (`std=0.51`) |
| P1 fix verified | left padding is filled with the first valid frame and the conv output is masked afterwards (`:202-208`) |
| verdict | **active**. Caveat: the conv is followed by a global self-attention stack, so the model's receptive field is the whole window; the conv is a local feature mixer, not the temporal modelling mechanism. This is why the multi-scale bank has little room to add (§6.4). |

### 2.5 Per-sensor temporal Transformer — ACTIVE (86.9% of parameters)

| | |
|---|---|
| code | `paper_b/models/dspt.py:212-218` (tokenisation, padding mask), `:125-129` (2 × `TransformerEncoderLayer`, 4 heads, pre-norm, GELU) |
| what it does | tokens are `(B·N, L, D)` — each sensor's trajectory is a sequence; attention runs **within** a sensor, never across sensors when `sa_mode="off"` |
| runtime | forward hook on layer 0: input `(56, 70, 64)` = `B·N × L × D`; 2 layers, 4 heads |
| parameters | 66,944 of 77,058 (86.9%) |
| verdict | **active**; this is the bulk of the model. |

### 2.6 Learned sensor pooling — ACTIVE and genuinely learned

| | |
|---|---|
| code | `paper_b/models/dspt.py:141-142` (`sensor_score` exists for `sa_mode in {off, basic, identity, cond, identity_cond}`), `:237-238` (softmax over sensors, per time step) |
| what it does | `score = softmax_N(W·token)`, `pooled_t = Σ_n score_{t,n} · token_{t,n}`; the weights are **per (sample, time step)**, not a static per-sensor vector |
| runtime | `sensor_score.weight` shape `(1,64)`; gradient norm 0.93 after one backward (learned, not frozen); scores sum to 1; `score_is_time_varying=True`; mean max weight 0.154 vs 0.071 uniform |
| verdict | **active and learned**. It is the only cross-sensor mixing present in Base (`sa_mode="off"` builds no `sensor_attn`, no `cond_emb`). |

### 2.7 Dual-span prediction — STRUCTURALLY PRESENT, FUNCTIONALLY NEUTRAL

| | |
|---|---|
| code | `paper_b/models/dspt.py:246-271` |
| what it does | `h_A` = masked mean over the last 50 valid frames, `h_B` = masked mean over all valid frames, `g = clamp((m−50)/50, 0, 1)`, `h_f = g·h_B + (1−g)·h_A`; prediction is `reg_head(h_f)` |
| runtime | `gate = [0.4, 0.4, 0.4, 0.0]` for window lengths (70, 70, 70, 22) → `g` is a deterministic function of valid length; **no gate parameters exist**; `h_A` vs `h_B` relative difference **0.80%**; `max|y_A − y_B| = 0.0024` |
| real-val (FD001 s101) | engine-level RMSE: `dual` 11.379, `long` 11.306, `short` 11.612 → **dual is 0.073 worse than long-only**; max per-engine prediction difference dual vs long = 4.19 cycles |
| verdict | **present but not a learned gate, and not beneficial at span=70**. Because `m` saturates at 70, `g` is a constant 0.4 for every mature window; the last-50 mean and the all-70 mean are 99.2% identical. The name "dual-span gate" overstates what it does. |

### 2.8 Functions that are active but not in the advertised list

| function | code | evidence | verdict |
|---|---|---|---|
| learned distance-from-end positional table | `dspt.py:121-123`, applied `:213-214` | `temporal_pe` shape `(1,70,64)` = 4,480 params (5.8%) | **active**; this is the model's real positional encoding (there is no sinusoidal PE) |
| residual LayerNorm after the conv | `dspt.py:116`, `:210` | 128 params | active |
| asymmetric loss | `train.py:129` | `w(e)=1.8` if `e>0` else `1.0`, added at 0.5 weight → effective 1.9 late / 1.5 early | active; aligns the objective with the NASA score's late-penalty direction |
| auxiliary dual-span loss | `train.py:130-133` | `0.25·(MSE(y_A)+MSE(y_B))` | active during training only |
| `cond` input | `dspt.py:225-227` | no `cond_emb` is built when `sa_mode="off"` | passed but **unused** in Base |
| HI head | `dspt.py:148-149` | `has_hi_head=False` when `hi_source="none"` | **absent** in Base |

### 2.9 What Base does not contain

`ms_mode="off"` → no `conv5`, no `conv7`, no `ms_proj`; `sa_mode="off"` → no
`sensor_attn`, no `sensor_id`, no `cond_emb`; `hi_source="none"` → no `hi_head`;
`instnorm=False` → no 4th normalised channel; no static sensor weights.

---

## 3. Parameter budget

| component | DSPT-Base | Vanilla (raw) |
|---|---:|---:|
| temporal Transformer (2 layers, d=64) | 66,944 | 66,944 |
| learned positional table | 4,480 | — |
| regression head | 4,225 | 4,225 |
| input embedding | 256 | 960 |
| cycle projection | 576 | — |
| depthwise conv3 | 256 | — |
| LayerNorms | 256 | 128 |
| sensor score | 65 | — |
| **total** | **77,058** | **72,257** |

The two models differ by 4,801 parameters (6.6%). Capacity is therefore not a
credible explanation for a performance gap; the difference must come from the
input representation, the inductive biases, or the optimisation path.

---

## 4. Leakage audit

| check | method | result |
|---|---|---|
| training path never opens the official test files | monkeypatch `data.load_test` to raise, then call `prepare_train` for FD001 | **pass** — training completes; `training_path_opens_test=False` |
| scaler fitted on training engines only | recompute per-cluster means from the training frame and compare with `scaler.means` | **pass** — max abs difference **0.0** on FD001–FD004 |
| train / validation engines disjoint | set intersection | **pass** — 80/20, 208/52, 80/20, 199/50 |
| baseline reference is causal | perturb frames after the window end; then perturb frames before the last five early frames | **pass** — future perturbation changes nothing; only the last five early frames (`cycle ≤ 30`, `index ≤ t`) are used |
| validation cuts never read official RUL | cuts are `round(f·n)` on the training-file engine length | **pass** — e.g. `n=200` → cuts (80,110,140,170,190), targets (120,90,60,30,10) |
| model selection uses validation engines | `train.py:103-160` selects on `per_engine_rmse(va['unit'], …)` | **pass** — test is only opened by `evaluation/evaluate_test.py` behind a freeze manifest |

**The one real defect is the label definition, not contamination.**
`data.py:245` writes `float(min(rul_map[uid], max_rul))` into the test target.
The model also saturates near the cap: the audited `A8_base_L70` FD001 checkpoints
predict maxima of 127.5–128.0 while `y_true` maxes at 125. Capped and raw RMSE
therefore diverge sharply on FD002/FD004. That is a scoring-protocol
problem (郑总's audit quantifies it: FD002 11.118 → 22.965, FD004 12.423 → 23.881),
not evidence that information from the test set entered training.

**A second, non-leakage caveat.** The absolute cycle count is an observable input at
test time, so using it is legitimate. But §2.3 shows the model leans on it heavily
(+4.91 RMSE when shuffled). Any comparison against a method that does not consume
absolute cycle is therefore not a comparison of equal information sets (§6.3).

---

## 5. True Vanilla Transformer and the 2×2 comparison

### 5.1 Definition

`paper_b/models/vanilla.py` is an independent model, not a DSPT with switches off:

```
normalised 14-sensor vector → Linear(14→64) → fixed sinusoidal PE
→ 2 × pre-norm TransformerEncoderLayer (4 heads, GELU, padding mask)
→ LayerNorm → masked mean pool → Linear(64→64) → GELU → Linear(64→1)
```

Runtime checks: `embed.in_features=14`, **0** conv modules, no cross-sensor
attention, no learned positional table, no cycle projection, no dual-span, 72,257
parameters. It shares only `masked_mean` (a three-line helper) with `dspt.py`.

### 5.2 Feature controls

The method draft (`PAPER_B_METHOD_DRAFT.md:67-75`) promises feature-matched controls.
The grid had only two of the four corners, so the missing two were run on 2026-11-12
(24 cells, all complete):

| arm | input | cycle | params |
|---|---|---|---|
| `VAN_raw_L70` | 14 normalised sensors | no | 72,257 |
| `VAN_raw_cycle_L70` | 14 normalised sensors | yes | 72,833 |
| `VAN_dsp_L70` | 3-channel (x, dev, EMA) | no | 74,049 |
| `VAN_dsp_cycle_L70` | 3-channel | yes | 74,625 |

### 5.3 Results (engine-level validation RMSE, mean ± sd over seeds 101/202/303)

| arm | FD001 | FD002 | FD003 | FD004 | mean |
|---|---|---|---|---|---|
| **DSPT+3ch+cycle (Base)** | 11.085 ± 0.470 | 10.974 ± 0.069 | 11.381 ± 0.598 | 11.806 ± 0.448 | **11.312** |
| Vanilla raw | 12.142 ± 0.440 | 12.672 ± 0.214 | 12.353 ± 0.871 | 14.046 ± 0.485 | 12.803 |
| Vanilla +cycle | 11.824 ± 0.568 | 11.526 ± 0.057 | 11.529 ± 0.993 | 12.151 ± 0.562 | 11.758 |
| Vanilla +3ch | 12.077 ± 0.412 | 12.929 ± 0.504 | 13.108 ± 0.432 | 13.474 ± 0.506 | 12.897 |
| Vanilla +3ch+cycle | 11.494 ± 0.873 | 11.045 ± 0.480 | 11.796 ± 0.412 | 12.142 ± 0.301 | 11.619 |

All 48 Vanilla cells were produced on a40 (the two original FD001 arms were re-run
there so the whole 2×2 is same-machine). Base cells are the stored A8 runs from
lab3090; a same-config re-run of `A8_base_L70/FD001/s101` on a40 moved it from
11.601 to 11.379, so the cross-machine floor is **≈0.2 RMSE**. Effects below that
are not individually resolvable.

### 5.4 What the decomposition says

| effect (negative = first arm better) | FD001 | FD002 | FD003 | FD004 | mean |
|---|---|---|---|---|---|
| architecture: DSPT − Vanilla at 3ch+cycle | −0.409 | −0.071 | −0.414 | −0.335 | **−0.307** |
| cycle: Van+raw → Van+raw+cycle | −0.317 | −1.147 | −0.824 | −1.895 | **−1.046** |
| 3-channel alone: Van+raw → Van+3ch | −0.065 | +0.256 | +0.755 | −0.572 | **+0.094** |
| 3-channel given cycle: Van+cycle → Van+3ch+cycle | −0.330 | −0.480 | +0.267 | −0.010 | −0.138 |

Base's advantage over Vanilla-raw, **on the four-subset mean**, is 1.491
(12.803 − 11.312) and decomposes as:

| source | RMSE | share |
|---|---:|---:|
| absolute cycle feature | 1.046 | 70% |
| architecture (conv + sensor pooling + learned PE + dual span) | 0.307 | 21% |
| cycle × 3-channel interaction | 0.232 | 16% |
| 3-channel input on its own | +0.094 | +6% (costs) |

**Scope warning — this decomposition is scope-dependent, and the shares are not
portable across subsets.** Both rows below are engine-level (per-engine RMSE, then
unweighted mean over engines); they differ only in which engines are averaged:

| scope | total gap | cycle | architecture | 3-channel | interaction |
|---|---:|---:|---:|---:|---:|
| FD001 only | 1.056 | 0.317 (30%) | 0.409 (39%) | 0.065 (6%) | 0.265 (25%) |
| four-subset mean | 1.491 | 1.046 (70%) | 0.307 (21%) | +0.094 (−6%) | 0.232 (16%) |

The difference is FD002–FD004, where cycle is worth 1.15–1.90 while architecture
is worth 0.07–0.42. So:

- **"cycle is the largest single contributor" holds in both scopes.**
- **"cycle is 70%" is a four-subset statement**; on FD001 it is 30% and the
  architecture is 39%. Any quoted share must name its scope.
- The 1.491 four-subset figure is **not** the pooled number. It happens to sit
  within 0.001 of the FD001 *pooled* gap (13.366 − 11.874 = 1.492); that is a
  coincidence, and conflating the two is exactly the error this table prevents.

Three readings, in order of confidence:

1. **The cycle feature is the largest single factor in every scope**, worth ~1.0
   RMSE on the four-subset mean and up to 1.9 on FD004. It is consistent in sign
   on all four subsets.
2. **The 3-channel representation has no detectable effect of its own.** It is
   +0.09 on the four-subset mean and −0.07 on FD001, i.e. it flips sign, and both
   magnitudes are well below the 0.22 cross-machine floor. It cannot be called
   "harmful"; the correct statement is "no detectable independent effect". It only
   pays off in combination with the cycle code (interaction −0.23).
3. **The architecture adds a real but small margin**, 0.07–0.41 per subset. Because
   that is the same order as the cross-machine floor (0.2), the honest statement is
   "≈0.3, not individually resolvable per subset", not "DSPT beats a Vanilla
   Transformer by 0.3".

One caveat on the cycle result: the arm difference (1.0) and the inference-time
shuffle test in §2.3 (4.9) measure different things. The arm difference asks what a
model loses when it is *trained* without the feature and can compensate with the
sensors; the shuffle test asks how much a model that *was* trained with the feature
relies on it. Both are true: cycle is the largest single contributor, and the
sensors still carry most of the recoverable signal.

---

## 6. Answers to the five audit questions

### 6.1 Why is Base so strong?

Six factors, each with a measurement. They are not equally important.

1. **The metric is forgiving because the target is capped.** `data.py:245` clips
   the test target at 125 and the model saturates at 127.5–128.0, so on the capped
   criterion the error is measured against a target the model is already sitting on.
   Under raw labels the same predictions roughly double on FD002/FD004
   (11.118 → 22.965 and 12.423 → 23.881, 郑总's audit). Part of "strong" is a
   property of the scoring convention, not of the model.
2. **The absolute cycle feature carries most of the gap.** It accounts for 70% of
   Base's *four-subset* advantage over Vanilla-raw (30% on FD001 alone; §5.4), and
   corrupting it at inference costs 4.91 RMSE (§2.3). Base is, to a first
   approximation, a sensor model with a strong engine-age prior.
3. **The architecture adds ~0.3 RMSE.** At matched 3-channel + cycle inputs, DSPT
   beats Vanilla by 0.07–0.41 per subset (§5.4). Real, consistent in sign on three
   of four subsets, but at the cross-machine reproducibility floor.
4. **Capacity is not the explanation.** 77,058 vs 72,257 parameters (+6.6%), and
   86.9% of both models is the same 2-layer Transformer (§3).
5. **The loss matches the score.** The asymmetric term (`train.py:129`) weights
   late errors 1.9× vs early 1.5×, which is the same direction as the NASA
   asymmetric score.
6. **Sensor pooling is learned but only mildly selective.** Mean maximum sensor
   weight 0.154 vs 0.071 uniform (§2.6): the model concentrates somewhat, it does
   not collapse onto a few sensors.

The short answer: **Base is strong mostly because the protocol is forgiving and
because it consumes a strong absolute-cycle feature; the architecture contributes a
modest margin, not the main effect.**

A direct consequence for the paper: any claim of the form "our architecture reaches
RMSE 11" must state the label cap, the scoring convention, and whether absolute
cycle is an input. Without those three, the number is not comparable to the
literature (§6.3).

### 6.2 Is there actual data leakage?

No (§4). The pipeline never opens the official test files during training, the
scaler and HI statistics are fitted on training engines only, the baseline reference
is causal, and model selection is on held-out engines from the training file.

Two things must not be confused with leakage:

1. **The 125-cycle cap on the test target** (`data.py:245`) is a label-definition
   defect. It flatters capped RMSE and is why raw scoring roughly doubles the error
   on FD002/FD004.
2. **The absolute cycle feature** is observable at inference and therefore legal,
   but it is a strong shortcut (§2.3, §5.4) that changes what the model is actually
   doing.

### 6.3 Which published numbers are directly comparable?

The paper-level protocol matrix is 牢莫's `PAPER_PROTOCOL_MATRIX.md`; this audit adds
the model-side axes that must also match. A published result is directly comparable
to DSPT-Base only if all six hold:

1. training target capped at 125 with a piecewise-linear end-point;
2. test target **raw** (or the paper's own cap stated explicitly);
3. prediction clipping stated (none / [0,125] / [0,∞));
4. scoring function stated (RMSE, MAE, NASA asymmetric sum with its own constants);
5. validation split by engine, not by window;
6. **whether absolute cycle is an input** — a method that does not consume cycle is
   operating on a strictly smaller information set (§2.3, +4.91 RMSE when shuffled).

Axis 6 is the one this audit adds: it is usually not reported, and it can move RMSE
by more than any of the modules under study.

### 6.4 Do MS, SA and HI have any independent gain?

A8 grid, main effect of each switch (mean over the four configs with the switch on,
minus the four with it off; negative = helps). Validation, engine level, 3 seeds:

| switch | FD001 | FD002 | FD003 | FD004 | mean | seed sd of the arm means |
|---|---:|---:|---:|---:|---:|---:|
| ms | −0.368 | +0.068 | +0.019 | −0.047 | −0.082 | 0.14–0.60 |
| sa | −0.457 | −0.159 | −0.364 | **+0.330** | −0.163 | 0.15–0.85 |
| hi | +0.037 | −0.049 | +0.014 | −0.200 | −0.050 | 0.15–0.60 |

**No switch shows an independent gain larger than the seed spread.** The strongest
statement the data supports is narrower:

- `sa` is the only switch with a consistent direction on three subsets (helps by
  0.16–0.46 on FD001–FD003). That is at most ~1 sd, and it reverses on FD004.
  Under the raw criterion the FD004 effect flips sign (+0.26 capped → −0.82 raw,
  郑总's §7), so it cannot be written as a mechanism.
- `ms` is mixed and small (−0.37 on FD001, ≈0 elsewhere).
- `hi` is stably **near zero**, not stably useful. "Stable" and "effective" are
  different claims; only the first is supported.

The functional audit explains why the gains are small:

- **MS has little headroom**: the base already runs a global self-attention stack,
  so a 5/7-tap conv adds local smoothing on top of an already global receptive
  field (§2.4). It cannot add temporal context that attention does not already see.
- **SA is the only true cross-sensor mixing** beyond pooling, so it is the one with
  a plausible mechanism — but its effect is subset-dependent and small (§2.6).
- **HI is auxiliary supervision only.** With `hi_source="hf"` the prediction path
  is unchanged (`reg_head(h_f)` before and after); the HI head adds a loss term on
  the same embedding. An auxiliary regulariser that does not touch inference can
  only produce a small effect, and that is what is measured.
- **The dual span is not a gain at all**: `dual` is 0.073 RMSE *worse* than
  `long`-only on the audited cell (§2.7).

### 6.5 Is the current structure enough for an IEEE TIM method contribution?

**PENDING-L2.** This is a novelty judgement and is routed to the Codex L2 channel;
it is not decided here. The factual inputs for that decision are:

- the architecture is a composition of published components (depthwise temporal conv,
  per-sensor Transformer, attention pooling, Fourier cycle features, multi-span pooling);
- the measured independent gains of the three switches are within the seed spread,
  and the only consistent signal is that `sa` helps on FD001–FD003;
- the strongest empirical regularity found here is a **feature** (absolute cycle),
  not a mechanism;
- the dual-span gate, one of the advertised mechanisms, is a fixed length-dependent
  mix that is 0.073 RMSE worse than long-only on the audited cell.

Whether that composition plus the (currently unsupported) mechanism story clears a
TIM novelty bar is exactly the kind of judgement that must not be made by the
executor.

---

## 7. Threats to validity

1. **Runtime evidence is structural.** §2 proves modules are wired and active; it
   does not prove they help. Performance statements are restricted to §5 and §6.4.
2. **Machine non-determinism.** Retraining `A8_base_L70/FD001/s101` on a40 gave
   engine-level 11.379 versus 11.601 for the stored lab3090 run — a 0.22 RMSE
   difference at identical config and seed (`cudnn.benchmark=True`, bf16 autocast).
   The Base column in §5.3 is therefore a lab3090 run while all Vanilla cells are
   a40 runs; effects below ≈0.2 are not individually interpretable. Seed-level
   differences below that floor are not interpreted anywhere in this document.
3. **Small validation sets.** 20/52/20/50 engines per subset; the seed spread is
   0.07–0.99 RMSE, which is the same order as every module effect.
4. **Test coverage.** The A8 grid evaluates the official test set for seed 101 only;
   all module claims here are validation-based, with test used descriptively.
5. **Cycle ablation is inference-time.** Zeroing or shuffling `cycle` is
   out-of-distribution for the trained projection; the shuffle version preserves the
   marginal distribution and is the one quoted (+4.91).
6. **The DSPT-without-cycle arm was not trained.** It was queued, then stopped when
   an unrelated `code/unified/train.py` job took most of the a40 GPU and the run
   estimate rose to several hours. The cycle effect *within DSPT* is therefore
   estimated by the paired inference-time shuffle (§2.3), not by a retrained arm;
   the cross-architecture cycle effect in §5.4 is measured within Vanilla only.

---

## 8. Reproduction

All commands run from the repository root on a40 with
`source /media/sangfor/vdb/user/zxk/env.sh`.

```bash
# module wiring / runtime evidence
python -m paper_b.experiments.audit_dspt_base --out /tmp/audit_dspt_base.json

# preprocessing and leakage checks
python -m paper_b.experiments.audit_preprocessing --data-dir data/CMAPSS

# dual-span and cycle effect on real validation (needs a checkpoint)
python -m paper_b.experiments.retrain_base_for_audit          # -> /tmp/audit_runs
python -m paper_b.experiments.audit_dual_span_effect \
  --run-dir /tmp/audit_runs/A8_base_L70/FD001/s101 --data-dir data/CMAPSS

# missing Vanilla feature controls (24 cells, completed)
python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/vanilla_features2.json --data-dir data/CMAPSS

# same-machine re-run of the two original Vanilla FD001 arms (6 cells, completed)
python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/audit_van_fd001_a40.json \
  --out-root paper_b/results/raw_a40 --data-dir data/CMAPSS

# arm table (Base cells fall back to a8_engine_level.csv)
python -m paper_b.experiments.audit_arm_table \
  --roots paper_b/results/raw_a40,paper_b/results/raw
```

## 9. Evidence index

| artifact | content |
|---|---|
| `paper_b/experiments/audit_dspt_base.py` | module wiring, feature channels, pooling, gate, param budget |
| `paper_b/experiments/audit_preprocessing.py` | scaler provenance, causality, train/test separation |
| `paper_b/experiments/audit_dual_span_effect.py` | dual/long/short and cycle shuffle on real validation |
| `paper_b/experiments/audit_arm_table.py` | 2×2 arm table from saved `val_pred.csv` |
| `paper_b/results/vanilla_engine_level.csv` | 48 Vanilla cells, engine-level |
| `paper_b/results/audit_arm_table.csv` | merged arm table |
| `paper_b/results/dual_span_effect.json` | real-val dual-span / cycle numbers |
| `paper_b/experiments/grids/vanilla_features2.json` | the two missing feature-control arms |
| `paper_b/experiments/grids/audit_van_fd001_a40.json` | same-machine re-run of the two original Vanilla FD001 arms |
