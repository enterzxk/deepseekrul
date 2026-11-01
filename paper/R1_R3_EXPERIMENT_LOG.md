# R1–R3 experiment log (2026-11-01)

All runs: lab3090, `/home/user/yr/zxk/bearing-rul-pi`, env `baseline`, FD001 fold 0,
seed 2026, engine-level split from the frozen E21 protocol (`dsh_e21_test_all.split_frames`).
"test" = official FD001 test set (100 engines), capped truth, one prediction per engine
at its last observed cycle.

## R1 — is a compact innovation filter trainable at all? (gate run)

Model: `h_t = h_{t-1} + 0.1 tanh(W_h h + W_nu nu + b)`, `nu_t = x_t - g(h_{t-1})`,
discrete hazard over RUL 0..124 + `>=125` tail, censored NLL. 10,844 params, 72 s.

| gate | result | threshold | verdict |
|---|---|---|---|
| G1 dev RMSE vs constant (34.17) | 11.83 | < 27.3 | PASS |
| G2 time-reversal mean delta | 41.8 cycles | >= 1 | PASS (CET: 2e-5) |
| G3 innovation std / dh std / grad(W_nu) | 1.58 / 0.070 / 6.4 -> 0.13 | > 0 | PASS |
| G4 absorbing clamp | none in graph | none | PASS |

Official test: **13.45**. Capacity sweep: d=16 best, d=24 11.91 dev, d=32 13.49 dev
(overfits). Conclusion: the recursion trains, and the output genuinely depends on the
input — the CET deadlock does not reproduce here.

## Negative results (each with a mechanism, not a tuning excuse)

1. **Deployment-style truncation training** (one random prefix per engine per epoch):
   test 15.89 (d16) / 16.26 (d24), worse than full-sequence training 13.45.
   Dense per-cycle supervision over the whole run-to-failure sequence is better here.
2. **Encoder-conditioned observation** (`mu_t = g(h_{t-1}, e_{t-1})`, causal conv encoder):
   test **17.03**, dev 20.10, and dev degrades monotonically after epoch 20 while train
   NLL falls. Mechanism: `e_{t-1}` already contains `x_{t-1}`, so `nu_t ~ dx_t`; the state
   integrates a difference signal and loses its level anchor. Also 80k params memorise
   engine fingerprints through the head (fixed by restricting the head to `[h, age]`,
   which then exposed the innovation problem).
3. **Bias state fed by the full innovation**: test **19.81** vs 14.02 control. The bias
   absorbs the common degradation drift and starves the health state — the h-vs-bias
   confounding the CIDO audit predicted.
4. **Bias state fed by the channel-specific innovation** (`nu - mean(nu)`), decay 0.995:
   test 13.83 vs 13.41 control — no longer harmful, but no gain.

## The operator's falsification test (the one that decides the line)

Fault-augmented training (50% of engines per epoch get one random channel, random start,
1-4 sigma step). Then inject a held-out 3 sigma step on one channel at 60% of life.

| model | test RMSE | delta RUL @3σ | delta RUL @6σ | attribution ratio |
|---|---|---|---|---|
| bias=0 (control) | 12.96 | 25.9 | 31.8 | — |
| bias=1 (specific) | 13.89 | 18.1 | 22.2 | 0.69 |

Attribution ratio = mean |b| on the injected channel / mean |b| on the others
(>1 means the fault was localised). Channels 11 / 4 / 9 at 3 sigma give
0.69 / 0.64 / 0.71: consistently **below 1**, i.e. the bias state shifts to the wrong
channels. Robustness gain is ~30%, not the near-invariance the mechanism claims.

**Verdict: the per-channel bias operator fails its own pre-registered test. NO-GO.**

## Where the filter's error lives (bias=0, test)

| RUL band | n | RMSE | bias | cover90 |
|---|---|---|---|---|
| [0,25) | 19 | 4.6 | +0.9 | 0.95 |
| [25,50) | 11 | 8.1 | +0.8 | 0.82 |
| [50,75) | 13 | 11.4 | +5.0 | 0.62 |
| [75,100) | 23 | 17.6 | +4.4 | 0.57 |
| [100,126) | 34 | 15.4 | -3.9 | 0.62 |
| overall | 100 | 13.41 | +0.58 | 0.69 (width 18.6) |

Signature: predictions compressed toward the middle (~90-95) and intervals too narrow.
The filter is not sharpening near failure; 90% coverage is 0.69 against nominal 0.90.

## Standing numbers for comparison

* A-arm (window model, 41.9k): official FD001 **10.45**, FD002 11.96, FD003 10.09, FD004 12.27.
* Dual-Mixer reproduction (unified protocol): 13.39.
* Best compact filter so far: **12.96** (bias=0, fault-augmented), single fold/seed.

Gap to the A-arm: 2.5 RMSE. No configuration tested here closes it; every added
component (encoder, bias, truncation) moved the number the wrong way for an identifiable
reason.
