# Innovation & evidence map (working note, 2026-10-30 00:30)

Purpose: align the three claimed contributions with what actually exists, and list
the experiments that would close each gap. Written before the FD001 replication
returns, so the "coverage" column will change.

## 1. Contribution → evidence → gap

### C-A. Resolution: how large a difference can this benchmark resolve?

| evidence | status |
|---|---|
| variance decomposition 2.6% / 31.5% / 65.9% | point estimates only, no CI (E6 downgrade) |
| MDE 1.60 (260) / 1.14 (509) / 0.97 (709) | arithmetic, **no artefact**; must be rebuilt as a script |
| engine-clustered CIs of real contrasts | recomputed: ±2–4 RMSE on 52 engines (E3), ±3.5 on R3.1 |

**Gap**: the headline number (1.6) has no code behind it. Fix: one script that takes
per-engine residuals and returns MDE, validated against the recorded 1.58 simulation.
**Innovation risk**: medium — it is a measurement claim, and it must be computable by
a reader in one command.

### C-B. Time is the dominant, and partially illegitimate, signal

| evidence | status |
|---|---|
| linear probe, 360 engines: removing time costs 6.89 [5.62, 8.17] | recomputed, E6 convention |
| frozen pipeline FD002: BE − A = +2.69 [1.35, 4.10] | recomputed; budget-robust (T1) |
| bounded encoding: −0.90 [−2.08, +0.32] | pre-registered margin not met (T0) |
| lifespan oracle +8.19 [5.39, 11.07]; deployable estimator −0.05 [−2.25, +2.13] | recomputed |
| E2: monotone index lifts the score but not `corr(T̂,T)` (0.347) | closed FAIL |

**Gap**: everything neural is FD002. FD001 replication launched 00:28.
**Innovation risk**: low if FD001 agrees; if it disagrees, the claim becomes
regime-specific and §4.3 must say so — still publishable, less quotable.

### C-C. Protocol and reproducibility

| evidence | status |
|---|---|
| from-scratch reproduction, Δ = 0.00000, replay 7.2e-6 | done (E4) |
| three self-corrections documented with dates (R3.1 leak, E3 bootstrap, C3 withdrawal) | done |
| pre-registration + amendment discipline | done |
| a reader-facing tool (MDE calculator + checklist) | **missing** |

**Gap**: the paper promises "protocol and tooling" in its thesis but currently ships
only the protocol. Fix: `scripts/mde.py` with a CLI, a worked example on the E3
residuals, and a 10-item checklist derived from the failures we actually hit.

## 2. Dataset coverage (the user's point)

| subset | linear probe | frozen pipeline | lifespan arms | status |
|---|---|---|---|---|
| FD001 | yes (R3-E, 100 engines) | **running** (A/BE/PE, 3 seeds) | no | partial → soon complete for §4.3 |
| FD002 | yes (260 engines) | yes (A/BE/PE, 3 seeds) | yes (6 arms, 3 seeds) | complete |
| FD003 | no | no | no | **sealed by protocol** |
| FD004 | no | no | no | **sealed by protocol** |

So the honest sentence for the paper is: *two of four subsets, chosen because the
other two are held out by the protocol* — not "C-MAPSS". FD003/FD004 cannot be added
without an explicit protocol change from the human.

## 3. Results section: what can be drawn today

| item | artefact | status |
|---|---|---|
| Fig. variance decomposition | `review_r1/c_interaction.json` | ready, needs a plotting script |
| Fig. MDE curve vs engine count | none | **blocked** on C-A |
| Fig. forest plot of contrasts | `results/dsh/e6/E6_stats.json` | ready |
| Fig. time arms (probe + pipeline, 2 subsets) | R3_preds + e3/e3_fd001 | ready after FD001 |
| Fig. lifespan six arms + T̂–T scatter | `r3_v3/*`, `L_strat.csv` | ready |
| Fig. severe-definition comparison | `A5_metrics_frozen_severe.csv` | ready |
| Table: protocol/data | this repo | ready |
| Table: six-arm R3.1 | `R3_1_final.json` | ready |
| Table: 21 pairs | — | **withdrawn** (E6) |
| Table: MDE bins | none | blocked on C-A |

## 4. What I would run next, in order of value per GPU-hour

1. **FD001 replication** (running, 0.6 GPU-h) — turns §4.3 from single- to
   two-subset. Highest value, already paid for.
2. **MDE calculator + validation** (CPU, ~1 h) — removes the only "no artefact"
   headline. No GPU.
3. **Capacity control for PE** (3 runs, ~0.6 GPU-h) — only if we want to keep any
   sentence about the bounded encoding beyond "compatible".
4. **7-cell tournament rebuild** (7 × 3 runs, ~4 GPU-h) — the only way to resurrect
   the multiplicity claim; otherwise delete it.
5. FD003/FD004 — **not without human authorisation** (protocol change).

Items 1 and 2 change the paper's defensibility; 3 is cosmetic until 1 lands; 4 is a
daytime decision because it is expensive and the claim can simply be dropped.
