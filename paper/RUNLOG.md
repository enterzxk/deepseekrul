# RUNLOG

One row per completed task. Columns: task | UTC | command (abridged) | exit | artefacts | sha | verdict.

| task | utc | command | exit | artefacts | sha | verdict |
|---|---|---|---|---|---|---|
| probe | 2026-10-29T15:51Z | automation one-shot: append line | 0 | paper/RUNLOG.md | — | DRIVER PROBE OK |
| T0 | 2026-10-29T15:20Z | `dsh_e3_bootstrap_stability.py --big 100000 --seeds 9000..9004` | 0 | `results/dsh/e3/bootstrap_stability.{json,csv}`, `paper/E3_AMENDMENT_1.md` | — | **FAIL**: converged upper bound +0.317 > +0.30 |
| T1 | 2026-10-29T16:05Z | `run_candidate.py --variant BE --patience 80` ×3 | 0 | `experiments/e3fix/*` | runner 13bb6ede | no change: BEfix = BE to 4 d.p. |
| T2 | 2026-10-29T16:20Z | clean-dir A_s2027 + `dsh_e4_replay.py` | 0 | `experiments/e4/*`, `results/dsh/e4/replay.json` | runner 13bb6ede | **PASS**: ΔRMSE 0.00000, replay 7.2e-6 |
| T3 | 2026-10-29T16:05Z | `paper_e6_stats.py` (100k × 3 RNG × 10 contrasts) | 0 | `results/dsh/e6/E6_stats.json`, `paper/E6_STATS_AUDIT.md` | — | CLOSED; C3 withdrawn (no per-engine artefact) |
| T4 | 2026-10-29T16:12Z | writing only | — | `paper/SECTION_4_3.md` | — | drafted, new sign convention |
| T5 | 2026-10-29T16:15Z | writing only | — | `paper/MORNING_20261030.md` | — | done |
| T6 | 2026-10-29T16:43Z | `python scripts/e1_verify_refs.py fetch` then `… write` (Crossref + OpenAlex + S2 resolution; quote asserted as exact substring) | 0 | `paper/refs_verified.csv` (12 rows), `results/dsh/e1/refs_metadata.json`, `results/dsh/e1/refs_verified.pre_T6.csv`, `paper/E1_PREREGISTRATION.md` | refs_verified.csv `cc88f3bc`; e1_verify_refs.py `7a2cce33`; prereg `4e76604e`; metadata `d422f757` | **PASS**: 12 verified rows (target 12–15); 6 added, 6 withheld for missing abstract |

| T5b | 2026-10-29T16:30Z | `run_candidate.py --subset FD001` A/BE/PE x3 | 0 | `experiments/e3_fd001/*`, `results/dsh/e3/FD001_contrasts.json`, `paper/E3B_FD001_RESULT.md` | prereg 3c2b8ea0 | PE margin FAIL again (+0.319); BE-A 1.67 with CI crossing 0 (MDE 2.38) |
| T5c | 2026-10-29T16:55Z | `mde.py` on E3/E3b residuals | 0 | `results/dsh/e6/MDE.json`, `scripts/mde.py` | — | MDE 1.46 @260 (FD002), 1.14 (FD001); unsourced 1.60 replaced |
| T6 | 2026-10-29T16:05Z | hourly driver hop (00:05 local) | 0 | `paper/refs_verified.csv` (+6 rows) | — | done by driver; **audited**: 1 verbatim, 5 metadata-only, no fabricated DOI |
| T6a | 2026-10-29T16:50Z | concurrency audit + lock fix | — | `paper/SCHEDULER_PROBE.md` | — | driver ran despite the lock (content timestamp misread); automation replaced with an mtime-based guard |
| T7 | 2026-10-29T17:05Z | writing only | — | `paper/SECTION_2.md` | — | §2 drafted from verbatim refs only |
| T8 | 2026-10-30T00:45Z | `run_candidate.py --variant PS` x3 seeds x2 subsets | 0 | `experiments/e3c/*`, `paper/E3C_RESULT.md` | model e2237626 / runner 65eb5aca | PS-PE +3.51; PS-BE -0.08 -> capacity hypothesis rejected |
| T9 | 2026-10-30T12:52Z | `eval_resolution.py` (post-hoc, not preregistered) | 0 | `results/dsh/e7/RESOLUTION.json`, `paper/E7_RESOLUTION_NOTE.md` | — | multi-cutoff width 0.66-0.75 at fixed engines; cutoff placement flips the time contrast |
| T10 | 2026-10-30T13:00Z | `dsh_e9_variance_decomp.py` | 0 | `results/dsh/e9/variance_decomp.json` | — | engine share 97.6%, seed 2.4% -> more seeds cannot help; only more held-out engines can |
| T11 | 2026-10-30T13:05Z | `run_candidate.py --fold k` added (default -1) | 0 | `scripts/run_candidate.py` (65ea87ed), `src/candidates/model.py` (e2237626) | equality check pass=True | fold 0 == archived split7 for both subsets; 5 folds cover 260/260 and 100/100 with no overlap |
| T12 | 2026-10-30T13:10Z | 24 runs `--fold 1..4 --variant {A,PE} --seed 2027-2029` | running | `experiments/e10/*` | prereg aa7072f7 | E10 5-fold CV; 4 concurrent, ~27 s/epoch |
| T13 | 2026-10-30T13:12Z | `paper_variance_decomp.py` | 0 | `results/dsh/e10/variance_two_designs.json` | — | design B reproduced: 2.57 / 31.52 / 65.91 (matches the archived 2.6/31.5/65.9) |

| T14 | 2026-10-30T14:50Z | 24 runs + 2 reruns; `dsh_e10_cv_analyze.py` | 0 | `experiments/e10/*`, `results/dsh/e10/E10_cv.json`, `paper/E10_RESULT.md` | prereg aa7072f7 | **NOT_REPRODUCED**: PE-A pooled -0.126 (-0.9%) CI [-1.073,+0.795]; per-fold -0.90..+1.70; PE wins 51% |
| T18 | 2026-10-30T15:30Z | 12 BE runs; `dsh_e10b_analyze.py` | 0 | `experiments/e10b/*`, `results/dsh/e10/E10b_be_cv.json`, `paper/E10B_RESULT.md` | prereg 532a1d5d | **REPRODUCED**: BE-A +1.133 (+8.1%) CI [+0.245,+2.023]; single split +2.693 was 2.4x too high |
| T19 | 2026-10-30T15:45Z | writing only | — | `paper/SECTION_4_1.md`, `SECTION_4_2.md`, `SECTION_4_3.md`, `SECTION_4_4.md`, `SECTION_5.md`, `SECTION_6.md` | — | sections 4.1-4.6 drafted around the two-contrast calibration |

| T20 | 2026-10-31T09:45Z | fold composition check | 0 | `results/dsh/e16_fold_composition.csv` | — | folds balanced on lifespan (202-216 vs overall 206.8); condition mix varies 8-23% but no association with fold error |
| T21 | 2026-10-31T09:50Z | TOST on within-tier pairs | 0 | `results/dsh/e16_tost.json` | — | **4/4 fail at delta=1.0**; min margin 1.10/1.25/1.68/1.72 -> "equivalent" is not claimable |
| T22 | 2026-10-31T09:55Z | corrected architecture contrasts | 0 | `results/dsh/e16/arch_corrected.json`, `scripts/dsh_e16_arch_corrected.py` | — | **CI bug found** (fold 0 counted 3x, fold 4 missing): A-tf -1.433 [-3.00,+0.13] NOT significant; 4 of 5 contrasts stand |
| T23 | 2026-10-31T10:05Z | E17 first launch | 0 | void, 7 runs | model 4c57... runner 60f14f50 | **no-op**: baseline branch hard-codes d_model=88, --d-model never applied; the 7 runs are bit-identical re-runs of E11 (kept as a determinism check) |
| T24 | 2026-10-31T10:10Z | E17b capacity sensitivity (d=128) | running | `experiments/e17b/*` | prereg 8335783d | 10 runs, tf and cnn1d at doubled width |

| T25 | 2026-10-31T14:45Z | E20 launch attempt 1-2 | 1/90 | `/tmp/e20_status.txt` (void) | — | **three launch defects, zero results lost**: (1) xargs -a without -n ran 1 job of 45; (2) prepare() asserted every fold = n_val but FD004 has 249 engines and the prereg already declared fold 4 = 49, so 9 runs hit the assertion; (3) a corrupted quote in run_candidate.py was caught by py_compile before any job started |
| T26 | 2026-10-31T14:45Z | corrected relaunch | running | `experiments/e20/*` | prereg 3f0142f9 (Amendment 1) | FD003 44 runs (4 slots) + FD004 fold 4 9 runs (2 slots, 49 val engines); folds 0-3 of FD004 (36 runs, rc=0) unaffected; a fourth defect (CRLF line endings from a Python-written launcher) was found by bash -n before launch |

## Scheduler

* WorkBuddy CLI: `EADDRINUSE` on the default port, then `Authentication required`
  with exit 0 — abandoned, no desktop token was touched (`paper/SCHEDULER_PROBE.md`).
* Automation: one-shot dry run verified at 23:51; hourly driver active 00:05–09:00
  with a 90-minute expiring lock in `paper/LOCK`.
