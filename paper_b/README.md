# Paper B: DSPT RUL point prediction

This directory is the ACTIVE Paper B line. Paper A (conformal RUL intervals)
lives outside it and is not touched.

## Layout

```
paper_b/
  models/dspt.py            original DSPT with auditable module switches
  models/vanilla.py         independent Vanilla Transformer baseline
  training/data.py          C-MAPSS split, scaler, windows, HI, test builder
  training/train.py         unified trainer (engine-level early stopping)
  training/metrics.py       RMSE / MAE / NASA / bias
  evaluation/evaluate_test.py  official test, gated by the freeze manifest
  experiments/make_grids.py generates the grids from configs/protocol.json
  experiments/run_grid.py   idempotent, shardable grid runner
  experiments/analyze.py    aggregates raw runs into tables
  experiments/mechanism_probe.py  CPU probes (HI target, sensor heterogeneity)
  configs/protocol.json     frozen protocol constants
  experiments/grids/*.json  run definitions
  results/                  raw runs, summaries, freeze manifest
  reports/                  CURRENT_STATUS, EXPERIMENT_REGISTRY, ARCHIVE_INDEX,
                            PROTOCOL_AUDIT, DSPT_ABLATION_ANALYSIS,
                            IMPROVEMENT_RESULTS, NEXT_RESEARCH_DECISION
  archive/legacy_2026_11/   pre-takeover code and results (never paper evidence)
```

## Protocol in one paragraph

Official training engines are split 80/20 by engine id (`split_seed=7`,
`fold=0`). Model selection uses five engine-level truncations per validation
engine (40/55/70/85/95% of its life); the true RUL at a cut is the remaining
cycles. Model seeds are 101/202/303. The official test set is opened only by
`evaluation/evaluate_test.py` for tags listed in `results/FREEZE.json`, after
the structure and hyperparameters are frozen. All preprocessing statistics and
HI labels are fitted on training engines only.

## Reproduce

```bash
# 1. generate grids
python -m paper_b.experiments.make_grids

# 2. train one shard (idempotent: completed runs are skipped)
python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/ablation8.json \
  --shard 0 --nshards 2 --device cuda:0 --data-dir data/CMAPSS

# 3. aggregate
python -m paper_b.experiments.analyze

# 4. after creating results/FREEZE.json with the frozen tags:
python -m paper_b.evaluation.evaluate_test \
  --run-dir paper_b/results/raw/A8_base_L70/FD001/s101
```

## Non-negotiable rules

1. No test-set number may be produced before the freeze manifest exists.
2. No historical number is paper evidence unless it has a checkpoint and a
   prediction file produced by the current evaluator.
3. Negative results are reported, not hidden. The archive index keeps the old
   failures recoverable.
