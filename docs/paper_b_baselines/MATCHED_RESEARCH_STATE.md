# Matched baseline experiment restore point

2026-11-12: user explicitly authorized uniform-protocol reruns and a fair
DSPT comparison. New source/code/protocol and84 cells are registered in
paper_b/baselines/open_source_matched_20261112/. Snapshot9 files are byte
exact; main source observed HEAD7d8fa3ecc4a47bfd089c4822b24942ce8466b436.
Old120-run results are untouched and must never be pooled with this batch.

Models: aidgn,two_phase,dual_mixer,ffve,fcstgnn,dspt_v0_cycle(primary),
dspt_train_defaults(auxiliary); fourFDs,seeds101/202/303. Common FP32,
TF32 off, all-prefix sample universe, engine-mean selection,40/12/1e-4.
All84 train first, immutable batchfreeze, then one official TEST per run.

Remote root /media/sangfor/vdb/user/zxk/paper_b_matched_baselines_20261112;
Python /home/yndx/miniconda3/envs/a40_env/bin/python; data read-only
/media/sangfor/vdb/user/zxk/deepseek-rul/data/CMAPSS.

Preflight attempt1 passed28 real-data architecture/backward/data cases.
A pre-launch independent review found a failure-retry gap in the TEST quota;
a permanent TEST_ATTEMPT and failure markers now block every repeat. Train
and paired-data hashes are rechecked before freeze and TEST, immutable
BATCH_FREEZE and independent SSE/state/earlystop replay added. Preflight2 is
running against the final462-file execution manifest before first training.

Remaining: complete preflight2, commit exact launch source, launch train84;
while training add reporting/independent artifact validation without changing
frozen training modules; independent trained-run audit; batchfreeze; test84;
collect all checkpoints/predictions/time/params/config hashes; compare both
fixed DSPT references with all baselines, separate Two-Phase extra data;
verify local/Git artifact hashes and frozen boundaries; commit/push GitHub.
No DSH invocation and no user clarification/approval is needed for this scope.
