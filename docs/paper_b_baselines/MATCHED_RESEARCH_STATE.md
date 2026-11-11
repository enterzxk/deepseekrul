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

## Launch checkpoint

Preflight2 passed all28 real-data/model/backward cases. The final execution
manifest has462 files, SHA256
3bbb4c5a6e5d48551adc75089eb4c8de5582ee7ff2c9a7dc65beaa63fac3ef9a.
Local committed launch source2e7e5beb785389c7f587e3498a32088c82f30def.
Final integrity audit PASS (462 sourcehashes,9snapshot hashes, noP1).
Training84 launched background PID1755517,3workers, train_matrix.log.
First snapshot: AIDGN+TwoPhase FD001s101 TRAINED, both DSPT references
FD001s101 and Dual-Mixer active;0failures; no new official TEST opened.
Use new monitor.py --root experiments for compact counts; reporting-only
files added after source freeze are not training dependencies. ANALYSIS_PLAN
predeclares paired comparisons against both named references and separate
Two-Phase resource condition. Do not edit frozen runner/common/data/reference
modules after launch. Old files unchanged.

## Shared-GPU resource recovery (training code unchanged)

First sevenFD001/s101 configurations allTRAINED,0failures,0TEST. External
HC resource control SIGSTOPped original orchestrator1755517 (confirmed
T/do_signal_stop); all its remaining children later finished and were
zombies. Main workspace research/agent-state/state.json explicitly records
the shared40GB gate/44GB redline and the SIGSTOP; no model/data failure.
After verifying exact own cwd/cmdline and no active own RUNNING workers,
terminated only that stopped CPU controller (SIGTERM thenSIGCONT), archived
its claim, and adopted all7 completed cases without re-running them.
No other job/process was signalled.

New finite memory_queue.py PID1838748, log train_resume_matrix.log. Admission
uses36000MiB total-budget and conservative per-model peak reservations, max3
own workers; first fitting remaining cell may run while heavy references
wait. Same sealed runner, configs, seeds,84 tuples and462 source hashes; no
training mathematics changed. Scheduler-only code hash/resource explanation
recorded in experiments/RESOURCE_LAUNCH_CONTROL.json and preserved separately.
Do not resume old controller. Monitor newcontroller state; if external
resource control stops it again, respect admission budget and do not stop
other jobs. Existing otherHC jobs currently~33GB, so memory gate can wait
until capacity is available. This is resource waiting, not a request to user.

## Memory-control refinement and metadata review

Integrity reviewer found only metadata/control gaps; frozen training remains
unchanged. Collector now rejects before anyRUL read unless immutable batch
freeze and all84TEST completions exist. Verifier binds actual archived
execution/snapshot/protocol bytes and84unique batch entries to eachconfig.
Accepted epoch/state, score rows, complete summary and paired statistics
are independently checked. Own frozen executable code/snapshot is archived
without re-uploading unlicensed upstream Python (manifests/fetcher restore it).

MemoryqueueV1 reservations were conservative but could undercharge an
observed task above its reservation. Preserve V1 code/hash/claim/log; stop
only itsCPU controller1838748, leave allrunning workers unchanged, adopt
live claims into newmemory_queue_v2.py PID1883851. Charge eachactive task
max(reservation,observed); budget36000MiB/max3. Log train_resume_v2.log;
event experiments/RESOURCE_LAUNCH_CONTROL_V2.json. Adopted live PID1839082
TwoPhaseFD002s101,1869092DSPT-V0FD002s101,1879800DualFD003s101.
No model was restarted, no training worker or other job signalled.
Frozen math still462files/SHA3bbb4c5... and original launchcommit2e7e5be.
All sourcecontrol versions retained; no newTEST has been opened.

## Validation-only fairness amendment before any TEST

Independent L2 PB-OS-009 chooses second condition: fixed40 kept as budget
sensitivity; method-author ceiling condition becomes strong-baseline main
comparison. Evidence TwoPhaseFD002s101 best40/val24.1236 still sharply
improving (32val34.5001), sourceLR1e-4/max600. Do not weaken this baseline
by interpreting an unconverged40epoch cutoff as method capability.

Author ceilings AIDGN300,TwoPhase200/600,Dual100,FFVE100000,FC41,DSPT40.
All84case eligibility follows exactregistered rule oldepochs_run40 AND
40-best_epoch<12 ANDceil>40. Eligible fresh same-seed fullretrain in newroot
experiments_author_limits using new siblingpackage
paper_b/baselines/open_source_author_limits_20261112. Othercases reuse
oldweights honestly with originalconfig/hash and scheduler max-independence
proof. No direct continuation from model-only checkpoints. Both conditions
fully trained/frozen before anyTEST; preserve old40root and everyresult.
No TEST has been opened; no hyperparameter selected fromTEST.

Newsource460files frozen in that package's EXECUTION_SOURCE_MANIFEST;
commonconfiguration alone adds method epoch ceilings and condition tags;
runner AST equal except metadata protocol string, data/referencewrapper
byteequal, same9snapshot, sameupstreamadapters. AUTHOR_LIMIT_POLICY.json
and AUTHOR_LIMIT_EQUIVALENCE.json in originalmatched package.

Combinedqueue.py is scheduler-only: adopts originalfixed40 liveworkers,
max3/36GB across both conditions; original84+eligible fresh cases, dynamic
qualification only from immutable TRAINED metadata, no numerical result
ranking. Final AUTHOR_LIMIT_ELIGIBILITY.json must have84rows andpolicyhash
before freeze/test. Need later build jointfreeze/evaluationplan andcollect
primary author-limit logical84 plus fixed40logical84, actualtraining84+K.
Do not call them168 independent trainings. DedupTEST only after proving
same state/input/evalcontract; preserve reuse sourcepred hashes.
