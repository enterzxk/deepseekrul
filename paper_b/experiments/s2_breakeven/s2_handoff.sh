#!/bin/bash
# S2 handoff: wait for the S1-A grid to exit, then split S2 across both 3090 GPUs.
#
# Rules baked in (learned the hard way tonight):
#   * the wait condition is the process table (pgrep -f run_grid), never the clock
#   * every decision is logged with the timestamp and the evidence it was based on
#   * completed cells are skipped by train_s2.py itself, so a restart never re-trains
set -u

S2=/home/user/s2
LOG=$S2/handoff.log
PY=/home/user/miniconda3/envs/ncmapss/bin/python
START=$(date +%s)

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" >> "$LOG"; }

log "handoff started (pid $$); waiting for 'run_grid' to disappear"

# ---- 1. block until the S1-A grid process is gone -------------------------
while true; do
    pids=$(pgrep -f "run_grid" | tr '\n' ' ')
    if [ -z "${pids// /}" ]; then
        log "run_grid not present -> S1-A finished; waited $(( $(date +%s) - START ))s"
        break
    fi
    log "waiting: run_grid pids=[$pids]"
    sleep 60
done

# ---- 2. stop the single-card S2 -------------------------------------------
old=$(pgrep -f "train_s2.py --grid" | tr '\n' ' ')
log "stopping single-card S2 pids=[$old]"
pkill -f "train_s2.py --grid"
for _ in $(seq 1 30); do
    pgrep -f "train_s2.py --grid" >/dev/null 2>&1 || break
    sleep 2
done
left=$(pgrep -f "train_s2.py --grid" | tr '\n' ' ')
log "single-card S2 stopped; remaining pids=[$left]"

# ---- 3. restart as two shards, one per physical GPU ----------------------
cd "$S2" || { log "FATAL: cannot cd $S2"; exit 1; }

CUDA_VISIBLE_DEVICES=1 nohup "$PY" train_s2.py --grid --shard 0 --nshards 2 \
    --device cuda:0 > "$S2/grid_shard0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=0 nohup "$PY" train_s2.py --grid --shard 1 --nshards 2 \
    --device cuda:0 > "$S2/grid_shard1.log" 2>&1 &
p1=$!
log "launched shard0 pid=$p0 (GPU1) and shard1 pid=$p1 (GPU0)"

sleep 20
alive0=$(ps -p "$p0" -o pid= | tr -d ' ')
alive1=$(ps -p "$p1" -o pid= | tr -d ' ')
log "post-launch check: shard0 alive=[${alive0:-DEAD}] shard1 alive=[${alive1:-DEAD}]"
log "shard0 head: $(head -2 "$S2/grid_shard0.log" 2>/dev/null | tr '\n' ' ')"
log "shard1 head: $(head -2 "$S2/grid_shard1.log" 2>/dev/null | tr '\n' ' ')"
log "handoff complete"
