#!/usr/bin/env bash
# Restore the six round-1 cells per subset that the accidental round-2 start
# overwrote. Round 1 is bit-reproducible from ad53371 (verified earlier: the pilot
# and the four-subset batch agreed on all 480 metrics, max diff 0.0), so re-running
# the same folds under that commit regenerates identical files.
set -u
cd "$HOME/deepseek-rul" || exit 1
PY=/home/user/miniconda3/envs/ncmapss/bin/python
WT=/tmp/r1
rm -rf "$WT" /tmp/r1out
git worktree remove --force "$WT" 2>/dev/null || true
git worktree add --detach "$WT" ad53371 2>&1 | tail -1
cd "$WT" || exit 1
echo "restore worktree at $(git rev-parse --short HEAD)"
for s in DS02 DS06; do
  h5=$(ls "$HOME"/deepseek-rul/data/NCMAPSS/extracted/N-CMAPSS_"$s"*.h5 | head -1)
  "$PY" -m paper_b.ncmapss.run_loeo --h5 "$h5" --subset "$s" --folds 1 \
        --rungs N0,N1 --out-root /tmp/r1out --ledger "/tmp/r1out/${s}.csv" \
        > "/tmp/r1_${s}.log" 2>&1
  echo "$s exit=$? rows=$(($(wc -l < "/tmp/r1out/${s}.csv") - 1))"
done
