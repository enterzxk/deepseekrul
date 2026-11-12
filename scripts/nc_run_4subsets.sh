#!/bin/bash
# Four-subset LOEO runner. One process per GPU, two subsets per process, sequential.
# Each subset is a separate invocation so a crash cannot take the next subset down.
set -u
cd /home/user/deepseek-rul
PY=/home/user/miniconda3/envs/ncmapss/bin/python
D=data/NCMAPSS/extracted
run_one() {
  local gpu=$1 subset=$2 h5=$3 sha=$4
  echo "[$(date -Is)] start $subset on GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY -m paper_b.ncmapss.run_loeo \
    --h5 "$D/$h5" --subset "$subset" --data-sha256 "$sha" \
    --out-root experiments/ncmapss \
    --ledger "experiments/ncmapss/$subset/runs_dev.csv" \
    > "logs/$subset.log" 2>&1
  echo "[$(date -Is)] done $subset rc=$?"
}
case "$1" in
  0) run_one 0 DS02 N-CMAPSS_DS02-006.h5 47971a68b239ecb756833218a95d68ded6eb7e63ee84e86671c8b188de1ca765
     run_one 0 DS05 N-CMAPSS_DS05.h5 ef99ead42ac184503035895797c16e9a7e42554a08074aa760d0bba39db42e23 ;;
  1) run_one 1 DS06 N-CMAPSS_DS06.h5 91ccf478b21dd6337cfa608f20b647ec04228d59cf4ee59be87fad5f08b17b20
     run_one 1 DS07 N-CMAPSS_DS07.h5 2f88abc61a532d2e30a6e5ff100a511be4921484a6c7071bc8504e300da4d3bd ;;
esac
