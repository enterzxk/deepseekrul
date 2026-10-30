#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
D=results/dsh/review_r3/sources
mkdir -p $D
dl() {
  local url=$1 out=$2
  for try in 1 2; do
    if [ -s "$D/$out" ]; then echo "$out exists"; return 0; fi
    curl -sSL --max-time 120 --retry 0 -o "$D/$out" "$url" && echo "$out try$try ok $(stat -c%s $D/$out)" && return 0
    echo "$out try$try FAILED rc=$?"
  done
  return 1
}
dl https://arxiv.org/pdf/2607.24809 p2607_24809.pdf
dl https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf ntrs_ncmapss.pdf
dl https://arxiv.org/pdf/2211.14730 patchtst.pdf
echo '--- sizes ---'
ls -l $D | tail -5
echo '--- extract ---'
for f in p2607_24809 ntrs_ncmapss patchtst; do
  if [ -s "$D/$f.pdf" ]; then
    pdftotext -layout "$D/$f.pdf" "$D/$f.txt" 2>/dev/null && echo "$f pages=$(grep -c $'\f' $D/$f.txt 2>/dev/null) chars=$(wc -c < $D/$f.txt)"
  fi
done
