#!/bin/bash
cd ~
if [ -d femto ]; then echo "exists"; else
  git clone --depth 1 https://github.com/Eng-Ismail7/ieee-phm-2012-data-challenge-dataset.git femto 2>&1 | tail -2
fi
cd ~/femto
du -sh . 2>/dev/null
echo "--- top level ---"
ls
echo "--- structure sample ---"
find . -maxdepth 2 -type d | head -12
echo "--- file sample ---"
find . -maxdepth 3 -type f | head -8
