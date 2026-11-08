#!/bin/bash
cd ~/femto
for d in Learning_set Test_set Full_Test_Set; do
  echo "[$d] $(ls $d 2>/dev/null | wc -l) bearings: $(ls $d 2>/dev/null | tr '\n' ' ')"
done
echo "--- snapshots per bearing (Learning) ---"
for b in Learning_set/*; do echo "$(basename $b): $(ls $b | wc -l)"; done
