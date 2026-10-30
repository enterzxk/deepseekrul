
cd /home/yndx/zxk/bearing-rul-pi
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/p10_summarize.py
awk '/FD002 5-seed/,0' results/p10/p10_summary.md | head -n 32
