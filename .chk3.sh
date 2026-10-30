
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate') controllers=$(pgrep -fc '[h]2.sh')"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ls experiments/order03/H2_*/FD002/*/metrics.json 2>/dev/null | wc -l
ls experiments/order03/M4_ckpt/FD002/*/best_model.pt 2>/dev/null | wc -l
ls -t logs/order03/H2_*.log 2>/dev/null | head -2 | xargs -r tail -n 1
