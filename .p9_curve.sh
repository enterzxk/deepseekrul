
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import pandas as pd, glob
for p in ("experiments/p9/gate/FD002_p9_full/seed_2027/p9_full/history.csv",
          "experiments/p8/final/FD002/p8_full_s2027/seed_2027/p8_full/history.csv",
          "experiments/p9/gate/FD001_p9_full/seed_2027/p9_full/history.csv"):
    try:
        h = pd.read_csv(p)
    except Exception as e:
        print(p, "ERR", e); continue
    print("==", p.split("/")[2], p.split("/")[-2])
    for ep in (1, 3, 5, 8, 12, 16, 20):
        r = h[h.epoch == ep]
        if len(r):
            r = r.iloc[0]
            print(f"  ep{ep:02d} loss={r.train_loss:9.2f} val={r.validation_rmse:6.3f} late={r.validation_late_fraction:.2f} bias={r.validation_mean_bias:7.2f}")
EOF
