
cd /home/yndx/zxk/bearing-rul-pi
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
python - <<'EOF'
import pandas as pd, glob
for p in sorted(glob.glob("experiments/p9/final/FD002/transformer_s2027/seed_2027/transformer/history.csv") +
                glob.glob("experiments/p9/final/FD002/p9_full_s2027/seed_2027/p9_full/history.csv")):
    h = pd.read_csv(p)
    print("==", p.split("/")[4])
    best = h.loc[h.validation_rmse.idxmin()]
    print(f"   best monitor rmse {best.validation_rmse:.3f} at epoch {int(best.epoch)}; final epoch {int(h.epoch.max())} rmse {h.validation_rmse.iloc[-1]:.3f}")
    for ep in (1, 3, 5, 8, 10, 15, 20, 30, 40):
        r = h[h.epoch == ep]
        if len(r): print(f"   ep{ep:02d} val={r.iloc[0].validation_rmse:6.3f} train_loss={r.iloc[0].train_loss:8.1f}")
EOF
