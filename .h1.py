
import sys, glob, json
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from run_candidate import prepare
from candidates.model import CandidateModel

ckpts = sorted(glob.glob("experiments/order03/M4_racdr/FD002/RACDR_s2027/seed_2027/RACDR/final_model.pt"))
ckpts += sorted(glob.glob("experiments/order03/M4_racdr/FD002/RACDR_s2027/seed_2027/RACDR/best_model.pt"))
print("ckpt candidates:", [c.split('/')[-1] for c in ckpts])
path = ckpts[0]
state = torch.load(path, map_location="cpu")

train_set, val_set, *_ = prepare("FD002", 7, 50, 125.0, True, 6)[:4]
model = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4,
                       num_layers=1, dropout=0.15, use_baseline=True,
                       use_ema_features=True, use_cycle=True, use_graph=False,
                       use_dual_head=False, use_reliability=True)
model.load_state_dict(state, strict=False)
model.eval()

gates = []
def hook(module, inp, out):
    g = torch.sigmoid(out + model.sensor_gate_bias.view(1, 1, -1))
    gates.append(g.mean(dim=(1, 2)).detach())
h = model.reliability.register_forward_hook(hook)

loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False)
dis = []
with torch.no_grad():
    for x, y, cond, extras in loader:
        model(x, cond=cond, extras=extras)
        # e4 = exp(-deviation) at column sensors+1+3 = 18
        e4 = extras[:, 18].numpy()
        dis.append(-np.log(np.clip(e4, 1e-6, 1.0)))
h.remove()
g = torch.cat(gates).numpy().ravel()
d = np.concatenate(dis)
rho, p = spearmanr(g, d)
print(f"n={len(g)} gate_mean={g.mean():.4f} disagreement_mean={d.mean():.4f}")
print(f"spearman(gate, disagreement) = {rho:.4f}  p={p:.2e}")
q = np.quantile(d, [0.25, 0.5, 0.75])
bins = np.digitize(d, q)
for b in range(4):
    m = bins == b
    print(f"  disagreement quartile {b}: n={m.sum():4d} gate_mean={g[m].mean():.4f} disagreement={d[m].mean():.4f}")
json.dump({"spearman": float(rho), "p": float(p), "n": int(len(g)),
           "quartile_gate_means": [float(g[bins == b].mean()) for b in range(4)]},
          open("results/dsh/h1_gate_diagnostic.json", "w"), indent=2)
