import json, os
for f in range(3):
    p = "experiments/punet_e2e/FD004/f%d_s42/e2e_cqr_none.json" % f
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    print("f%d" % f,
          "winkler", round(d["winkler"], 1),
          "picp", round(d["picp"], 3),
          "pinaw", round(d["pinaw"], 3),
          "q_hat", round(d["q_hat"], 2),
          "centre_rmse", round(d.get("best_centre_rmse", float("nan")), 2),
          "epoch", d.get("best_epoch"),
          "engMin", round(d["engine_min"], 3))
