
import json, glob, csv, os
rows = []
for root in ("experiments/p9/final", "experiments/p9/final_ep6", "experiments/p9/gate"):
    for path in glob.glob(root + "/*/*/seed_*/*/metrics.json"):
        d = json.load(open(path)); tm = d["test_metrics"]
        rows.append({"root": root, "dataset": d["subset"], "variant": d["variant"],
                     "seed": d["seed"], "rmse": tm["rmse"], "nasa": tm["nasa_score"],
                     "mae": tm["mae"], "mono": tm["mono_violation_rate"],
                     "params": d["num_parameters"], "best_epoch": d["best_epoch"],
                     "epochs_ran": d["epochs_ran"], "val_rmse": d["best_validation_rmse"],
                     "seconds": d["elapsed_seconds"]})
rows.sort(key=lambda r: (r["root"], r["dataset"], r["variant"], r["seed"]))
os.makedirs("results/p9", exist_ok=True)
with open("results/p9/p9_runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("rows", len(rows))
