
import json, glob, csv, os
rows = []
for root in ("experiments/p10/final2", "experiments/p10/final3", "experiments/p10/gate", "experiments/p10/sweep"):
    for path in glob.glob(root + "/*/*/seed_*/*/metrics.json"):
        d = json.load(open(path)); tm = d["test_metrics"]
        rows.append({"root": root, "dataset": d["subset"], "variant": d["variant"],
                     "seed": d["seed"], "rmse": tm["rmse"], "nasa": tm["nasa_score"],
                     "mae": tm["mae"], "mono": tm["mono_violation_rate"],
                     "params": d["num_parameters"], "best_epoch": d["best_epoch"],
                     "epochs_ran": d["epochs_ran"], "seconds": d["elapsed_seconds"]})
rows.sort(key=lambda r: (r["root"], r["dataset"], r["variant"], r["seed"]))
os.makedirs("results/p10", exist_ok=True)
with open("results/p10/p10_runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("rows", len(rows))
