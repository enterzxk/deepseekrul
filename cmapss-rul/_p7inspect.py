import glob,json
for f in sorted(glob.glob("experiments/final/FD001_p7/seed_*/seed_*/full/metrics.json")):
 d=json.load(open(f)); m=d["test_metrics_clip125"]; print(d["seed"],round(d["best_validation_rmse"],3),round(m["rmse"],3),round(m["nasa_score"],1),round(m["late_fraction"],3),m["mono_violation_rate"])
