"""Print the paper table from the evaluation report."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/cal/eval_pooled/eval_report.json"
d = json.load(open(path))

print("sub   method   rmse  cov50 cov80 cov90   w50   w80   w90    crps [95% CI]        wink90")
for s, r in d.items():
    for name, lab in (("ours", "layer"), ("homoscedastic", "homo"),
                      ("split_conformal", "split"), ("cqr", "cqr")):
        m = r[name]

        def g(k, fm="%5.2f"):
            return fm % m[k]["value"] if k in m else "    -"

        def ci(k):
            v = m.get(k)
            if v and v["ci"][0] is not None:
                return "[%5.2f,%5.2f]" % tuple(v["ci"])
            return " " * 13

        crps = "%6.2f" % m["crps"]["value"] if "crps" in m else "     -"
        print("%-5s %-7s %s %s %s %s %s %s %s  %s %s %s"
              % (s, lab, g("rmse"), g("cover50"), g("cover80"), g("cover90"),
                 g("width50"), g("width80"), g("width90"),
                 crps, ci("crps"), g("winkler90")))
    print("      PIT:", [round(x, 2) for x in r["pit_hist"]])
    print()
