"""Generate the build_results spec from the locked reference."""
import json
from pathlib import Path
lock = json.loads(Path("REFERENCE_LOCK.json").read_text())
subsets = list(lock["subsets"])
spec = {"groups": [{
    "protocol": "C_fullbudget_outer",
    "base": "runs",
    "pattern": "{arm}_{seed}.csv",
    "arms": ["M0", "OPS", "FULL", "NORM", "PHASE", "DIFF"],
    "seeds": [101, 202, 303],
    "model_type": "single",
    "reference": {s: lock["subsets"][s]["reference"] for s in subsets},
}]}
Path("spec.json").write_text(json.dumps(spec, indent=2))
print(json.dumps(spec["groups"][0]["reference"], indent=2))
