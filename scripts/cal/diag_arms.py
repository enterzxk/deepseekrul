"""Compare development vs test error scale for each arm."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import ROOT, load_dev, load_test  # noqa: E402

stage = Path(ROOT)
print("arm  subset  n_dev  dev_rmse  dev_sd  test_rmse  test_sd  ratio  dev_bias test_bias")
for arm in ("A", "BE", "PE"):
    for s in ("FD001", "FD002", "FD003", "FD004"):
        dev = load_dev(stage, s, arm)
        te = load_test(stage, s, arm)
        rd = np.sqrt(np.mean((dev["y"] - dev["yhat"]) ** 2))
        rt = np.sqrt(np.mean((te["y"] - te["yhat"]) ** 2))
        print("%-4s %-6s %5d %9.2f %7.2f %10.2f %8.2f %6.2f %9.2f %9.2f"
              % (arm, s, len(dev["y"]), rd, np.std(dev["y"] - dev["yhat"]),
                 rt, np.std(te["y"] - te["yhat"]), rt / max(rd, 1e-9),
                 np.mean(dev["y"] - dev["yhat"]), np.mean(te["y"] - te["yhat"])))
