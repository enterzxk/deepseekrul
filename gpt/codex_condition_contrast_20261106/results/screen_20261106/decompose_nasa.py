"""Recompute the NASA score contributions from overestimates and underestimates."""
from pathlib import Path
import json

import numpy as np

from tim_contrast.analyze import materialize, read_runs


def main():
    frames, _ = read_runs(Path("outputs"))
    out = {}
    for subset in ("FD002", "FD004"):
        out[subset] = {}
        for label in ("pe50", "pe100", "scheduled", "fixed_trend", "learned_trend"):
            p = materialize(frames, subset, label, 101, [0, 2])
            error = p.y_pred.to_numpy() - p.y_true.to_numpy()
            penalty = np.expm1(np.where(error > 0, error / 10.0, -error / 13.0))
            by_engine = p.assign(
                nasa_late=np.where(error > 0, penalty, 0.0),
                nasa_early=np.where(error < 0, penalty, 0.0),
                error=error,
            ).groupby("unit").agg(
                nasa_late=("nasa_late", "mean"), nasa_early=("nasa_early", "mean"),
                min_error=("error", "min"), max_error=("error", "max"),
            )
            by_engine["nasa_total"] = by_engine.nasa_late + by_engine.nasa_early
            out[subset][label] = {
                "mean_nasa_overestimates": by_engine.nasa_late.mean(),
                "mean_nasa_underestimates": by_engine.nasa_early.mean(),
                "top_engines": by_engine.nlargest(5, "nasa_total").reset_index().to_dict("records"),
            }
    print(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    main()
