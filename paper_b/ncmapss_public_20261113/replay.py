"""Replay every sealed full-author AIDGN checkpoint; no optimizer or TEST.

Run directly next to collect.py, outside the immutable training runtime.
Inner and outer predictions remain separate; --limit is only a partial audit.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


def main():
    spec = importlib.util.spec_from_file_location("_aidgn_evidence_collector", Path(__file__).with_name("collect.py"))
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    parser = collector.parser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--phase", choices=("all", "inner", "outer"), default="all")
    args = parser.parse_args()
    collector.require(not args.output.exists(), "replay output exists; refusing overwrite")
    collection, context = collector.audit(args.runtime, args.cache, args.runs, args.author_file)
    collector.require(collection["status"] != "FAIL", "sealed training audit failed")
    if args.require_complete:
        collector.require(collection["full_gate"], "full 192-cell evidence is required")
    output = collector.new_output(args.output, (args.runtime, args.cache, args.runs, args.author_file.parent.parent))
    collector.write_result(output, collection)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    runner, protocol = context["runner"], context["protocol"]
    chosen = [(cell, item) for cell, item in sorted(context["verified"].items())
              if args.phase == "all" or item["config"]["phase"] == args.phase]
    if args.limit:
        collector.require(args.limit > 0, "limit must be positive")
        chosen = chosen[:args.limit]
    started = time.monotonic()
    rows = []
    for cell, item in chosen:
        config = item["config"]
        folder = Path(item["folder"])
        checkpoint = torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=True)
        model = runner.AIDGNAdapter(args.author_file, history=config["history"], points=config["points_per_flight"]).to(args.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        collector.require(runner.state_hash(model) == checkpoint["final_state_sha256"], "loaded tensor digest differs")
        model.eval()
        data = context["data"][config["subset"]][0]
        views = ("clean",) if config["phase"] == "inner" else collector.VIEWS
        for view in views:
            heldout = runner.bundle(data, [config["validation_unit"]], checkpoint["scalers"],
                                    {"span": 70, "prior_lag_flights": 5, "support_radius": .75, "label_cap": 125.},
                                    args.device, view)
            predictions = []
            with torch.no_grad():
                for lo in range(0, len(heldout["y"]), protocol["batch"]):
                    query = torch.arange(lo, min(lo+protocol["batch"], len(heldout["y"])), device=args.device)
                    x, age = runner.pack_window(heldout, query, config["history"], config["points_per_flight"])
                    predictions.append(model(x, age).cpu().numpy())
            actual = np.concatenate(predictions).astype(np.float64)*125.
            saved = item["predictions"][view]["unbounded"]
            collector.require(actual.shape == saved.shape and bool(np.isfinite(actual).all()), "nonfinite or mismatched replay vector")
            difference = float(np.max(np.abs(actual-saved)))
            collector.require(difference <= 1e-5, f"checkpoint prediction mismatch: {cell}/{view}: {difference}")
            rows.append({"cell": cell, "phase": config["phase"], "view": view,
                         "bitwise_equal": bool(np.array_equal(actual, saved)),
                         "maximum_absolute_difference_cycles": difference,
                         "checkpoint_sha256": item["files_sha256"]["checkpoint.pt"]})
        print(json.dumps({"cell": cell, "views_replayed": len(views)}), flush=True)
        del model, checkpoint, heldout
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    expected = {"all": 192, "inner": 120, "outer": 72}[args.phase]
    full = collection["full_gate"] and len(chosen) == expected
    report = {"status": "PASS" if full else "PARTIAL", "phase": args.phase,
              "checkpoints_replayed": len(chosen), "expected_checkpoints_for_phase": expected,
              "view_vectors_replayed": len(rows), "rows": rows,
              "maximum_absolute_difference_cycles": max((r["maximum_absolute_difference_cycles"] for r in rows), default=None),
              "all_bitwise_equal": bool(rows) and all(r["bitwise_equal"] for r in rows),
              "tolerance_cycles": 1e-5, "execution_sha256": collector.EXECUTION,
              "source_commit": collector.COMMIT, "replay_source_sha256": collector.digest(__file__),
              "collector_source_sha256": collector.digest(Path(__file__).with_name("collect.py")),
              "device": args.device, "seconds": time.monotonic()-started,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "optimizer_updates": 0,
              "official_test_read": False, "interpretation": "prediction integrity; no method-benefit or native-protocol claim"}
    with (output/"REPLAY_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    collector.write_csv(output/"replay_vectors.csv", rows)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    return 0 if full or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
