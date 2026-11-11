"""Evidence, exclusive run claims and deterministic experiment utilities."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def research_task_start(out, cfg):
    """O_EXCL run-ID claim: reject already-running/completed tasks, never overwrite."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"completed task: {out}")
    claim = {"pid": os.getpid(), "host": socket.gethostname(), "started_unix": time.time(),
             "config_sha256": config_hash(cfg), "task_id": str(out)}
    with (out / "RUNNING.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2)
    write_json(out / "config.json", {**cfg, "config_sha256": config_hash(cfg)})
    return claim


def source_record(key):
    records = json.loads((ROOT / "sources.lock.json").read_text())
    return next(x for x in records if x["model"] == key)


def predictions(path, truth, pred, unit=None, cut=None):
    import numpy as np
    import pandas as pd
    truth, pred = np.asarray(truth).reshape(-1), np.asarray(pred).reshape(-1)
    assert len(truth) == len(pred) and np.isfinite(truth).all() and np.isfinite(pred).all()
    frame = pd.DataFrame({"unit": np.arange(1, len(pred) + 1) if unit is None else unit,
                          "y_true": truth, "y_pred": pred, "signed_error": pred - truth})
    if cut is not None:
        frame["cut"] = cut
    frame.to_csv(path, index=False, float_format="%.10g")
    return frame


def metrics(truth, pred):
    from paper_b.training.metrics import summarize
    import numpy as np
    result = summarize(np.asarray(pred).reshape(-1), np.asarray(truth).reshape(-1))
    return {**result, "n": int(np.asarray(truth).size)}
