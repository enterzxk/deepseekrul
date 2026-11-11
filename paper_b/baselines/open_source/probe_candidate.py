"""Exercise the sixth candidate's released pretraining tensor contract on real data."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import traceback
import torch
from .adapters import module
from .common import seed_everything, source_record, write_json
from .data import prepare


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed_everything(101)
    data = prepare("FD001", Path(a.data_dir), {"model": "two_phase", "window": 40})
    mod = module("cpt", "CMAPSS/model/transformer_new.py", "candidate_original_model")
    cfg = SimpleNamespace(embed_size=36, seq_len=40, n_heads=2, dropout=0.1, e_layers=2)
    net = mod.Model(cfg).cuda()
    inputs = torch.from_numpy(data["train"]["x"][:4]).cuda()
    record = {"model": "cpt", "source": source_record("cpt"), "data": "real FD001 training engines only",
              "input_shape": list(inputs.shape), "training_completed": False,
              "author_pretrain_callsite": "CMAPSS/run_cmapss.py:261-266"}
    try:
        # These are the two consecutive released pretraining calls, unchanged.
        representation = net(inputs)
        record["forward_shape"] = list(representation.shape)
        net.classifier(representation)
        record["status"] = "PARTIAL"
        record["reason"] = "interface check passed; candidate not in the five-model training priority"
    except RuntimeError:
        record["status"] = "FAILED"
        record["reason"] = "released pretraining interface passes scalar regression output to embedding classifier"
        record["error"] = traceback.format_exc()
    write_json(a.out, record)
    print(record["status"], record["reason"], flush=True)
