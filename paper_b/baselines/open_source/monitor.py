"""Compact read-only progress snapshot for the current non-recurring batch."""
import argparse
import json
from pathlib import Path
import re

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    for version in sorted(root.glob("v*")):
        for protocol in ["author", "unified"]:
            folder = version / protocol
            if not folder.exists():
                continue
            for model in sorted(folder.iterdir()):
                completed = list(model.glob("FD*/s*/metrics.json"))
                running = list(model.glob("FD*/s*/RUNNING.json"))
                failed = list(model.glob("FD*/s*/FAILED.json"))
                print(version.name, protocol, model.name, "done", len(completed), "running",
                      [str(p.parent.relative_to(model)) for p in running], "failed", len(failed))
                for p in running:
                    subset, seed = p.parent.parent.name, p.parent.name[1:]
                    log = version / "logs" / f"{protocol}_{model.name}_{subset}_{seed}.log"
                    if log.exists():
                        content = log.read_text(errors="replace")[-15000:]
                        lines = [s for s in re.split(r"[\r\n]+", content) if re.match(r"(TRAIN |PRETRAIN |epoch |Epoch:|Epoch \d|.*training loss:)", s)]
                        if lines:
                            print(" ", lines[-1][:210])
