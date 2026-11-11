"""Exercise final acceptance against real artifacts and isolated corrupt copies."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paper_b.baselines.open_source.verify_artifacts import verify


def replace(path, data):
    # Copies are hardlinked for speed: replace the directory entry, never
    # modify a shared inode from the real evidence directory.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = stream.name
    os.replace(temporary, path)


def refresh(root, relative):
    path = root / relative
    manifest = root / "ARTIFACT_MANIFEST.json"
    records = json.loads(manifest.read_text())
    item = next(x for x in records if x["path"] == relative)
    item.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    replace(manifest, json.dumps(records).encode())


def main(root):
    root = Path(root).resolve()
    before = hashlib.sha256((root / "ARTIFACT_MANIFEST.json").read_bytes()).hexdigest()
    results = []
    with contextlib.redirect_stdout(io.StringIO()):
        verify(root, require_complete=True)
    results.append({"case": "real_complete_120", "result": "PASS"})
    for case in ["missing_registered_run", "row_checkpoint_hash", "stale_summary_mean"]:
        with tempfile.TemporaryDirectory(prefix="paper_b_acceptance_") as temp:
            fixture = Path(temp) / "evidence"
            shutil.copytree(root, fixture, copy_function=os.link)
            if case == "stale_summary_mean":
                relative = "summary.json"
                groups = json.loads((fixture / relative).read_text())
                groups[0]["rmse_mean"] += 0.5
                replace(fixture / relative, json.dumps(groups).encode())
            else:
                relative = "runs.csv"
                rows = list(csv.DictReader((fixture / relative).open(newline="")))
                fields = list(rows[0])
                if case == "missing_registered_run":
                    rows.pop()
                else:
                    rows[0]["checkpoint_sha256"] = "0" * 64
                content = io.StringIO(newline="")
                writer = csv.DictWriter(content, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                replace(fixture / relative, content.getvalue().encode())
            refresh(fixture, relative)
            try:
                verify(fixture, require_complete=True)
            except AssertionError as error:
                results.append({"case": case, "result": "REJECTED", "assertion": str(error).replace(str(fixture), "<isolated_fixture>")})
            else:
                raise AssertionError(f"corrupted evidence accepted: {case}")
    assert before == hashlib.sha256((root / "ARTIFACT_MANIFEST.json").read_bytes()).hexdigest()
    return {"official_test_forward_executed": False, "real_artifact_manifest_unchanged": True, "checks": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.root), ensure_ascii=False, indent=2))
