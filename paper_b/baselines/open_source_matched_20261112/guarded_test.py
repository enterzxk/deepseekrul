"""Supported evaluation entrypoint; legacy sealed executables are preserved.

Added after the completed experiment as entrypoint hardening. It is not
retroactively included in the evaluation source identity of the actual runs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from unittest.mock import patch
from . import joint
from . import test_instrumented


def selected_record(original, extended, namespace, model, subset, seed, data_dir):
    master = joint.validate_ready(original, extended)
    records = [r for r in master["physical_runs"] if
               (r["condition"], r["model"], r["subset"], r["seed"]) ==
               (namespace, model, subset, seed)]
    assert len(records) == 1, "requested physical run is not jointly registered"
    record = records[0]
    expected_root = original if namespace == "fixed40" else extended
    assert Path(record["root"]).resolve() == Path(expected_root).resolve()
    assert Path(record["path"]).resolve() == joint.runpath(expected_root, (model, subset, seed)).resolve()
    assert Path(record["data_dir"]).resolve() == Path(data_dir).resolve()
    return record, joint.old.digest(Path(original)/"JOINT_COMPARISON_FREEZE.json")


def main():
    p = argparse.ArgumentParser()
    for name in ["root", "author-root", "data-dir", "model", "subset"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--namespace", choices=["fixed40", "author_limit"], required=True)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    record, master_sha = selected_record(a.root, a.author_root, a.namespace,
                                        a.model, a.subset, a.seed, a.data_dir)
    folder = Path(record["path"])
    if (folder/"COMPLETED.json").exists() or (folder/"TEST_REUSED.json").exists():
        joint.result_identity(record, master_sha)
        print("ALREADY_CLOSED_VALIDATED_NO_FORWARD")
        return
    assert not any((folder/n).exists() for n in
                   ["TEST_ATTEMPT.json", "TEST_CLAIM.json", "FAILED_TEST.json"]), "quota consumed"
    if a.audit_only:
        print("JOINT_READY_PHYSICAL_IDENTITY_VALIDATED_NO_FORWARD")
        return
    argv = ["sealed-test-instrumented", "--namespace", a.namespace,
            "--model", a.model, "--subset", a.subset, "--seed", str(a.seed),
            "--root", record["root"], "--data-dir", a.data_dir]
    with patch.object(sys, "argv", argv):
        test_instrumented.main()
    joint.result_identity(record, master_sha)


if __name__ == "__main__":
    main()
