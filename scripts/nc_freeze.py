#!/usr/bin/env python
"""Sign and verify N-CMAPSS freeze manifests.

Two subcommands:

  sign    write freeze/<freeze_id>.json from the sealed dev ledgers, with
          freeze_ts_utc INSIDE the hash (rule v17) and epoch* values taken
          mechanically from docs/paper_b_ncmapss/registry/runs_DS0*.csv
  verify  re-check a committed manifest: hash, completeness (40/40 cross
          product, zero exclusions), epoch* values against the ledgers, and
          the freeze/eval ordering.

Hash rule v17 (changed from v16):
    freeze_content_sha256 = sha256( canonical JSON of the manifest with
    ONLY `freeze_content_sha256` and `eval_start_utc` removed )
    canonical: sort_keys=True, separators=(',', ':'), ensure_ascii=False, UTF-8

Why v17: under v16 `freeze_ts_utc` sat outside the hash, so the one field that
proves "declared before opened" could be edited to any earlier instant without
breaking the digest. `eval_start_utc` stays outside because it is written after
unsealing; the digest itself is outside to avoid self-reference.

Usage:
    python scripts/nc_freeze.py sign --base <previous manifest> [--drop key] [--extra @fields.json]
    python scripts/nc_freeze.py verify docs/paper_b_ncmapss/registry/freeze/NCM-FREEZE-YYYYMMDDTHHMMZ.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone

REG = os.path.join("docs", "paper_b_ncmapss", "registry")
SUBSETS = ["DS02", "DS05", "DS06", "DS07"]
RUNGS = ["N0", "N1", "N2", "N3"]
SEEDS = [101, 202, 303]
N0_SEEDS = [None]
OUTSIDE_HASH = ("freeze_content_sha256", "eval_start_utc")


def content_hash(d: dict) -> str:
    body = {k: v for k, v in d.items() if k not in OUTSIDE_HASH}
    s = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def ledger_path(subset: str, round_tag: str = "") -> str:
    """Dev ledger for a subset. round_tag is "" for the first batch, "_r2" for the rerun."""
    return os.path.join(REG, "runs_%s%s.csv" % (subset, round_tag))


def expected_cells(rungs=None, seeds=None) -> list[tuple]:
    """The full cross product. Driven by the manifest's own rung list, so a
    five-rung manifest (N0/N1/N2/N2s/N3) verifies without touching this file."""
    rungs = rungs or RUNGS
    seeds = seeds or SEEDS
    cells = []
    for s in SUBSETS:
        for r in rungs:
            for sd in (N0_SEEDS if r == "N0" else seeds):
                cells.append((s, r, sd))
    return cells


def epoch_table(round_tag: str = "", rungs=None, seeds=None) -> dict:
    """epoch* per cell = round_half_up(mean of the six LOEO epoch* values).

    Reads the dev ledger for the round. Asserts the seed gate on the way out:
    N0 exactly one seed per subset, every other rung exactly the declared set.
    """
    out = {}
    for s in SUBSETS:
        path = ledger_path(s, round_tag)
        rows = [r for r in csv.DictReader(io.open(path, encoding="utf-8"))
                if r["selection_rule"] == "inner_val"]
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["rung"], r["seed"])].append(int(r["epochs_run"]))
        for (rung, seed), vals in by.items():
            assert len(vals) == 6, "%s %s s%s: %d folds, expected 6" % (s, rung, seed, len(vals))
            vals = sorted(vals)
            out[(s, rung, (int(seed) if seed else None))] = (round_half_up(sum(vals) / 6.0), vals)
    want = set(expected_cells(rungs, seeds))
    got = set(out)
    missing = sorted(want - got)
    extra = sorted(got - want)
    assert not missing, "epoch* table is missing cells: %s" % missing[:8]
    assert not extra, "epoch* table has cells outside the grid: %s" % extra[:8]
    return out


def floor_stats(round_tag: str = "") -> dict:
    """Count floor-of-20 hits, per rung and overall, from the sealed ledgers."""
    per_rung = collections.Counter()
    per_rung_total = collections.Counter()
    n0_by_subset = {}
    for s in SUBSETS:
        path = ledger_path(s, round_tag)
        hit = tot = 0
        for r in csv.DictReader(io.open(path, encoding="utf-8")):
            if r["selection_rule"] != "inner_val":
                continue
            per_rung_total[r["rung"]] += 1
            if int(r["epochs_run"]) == 20:
                per_rung[r["rung"]] += 1
            if r["rung"] == "N0":
                tot += 1
                if int(r["epochs_run"]) == 20:
                    hit += 1
        n0_by_subset[s] = {"hits": hit, "folds": tot}
    grouped = " + ".join("%s %d/%d" % (s, n0_by_subset[s]["hits"], n0_by_subset[s]["folds"])
                         for s in SUBSETS)
    return {
        "floor_value": 20,
        "hits": {k: per_rung[k] for k in RUNGS},
        "folds": {k: per_rung_total[k] for k in RUNGS},
        "n0_by_subset": n0_by_subset,
        "n0_grouped": grouped,
        "n0_note": "N0: %d of %d folds sit on the floor (%s)"
                   % (per_rung["N0"], per_rung_total["N0"], grouped),
    }


def cmd_sign(args) -> int:
    base = json.load(io.open(args.base, encoding="utf-8")) if args.base else {}
    now = datetime.now(timezone.utc)
    fid = "NCM-FREEZE-%sZ" % now.strftime("%Y%m%dT%H%M")

    rungs = args.rungs.split(",") if args.rungs else (base.get("rungs") or RUNGS)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else (base.get("seeds") or SEEDS)
    ep = epoch_table(args.round or "", rungs, seeds)
    fs = floor_stats(args.round or "")

    m = dict(base)
    m["freeze_id"] = fid
    m["dataset"] = "N-CMAPSS"
    m["frozen"] = True
    m["freeze_ts_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    m["hash_rule_version"] = "v17: excludes eval_start_utc and freeze_content_sha256 only; freeze_ts_utc IS hashed"
    m["eval_start_utc"] = None

    m["rungs"] = rungs
    m["seeds"] = seeds
    m["runs"] = [{"subset": s, "rung": r, "seed": sd} for (s, r, sd) in expected_cells(rungs, seeds)]
    m["dev_ledger_pattern"] = "docs/paper_b_ncmapss/registry/runs_<subset>%s.csv" % (args.round or "")
    m["supersedes"] = args.supersedes
    m["sequencing"] = args.sequencing
    m["completeness"] = {
        "assertion": "runs == {DS02,DS05,DS06,DS07} x {%s} x ({N0:[null]} U {%s:[%s]})"
                     % (",".join(rungs), ",".join(r for r in rungs if r != "N0"),
                        ",".join(str(x) for x in seeds)),
        "expected_cells": len(expected_cells(rungs, seeds)),
        "observed_cells": len(m["runs"]),
        "excluded_cells": 0,
        "exclusion_list": [],
        "why_it_matters": "a complete cross product cannot express 'report only the favourable cells'; the gate exists to remove that choice, and this field is what proves the choice was removed",
        "verify_command": "python scripts/nc_freeze.py verify <this file>",
    }
    m["epoch_star_values"] = [
        {
            "subset": s,
            "rung": r,
            "seed": sd,
            "epoch_star": ep[(s, r, sd)][0],
            "fold_values": ep[(s, r, sd)][1],
            "rule": "round_half_up(mean(fold_values))",
        }
        for (s, r, sd) in expected_cells(rungs, seeds)
    ]
    m["floor_statistics"] = fs
    m["test_design"] = dict(m["test_design"])
    m["test_design"]["n0_floor_note"] = (
        "N0 epoch* is a floor hit, not a selection: %s. The estimated value still "
        "differs per subset (DS02 20 / DS06 21 / DS07 27 / DS05 32), so 'N0 is at the "
        "floor' holds at fold level, not as 'N0 epoch* == 20'. Written with the grouped "
        "detail inline because a bare aggregate cannot be checked without redoing the "
        "arithmetic." % fs["n0_note"]
    )
    m["test_design"]["epoch_star_source"] = (
        "epoch_star_values[] in this manifest, recomputed from the sealed dev ledgers "
        "at sign time; the runner must use these numbers, not recompute them"
    )
    m["test_design"]["floor_cell_warning"] = (
        "DS07 N1 s101 and s303 have all six folds at 20: their inner_val column is a "
        "20-pass model, not a converged point. Must be labelled as such in the test table."
    )
    m.pop("freeze_content_sha256", None)
    for k in (args.drop or "").split(","):
        if k.strip():
            m.pop(k.strip(), None)
    if getattr(args, "extra", None):
        m.update(args.extra)
    m["freeze_content_sha256"] = content_hash(m)

    os.makedirs(args.out_dir, exist_ok=True)
    args.out = os.path.join(args.out_dir, "%s.json" % fid)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    print("wrote %s" % args.out)
    print("freeze_id            %s" % m["freeze_id"])
    print("freeze_ts_utc        %s" % m["freeze_ts_utc"])
    print("freeze_content_sha256 %s" % m["freeze_content_sha256"])
    print("cells                %d" % len(m["runs"]))
    return 0


def _opens_records(manifest_path: str, d: dict) -> list:
    """Find eval_start records for this manifest: sibling file and opens/ snapshots."""
    fid = d.get("freeze_id")
    base = os.path.dirname(manifest_path)
    out = []
    p = os.path.join(base, "%s.eval_start.json" % fid)
    if os.path.exists(p):
        out.append(p)
    od = os.path.join(base, "opens")
    if os.path.isdir(od):
        for n in sorted(os.listdir(od)):
            if n.endswith(".json") and fid in n:
                out.append(os.path.join(od, n))
    return out


def _check_opens(manifest_path: str, d: dict) -> list:
    """Verify every unsealing record against the manifest. Returns a list of failures."""
    fails = []
    recs = _opens_records(manifest_path, d)
    if not recs:
        print("ordering        not yet unsealed (no eval_start record found)")
        return fails
    blob = d.get("repo", {}).get("protocol_blob")
    digest = d.get("freeze_content_sha256")
    fts = d.get("freeze_ts_utc")
    for path in recs:
        r = json.load(io.open(path, encoding="utf-8"))
        tag = os.path.basename(path)
        if r.get("freeze_content_sha256") != digest:
            fails.append("%s: freeze_content_sha256 does not match this manifest" % tag)
        if r.get("freeze_ts_utc") != fts:
            fails.append("%s: freeze_ts_utc %s != manifest %s" % (tag, r.get("freeze_ts_utc"), fts))
        for op in r.get("opens", []):
            s = op.get("subset")
            t = op.get("eval_start_utc")
            ok_order = bool(t and fts and t > fts)
            ok_blob = op.get("protocol_blob") == blob
            want_sha = d.get("data", {}).get(s, {}).get("sha256")
            ok_sha = (want_sha is None) or (op.get("data_sha256") == want_sha)
            want_test = d.get("data", {}).get(s, {}).get("test_units")
            ok_units = (want_test is None) or (op.get("test_units") == want_test)
            delta = ""
            if t and fts:
                from datetime import datetime as _dt
                fmt = "%Y-%m-%dT%H:%M:%SZ"
                delta = " +%ds" % int((_dt.strptime(t, fmt) - _dt.strptime(fts, fmt)).total_seconds())
            print("ordering        %s  %s opens %s at %s%s (blob %s, data %s, units %s)"
                  % ("OK" if (ok_order and ok_blob and ok_sha and ok_units) else "FAIL",
                     tag, s, t, delta,
                     "ok" if ok_blob else "MISMATCH",
                     "ok" if ok_sha else "MISMATCH",
                     "ok" if ok_units else "MISMATCH"))
            if not ok_order:
                fails.append("%s/%s: eval_start_utc %s is not after freeze_ts_utc %s" % (tag, s, t, fts))
            if not ok_blob:
                fails.append("%s/%s: protocol_blob %s != manifest %s" % (tag, s, op.get("protocol_blob"), blob))
            if not ok_sha:
                fails.append("%s/%s: data_sha256 mismatch" % (tag, s))
            if not ok_units:
                fails.append("%s/%s: test_units mismatch" % (tag, s))
    return fails


def cmd_verify(args) -> int:
    d = json.load(io.open(args.manifest, encoding="utf-8"))
    fail = []

    h = content_hash(d)
    if h != d.get("freeze_content_sha256"):
        fail.append("hash mismatch: stored %s, recomputed %s" % (d.get("freeze_content_sha256"), h))
    print("hash            %s  %s" % ("OK" if h == d.get("freeze_content_sha256") else "FAIL", h))

    v_rungs = d.get("rungs") or RUNGS
    v_seeds = d.get("seeds") or SEEDS
    v_round = ""
    pat = d.get("dev_ledger_pattern")
    if pat and "runs_<subset>" in pat:
        v_round = pat.split("runs_<subset>", 1)[1].split(".csv", 1)[0]
    got = [(r["subset"], r["rung"], r["seed"]) for r in d.get("runs", [])]
    want = expected_cells(v_rungs, v_seeds)
    if len(got) != len(set(got)):
        fail.append("duplicate cells in runs[]")
    if set(got) != set(want):
        fail.append("runs[] is not the %d-cell cross product: missing %s extra %s"
                    % (len(want), sorted(set(want) - set(got)), sorted(set(got) - set(want))))
    if d.get("completeness", {}).get("excluded_cells", 1) != 0:
        fail.append("completeness.excluded_cells is not 0")
    print("completeness    %s  %d/%d cells, %d exclusions"
          % ("OK" if set(got) == set(want) and len(got) == len(want) else "FAIL",
             len(got), len(want), d.get("completeness", {}).get("excluded_cells", -1)))

    ep = epoch_table(v_round, v_rungs, v_seeds)
    bad = []
    for e in d.get("epoch_star_values", []):
        k = (e["subset"], e["rung"], e["seed"])
        if k not in ep:
            bad.append("%s: cell not in ledgers" % (k,))
            continue
        if e["epoch_star"] != ep[k][0] or e["fold_values"] != ep[k][1]:
            bad.append("%s: manifest %s/%s, ledger %s/%s"
                       % (k, e["epoch_star"], e["fold_values"], ep[k][0], ep[k][1]))
    if len(d.get("epoch_star_values", [])) != len(want):
        bad.append("epoch_star_values has %d entries, expected %d"
                   % (len(d.get("epoch_star_values", [])), len(want)))
    if bad:
        fail.extend(bad)
    print("epoch_star      %s  %d values recomputed from the dev ledgers"
          % ("OK" if not bad else "FAIL", len(want)))
    for b in bad:
        print("   %s" % b)

    es, fs_ = d.get("eval_start_utc"), d.get("freeze_ts_utc")
    if es:
        ok = es > fs_
        print("ordering        %s  eval_start %s vs freeze %s" % ("OK" if ok else "FAIL", es, fs_))
        if not ok:
            fail.append("eval_start_utc <= freeze_ts_utc")
    else:
        fail.extend(_check_opens(args.manifest, d))

    if d.get("frozen") is not True:
        fail.append("frozen is not true")

    if fail:
        print("\nFAILED (%d)" % len(fail))
        for f in fail:
            print("  - %s" % f)
        return 1
    print("\nOK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sign")
    s.add_argument("--base", default=None, help="existing manifest to reuse as the base body")
    s.add_argument("--out-dir", default=os.path.join(REG, "freeze"), help="directory for <freeze_id>.json")
    s.add_argument("--supersedes", default=None, help="JSON object describing the superseded manifest")
    s.add_argument("--sequencing", default=None, help="JSON object describing the freeze/eval ordering disclosure")
    s.add_argument("--extra", default=None, help="JSON object merged into the manifest before hashing")
    s.add_argument("--drop", default=None, help="comma-separated top-level keys to remove before hashing")
    s.add_argument("--round", default="", help="dev ledger suffix: '' for batch 1, '_r2' for the rerun")
    s.add_argument("--rungs", default=None, help="comma-separated rung list; default from --base")
    s.add_argument("--seeds", default=None, help="comma-separated seed list; default from --base")
    s.set_defaults(func=cmd_sign)
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    v.set_defaults(func=cmd_verify)
    a = ap.parse_args(argv)
    for name in ("supersedes", "sequencing", "extra"):
        val = getattr(a, name, None)
        if isinstance(val, str):
            if val.startswith("@"):
                setattr(a, name, json.load(io.open(val[1:], encoding="utf-8")))
            else:
                setattr(a, name, json.loads(val))
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
