from pathlib import Path
p=Path('build_results.py'); s=p.read_text()
if 'second pass' in s:
    print('already'); raise SystemExit
old = """                    if ref_arm in group["arms"] and (ref_arm, seed) in tables:
                        c = paired(t, tables[(ref_arm, seed)])
                        row.update(delta_rmse=c["delta"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                                   gain=1.0 - m["rmse"] / pooled(tables[(ref_arm, seed)])["rmse"])
                    rows.append(row)"""
new = """                    rows.append(row)"""
assert old in s
s = s.replace(old, new)
old2 = """            detail[f"{protocol}|{subset}"] = {f"{a}|{s}": pooled(t) for (a, s), t in tables.items()}"""
new2 = """            # second pass: contrasts against the locked reference
            for row in rows:
                if row["protocol"] != protocol or row["subset"] != subset:
                    continue
                key = (row["arm"], row["seed"])
                if key in tables and (ref_arm, row["seed"]) in tables:
                    c = paired(tables[key], tables[(ref_arm, row["seed"])])
                    row.update(delta_rmse=c["delta"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                               gain=1.0 - row["rmse"] / pooled(tables[(ref_arm, row["seed"])])["rmse"])
            detail[f"{protocol}|{subset}"] = {f"{a}|{s}": pooled(t) for (a, s), t in tables.items()}"""
assert old2 in s
s = s.replace(old2, new2)
p.write_text(s); print('patched')
