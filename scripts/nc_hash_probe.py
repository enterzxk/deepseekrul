#!/usr/bin/env python3
"""Find which canonicalisation reproduces a manifest's declared content hash.

The gate has to know the rule rather than assume it. This prints, for every
plausible combination of exclusion set / ascii mode / separators, whether it
reproduces the declared hash, so the rule can be written down instead of guessed.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

DROPS = {
    'hash+eval': ('freeze_content_sha256', 'eval_start_utc'),
    'hash+eval+ts': ('freeze_content_sha256', 'eval_start_utc', 'freeze_ts_utc'),
    'hash_only': ('freeze_content_sha256',),
    'hash+eval+ts+scope': ('freeze_content_sha256', 'eval_start_utc',
                           'freeze_ts_utc', 'scope'),
}


def main():
    p = Path(sys.argv[1])
    raw = p.read_text(encoding='utf-8')
    man = json.loads(raw)
    declared = man.get('freeze_content_sha256')
    print(f'{p.name}: declared={declared}')
    hits = []
    for dname, drop in DROPS.items():
        body = {k: v for k, v in man.items() if k not in drop}
        for ascii_, seps in itertools.product(
                (True, False), (None, (',', ':'))):
            blob = json.dumps(body, sort_keys=True, ensure_ascii=ascii_,
                              **({'separators': seps} if seps else {}))
            h = hashlib.sha256(blob.encode('utf-8')).hexdigest()
            if h == declared:
                hits.append((dname, ascii_, seps))
                print(f'  MATCH drop={dname} ensure_ascii={ascii_} '
                      f'separators={seps}')
    if not hits:
        print('  no match under any tried rule')
        body = {k: v for k, v in man.items()
                if k not in ('freeze_content_sha256', 'eval_start_utc')}
        print('  keys:', sorted(man.keys()))
        print('  sample canonical:', json.dumps(body, sort_keys=True)[:300])


if __name__ == '__main__':
    main()
