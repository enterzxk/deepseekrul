#!/usr/bin/env python3
"""Self-test for the protocol 6.8 gate, without reading any test label.

Checks that the runner's manifest gate accepts exactly the run it should accept
and refuses the ways a manifest could be too narrow. It calls official_split()
(unit ids only) and never loads Y_test.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

from paper_b.ncmapss.data import official_split              # noqa: E402
from paper_b.ncmapss.run_loeo import (SEEDS, _check_manifest,  # noqa: E402
                                      _freeze_hash_ok, _load_freeze)

H5 = 'data/NCMAPSS/extracted/N-CMAPSS_DS02-006.h5'
SHA = '47971a68b239ecb756833218a95d68ded6eb7e63ee84e86671c8b188de1ca765'
SEEDS_BY_RUNG = {'N0': (None,), 'N1': SEEDS, 'N2': SEEDS, 'N3': SEEDS}


def expect_fail(label, fn):
    try:
        fn()
    except SystemExit as exc:
        print(f'  REFUSED (good): {label}: {str(exc)[:90]}')
        return
    raise AssertionError(f'{label} was accepted but should have been refused')


def main():
    p, man, rule = _load_freeze(Path('.'), '')
    print(f'manifest {p.name}')
    print(f'  freeze_id={man["freeze_id"]}  hash_rule={rule}  cells={len(man["runs"])}')
    print(f'  anchor={man["repo"]["protocol_blob"]}  frozen={man["frozen"]}')
    assert _freeze_hash_ok(man) == rule

    dev, test = official_split(H5)
    print(f'  DS02 dev={dev} test={test}')

    print('positive: full four-rung invocation')
    _check_manifest(man, 'DS02', dev, test, SHA, ('N0', 'N1', 'N2', 'N3'), SEEDS_BY_RUNG)
    print('  accepted (good)')

    print('negative cases')
    expect_fail('only N0 requested while the manifest lists four rungs',
                lambda: _check_manifest(man, 'DS02', dev, test, SHA, ('N0',),
                                        SEEDS_BY_RUNG))
    expect_fail('wrong h5 sha256',
                lambda: _check_manifest(man, 'DS02', dev, test, '0' * 64,
                                        ('N0', 'N1', 'N2', 'N3'), SEEDS_BY_RUNG))
    expect_fail('dev units from another subset',
                lambda: _check_manifest(man, 'DS02', (1, 2, 3, 4, 5, 6), test, SHA,
                                        ('N0', 'N1', 'N2', 'N3'), SEEDS_BY_RUNG))
    expect_fail('test units from another subset',
                lambda: _check_manifest(man, 'DS02', dev, (7, 8, 9, 10), SHA,
                                        ('N0', 'N1', 'N2', 'N3'), SEEDS_BY_RUNG))

    # the frozen flag and the hash are checked in _load_freeze, so test that path
    # with a real file rather than asserting on a dict that never reaches it
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        unfrozen = dict(man, frozen=False)
        fp = Path(td) / 'unfrozen.json'
        fp.write_text(json.dumps(unfrozen), encoding='utf-8')
        expect_fail('unfrozen manifest (frozen:false)',
                    lambda: _load_freeze(Path(td), str(fp)))

        tampered = dict(man, scope=man['scope'] + ' (edited after signing)')
        tp = Path(td) / 'tampered.json'
        tp.write_text(json.dumps(tampered), encoding='utf-8')
        expect_fail('manifest edited after signing (hash no longer reproduces)',
                    lambda: _load_freeze(Path(td), str(tp)))
    print('\ngate self-test: PASS')


if __name__ == '__main__':
    main()
