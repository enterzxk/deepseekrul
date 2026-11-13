"""Apply the M3C rank-diagnostic hunks to a train_d.py that has diverged.

Why a patch instead of copying the file: the 3090 copy is reported at 1153 lines
while the repository is 1070 at the same commit, so it carries local edits the
repository does not. A whole-file write would drop them silently -- the failure
that already lost the M2 registration twice. This applies only the four hunks
that put the ranking term into the D4.5 gradient probe.

Each hunk is recognised by a short MARKER rather than by its full replacement
text, because the same edit may be wrapped differently in a file that was
patched by hand. Marker present -> skip (idempotent). Marker absent but the old
text present -> apply. Neither -> refuse and write nothing, so a diverged file is
never half-patched.

Usage
  python docs/paper_b_hc/probes/patch_m3c_rank_diag.py [path/to/train_d.py]
  # default: paper_b/hc/train_d.py relative to the repo root
"""
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# (name, marker, old, new)
HUNKS = [
    (
        'term_grads signature',
        'rank_cfg=None):',
        "def term_grads(net, arm, ebt, dev, K, cap, s0, lam_t, lam_s, n_eng=3,\n"
        "               scr=False, lam_a=0.0, lam_dpc=0.0):",
        "def term_grads(net, arm, ebt, dev, K, cap, s0, lam_t, lam_s, n_eng=3,\n"
        "               scr=False, lam_a=0.0, lam_dpc=0.0, rank_cfg=None):",
    ),
    (
        'collect (D, r) from the reference engines',
        # NOT "_fos.append(...)": that string is a substring of the training
        # loop's "batch_fos.append(...)", so the marker would report the hunk as
        # applied on a file that never got it -- the patch would then compile and
        # crash at runtime on NameError. Anchor on the assignment instead.
        "\n    _fos = []\n",
        "    params = [p for p in net.parameters() if p.requires_grad]\n"
        "    acc = {}\n"
        "    for i in range(min(n_eng, len(ebt['lengths']))):\n"
        "        n = int(ebt['lengths'][i])\n"
        "        fo = forward_engine(net, arm, ebt, i, n, dev, K)\n",
        "    params = [p for p in net.parameters() if p.requires_grad]\n"
        "    acc = {}\n"
        "    _fos = []\n"
        "    for i in range(min(n_eng, len(ebt['lengths']))):\n"
        "        n = int(ebt['lengths'][i])\n"
        "        fo = forward_engine(net, arm, ebt, i, n, dev, K)\n"
        "        if rank_cfg is not None:\n"
        "            # M3C: the ranking term needs pairs, so it is built from the\n"
        "            # same reference engines the other terms use, not a private\n"
        "            # batch.\n"
        "            _fos.append(dict(D=fo['D'], r=fo['r']))\n",
    ),
    (
        'report L_rank with the optimiser weight',
        "w['rank'] =",
        "    w = {'rul': 1.0, 'terminal': lam_t, 'smooth': lam_s, 'alive': lam_a,\n"
        "         'dpc': lam_dpc}\n",
        "    w = {'rul': 1.0, 'terminal': lam_t, 'smooth': lam_s, 'alive': lam_a,\n"
        "         'dpc': lam_dpc}\n"
        "    if rank_cfg is not None and len(_fos) > 1:\n"
        "        _rs, _rn = rank_pair_loss(_fos, rank_cfg['margin'], rank_cfg['gap'])\n"
        "        if _rs is not None:\n"
        "            # Same weight the optimiser applies to L_rank, so\n"
        "            # g_ratio['rank'] is the gradient share and can be compared\n"
        "            # with lam_rank directly.\n"
        "            L['rank'] = _rs / float(_rn)\n"
        "            w['rank'] = (rank_cfg['lam_rank'] * rank_cfg['rul0']\n"
        "                         / rank_cfg['rank0'])\n",
    ),
    (
        'pass rank_cfg from the training loop',
        'rank_cfg=(dict(margin=rank_margin',
        "                         lam_dpc=((lam_dpc or 0.0) * (rul0 or 1.0) / (dpc0 or 1.0)))",
        "                         lam_dpc=((lam_dpc or 0.0) * (rul0 or 1.0) / (dpc0 or 1.0)),\n"
        "                         rank_cfg=(dict(margin=rank_margin, gap=rank_gap,\n"
        "                                        lam_rank=lam_rank, rank0=rank0, rul0=rul0)\n"
        "                                   if arm in RANK_ARMS else None))",
    ),
]


def main(argv):
    p = Path(argv[1]) if len(argv) > 1 else ROOT / 'paper_b' / 'hc' / 'train_d.py'
    if not p.exists():
        print('FAIL no such file: %s' % p)
        return 2
    src = p.read_text(encoding='utf-8')
    applied, skipped, missing = [], [], []
    for name, marker, old, new in HUNKS:
        if marker in src:
            skipped.append(name)
        elif old in src:
            src = src.replace(old, new, 1)
            applied.append(name)
        else:
            missing.append(name)
    if missing:
        print('REFUSED: no marker and no anchor for %d hunk(s): %s'
              % (len(missing), missing))
        print('the file has diverged in those regions; nothing was written')
        return 1
    if applied:
        p.write_text(src, encoding='utf-8')
    print('applied=%d skipped(already present)=%d' % (len(applied), len(skipped)))
    for n in applied:
        print('  + %s' % n)
    for n in skipped:
        print('  = %s (already applied)' % n)
    try:
        py_compile.compile(str(p), doraise=True)
        print('py_compile OK')
    except py_compile.PyCompileError as e:
        print('FAIL py_compile: %s' % e)
        return 1
    ok = (src.count('rank_cfg') >= 6 and "w['rank']" in src
          and "\n    _fos = []\n" in src
          and "_fos.append(dict(D=fo['D'], r=fo['r']))" in src)
    print("rank_cfg occurrences=%d  w['rank'] present=%s  _fos defined=%s"
          % (src.count('rank_cfg'), "w['rank']" in src, "\n    _fos = []\n" in src))
    print('PATCH %s' % ('OK' if ok else 'INCOMPLETE'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
