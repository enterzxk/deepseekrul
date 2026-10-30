import glob, os, time, json
pats = ['results/**/*.json','results/**/*.csv','exports/**/*','docs/**/*.json']
hits = []
for p in pats:
    for f in glob.glob(p, recursive=True):
        if not os.path.isfile(f): continue
        if any(k in f for k in ('fd002','robust','p3_locked','uncertainty','test')):
            hits.append((f, time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(os.stat(f).st_mtime))))
print('candidate artifacts from test-file scripts:', len(hits))
for f, t in sorted(hits, key=lambda x: x[1])[:20]: print(f'  {t}  {f}')
print()
import subprocess
print('--- git log of those four scripts (last 3 each) ---')
for s in ('scripts/run_fd002.py','scripts/eval_robustness_frozen.py','scripts/run_p3_locked.py','scripts/eval_uncertainty_frozen.py'):
    try:
        out = subprocess.run(['git','log','--format=%h %ad %s','--date=short','-3','--',s], capture_output=True, text=True, timeout=20).stdout.strip()
        print(f'  {s}\n    ' + (out.replace(chr(10), chr(10)+'    ') or '(no commits)'))
    except Exception as e:
        print('  ', s, 'git error', e)
