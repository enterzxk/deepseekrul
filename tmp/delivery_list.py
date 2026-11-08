from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
print('=== top-level files ===')
for p in sorted(HERE.glob('*')):
    if p.is_file():
        print('%-40s %8d'%(p.name,p.stat().st_size))
print('=== dirs ===')
for p in sorted(HERE.iterdir()):
    if p.is_dir():
        n=sum(1 for _ in p.rglob('*') if _.is_file())
        print('%-20s %d files'%(p.name,n))
