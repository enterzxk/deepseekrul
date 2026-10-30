from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
if 'def _sha(' in s:
    print('already present'); raise SystemExit
anchor = 'def main() -> int:'
assert anchor in s
helper = '''def _sha(path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


'''
s = s.replace(anchor, helper + anchor, 1)
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('_sha inserted; syntax ok')
