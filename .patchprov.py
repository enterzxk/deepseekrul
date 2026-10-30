from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
if 'code_sha256' in s:
    print('already records provenance')
    raise SystemExit
old = '    }, indent=2))'
new = '''        "provenance": {
            "code_sha256": {
                "runner": _sha(Path(__file__)),
                "model": _sha(ROOT / "src" / "candidates" / "model.py"),
                "dataset": _sha(ROOT / "src" / "dataset.py"),
            },
            "command": " ".join(["python"] + sys.argv),
            "argv": list(sys.argv),
        },
    }, indent=2))'''
assert old in s, 'anchor missing'
s = s.replace(old, new, 1)
helper = '''def _sha(path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def main():'''
s = s.replace('def main():', helper, 1)
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('provenance added; syntax ok')
