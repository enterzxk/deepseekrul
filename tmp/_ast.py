
import ast, os, json
roots = ["src"]
out = []
for root in roots:
    for dp, dn, fn in os.walk(root):
        if "__pycache__" in dp: continue
        for f in sorted(fn):
            if not f.endswith(".py"): continue
            p = os.path.join(dp, f)
            try:
                tree = ast.parse(open(p, encoding="utf-8", errors="ignore").read())
            except Exception as e:
                out.append({"file": p, "err": str(e)}); continue
            items = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    bases = [ast.unparse(b) for b in node.bases]
                    items.append(f"class {node.name}({', '.join(bases)})")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    items.append(f"def {node.name}({', '.join(args)})")
            out.append({"file": p, "lines": len(open(p, encoding='utf-8', errors='ignore').readlines()), "items": items})
open("tmp/_ast.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("files:", len(out))

