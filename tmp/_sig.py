
import ast, json
def show(path, names):
    src = open(path, encoding="utf-8", errors="ignore").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    res = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            seg = "\n".join(lines[node.lineno-1: min(node.lineno+18, len(lines))])
            res.append(seg[:900])
    return res
out = {}
out["model_forward"] = show("src/model.py", {"forward"})[:2]
out["bundle"] = show("src/dataset.py", {"CMAPSSDataBundle"})[:1]
out["prior"] = show("src/lifetime_prior/nested_prior.py", {"StaircasePrior"})[:1]
open("tmp/_sig.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")

