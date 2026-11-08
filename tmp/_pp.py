
import os, json, glob
from pypdf import PdfReader
out = []
for f in sorted(glob.glob("docs/papers/*.pdf")):
    try:
        r = PdfReader(f)
        t = " ".join(((r.pages[0].extract_text() or "")[:600]).split())
    except Exception as e:
        t = "ERR " + str(e)[:80]
    out.append({"file": os.path.basename(f), "size_kb": round(os.path.getsize(f)/1024), "head": t[:260]})
open("tmp/_papers.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("n=", len(out))

