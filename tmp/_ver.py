
import re
from pypdf import PdfReader
f = "docs/papers/Similarity-calibrated conformal prediction data-free coverage guarantees for remaining-useful-life intervals under operating-regime transfer.pdf"
r = PdfReader(f)
full = " ".join(" ".join((p.extract_text() or "").split()) for p in r.pages)
for pat in ["0.590", "0.060", "2L", "9.8"]:
    m = full.find(pat)
    print(pat, "->", "FOUND" if m >= 0 else "MISSING", ("| " + full[max(0,m-120):m+120]) if m >= 0 else "")

