
from pypdf import PdfReader
r = PdfReader("docs/papers/sensors-26-02249.pdf")
t = " ".join((r.pages[0].extract_text() or "").split())
print(t[:900])

