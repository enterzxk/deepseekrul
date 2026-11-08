
import re, json
from pypdf import PdfReader
files = {
 "rccp": "docs/papers/Retrieval-Corrected Conformal Prediction for Time Series.pdf",
 "localized": "docs/papers/Conformal Prediction for Time Series with Deep Sequence Models.pdf",
 "split_cp": "docs/papers/Valid and Efficient Split Conformal Regression for Time Series.pdf",
 "qr_ssm": "docs/papers/A Quantile Regression Approach for Remaining Useful Life Estimation with State Space Models.pdf",
 "qr_biz": "docs/papers/Uncertainty and Business-Aware Remaining Useful Life Estimation for Semiconductor Manufacturing.pdf",
}
out={}
for k,f in files.items():
    r=PdfReader(f); t=" ".join(" ".join((r.pages[i].extract_text() or "") for i in range(len(r.pages))).split())
    urls=sorted(set(re.findall(r"https?://[^\s,;\)\]]+", t)))
    gh=[u for u in urls if "github" in u or "zenodo" in u or "huggingface" in u]
    out[k]={"github":gh[:4],"n_urls":len(urls)}
open("tmp/_urls.json","w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=1))
print("ok")

