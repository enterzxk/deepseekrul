#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
mkdir -p results/dsh/review_r3/sources
echo '--- tooling ---'
command -v pdftotext >/dev/null && echo pdftotext=yes || echo pdftotext=no
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'PY'
mods = ['fitz','pypdf','PyPDF2','pdfminer']
for m in mods:
    try:
        __import__(m); print(m, 'ok')
    except Exception as e:
        print(m, 'missing')
PY
