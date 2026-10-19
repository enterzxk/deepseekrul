from playwright.sync_api import sync_playwright
import json,time,sys,os,urllib.parse,re
infile=sys.argv[1]; outdir=sys.argv[2]; os.makedirs(outdir,exist_ok=True)
cells=[l.strip() for l in open(infile,encoding='utf-8') if l.strip()]
base="https://database.batteryarchive.org/public/dashboards/61PzyYn9u56njtPDOh37VVBFasGusUlxc7WZ1FSj?org_slug=default&p_cell_id="
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True, args=["--no-sandbox"])
    for idx,cell in enumerate(cells):
        fn=os.path.join(outdir,re.sub(r'[^A-Za-z0-9_.-]+','_',cell)+'.json')
        if os.path.exists(fn):
            try:
                d=json.load(open(fn));
                if d.get('query_result',{}).get('data',{}).get('rows'): print('SKIP',cell); continue
            except Exception: pass
        ok=False
        for attempt in range(3):
            ctx=b.new_context(); pg=ctx.new_page(); saved=[]
            def on_resp(r):
                if '/api/queries/26/results' in r.url:
                    try: saved.append(r.text())
                    except Exception: pass
            pg.on('response',on_resp)
            url=base+urllib.parse.quote(cell,safe='')+'&_=%d'%int(time.time()*1000)
            try:
                pg.goto(url, wait_until='networkidle', timeout=60000); time.sleep(1.5)
            except Exception as e: pass
            good=None
            for s in saved:
                try:
                    d=json.loads(s); rows=d.get('query_result',{}).get('data',{}).get('rows')
                    if rows and any(cell in str(r.get('series','')) for r in rows):
                        good=s; break
                except Exception: pass
            if good:
                open(fn,'w',encoding='utf-8').write(good)
                n=len(json.loads(good)['query_result']['data']['rows'])
                print(f'OK {idx+1}/{len(cells)} {cell} {n}',flush=True); ok=True
            ctx.close()
            if ok: break
            time.sleep(2)
        if not ok: print(f'FAIL {idx+1}/{len(cells)} {cell}',flush=True)
    b.close()
