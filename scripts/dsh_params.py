import glob, json
for tag in ('M0_raw_tf','M6_tfA','M1_global','M7_rawback','M4_racdr'):
    fs = sorted(glob.glob(f'experiments/order03/{tag}/FD002/*/metrics.json'))
    if not fs: print(tag,'missing'); continue
    m = json.load(open(fs[0]))
    cfg = m.get('config', {})
    print(f"{tag:12s} params={m.get('params'):>7} variant={m.get('variant')} d_model={cfg.get('d_model','-')} "
          f"features={cfg.get('use_baseline')}/{cfg.get('use_ema_features')}/{cfg.get('use_cycle')}")
