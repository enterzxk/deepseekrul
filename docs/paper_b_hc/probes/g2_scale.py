"""G2 scale audit: how large is the cycle term at the point where it is added?

The claim under test is that A2 (scalar cycle) and A3a (Fourier cycle) differ by
~100x in projection output scale, which would make the G2 comparison confounded
by magnitude rather than by encoding form. This measures it instead of assuming.

Where the number is taken (question 3 of the brief):

  * A2 / A3a / A4 / A5: `cycle_enc(cycle)` is added to the encoder output in
    ``DSPT_HC.encode`` and **nothing normalises it afterwards**. So the output of
    the projection IS the tensor added to h -- there is no separate post-norm
    value to report.
  * A3p: the frozen path adds its term *inside* ``DSPT.encode``, before
    ``self.norm``; its projection output is therefore not directly comparable
    and is reported separately with that caveat.

Reported per (arm, subset), on the training windows:

  input stats, projection-output stats (per element and per vector),
  ||cycle term|| / ||backbone h||  (the quantity that actually matters),
  ||dL/d(cycle out)|| per sample and ||dL/dW_proj|| from one real backward pass.

Reads the ledger for the evidence run of each cell. No training, no writes.

Usage: python docs/paper_b_hc/probes/g2_scale.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import data_hc
from paper_b.hc.model_hc import build_hc

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
OUT = ROOT / 'experiments/hc'
LEDGER = ROOT / 'docs/paper_b_hc/registry/runs.csv'
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
ARMS = ['A2', 'A3a', 'A4', 'A5', 'A3p']


def ledger_runs():
    keep = {}
    with LEDGER.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['in_gate_a_table'].strip().lower() != 'true':
                continue
            if row['status'].strip() != 'ok':
                continue
            keep[(row['variant'], row['subset'])] = row['run_id']
    return keep


def load(run_id, dev):
    run = OUT / run_id
    cfg = json.loads((run / 'config.json').read_text())
    net = build_hc(cfg)
    net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                   map_location='cpu', weights_only=True))
    return cfg, net.to(dev).eval()


def stats(x):
    x = np.asarray(x, np.float64)
    return dict(mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), max=float(x.max()),
                p05=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)),
                p95=float(np.percentile(x, 95)))


def main():
    dev = torch.device('cuda:0')
    runs = ledger_runs()
    rows = []
    print('%-4s %-6s %10s %10s | %9s %9s %9s %9s | %8s %8s'
          % ('arm', 'subset', 'in_mean', 'in_std', 'out_mean', 'out_std',
             'out_min', 'out_max', '|term|', '|h|'))
    for v in ARMS:
        for s in SUBSETS:
            rid = runs.get((v, s))
            if rid is None:
                continue
            cfg, net = load(rid, dev)
            if not getattr(net, 'cycle_enc', None):
                print('%-4s %-6s  no cycle encoder (A3p uses the frozen path)' % (v, s))
                continue
            ds = data_hc.build(s, cfg['data_dir'], cfg['span'], cfg['val_cuts'],
                               cycle_source=cfg['cycle_source'])
            tr = ds['train']
            c = torch.from_numpy(np.asarray(tr['cycle_hc'], np.float32)).to(dev)

            with torch.no_grad():
                outs = []
                for i in range(0, len(c), 4096):
                    outs.append(net.cycle_enc(c[i:i + 4096]))
                out = torch.cat(outs, 0)
                # backbone h for the same windows: h_base = (h + term) - term
                x = torch.from_numpy(np.asarray(tr['x'][:512], np.float32)).to(dev)
                m = torch.from_numpy(np.asarray(tr['mask'][:512], np.float32)).to(dev)
                b = torch.from_numpy(np.asarray(tr['baseline'][:512], np.float32)).to(dev)
                cd = torch.from_numpy(np.asarray(tr['cond'][:512], np.int64)).to(dev)
                cy = c[:512]
                h_tot = net.encode(x, b, m, cy, cd)
                term = net.cycle_enc(cy).unsqueeze(1)
                h_base = h_tot - term
                n_term = float(term.norm(dim=-1).mean())
                n_h = float(h_base.norm(dim=-1).mean())

            # one real backward pass: gradient of the training loss through the term
            y = torch.from_numpy(np.asarray(tr['y'][:512], np.float32)).to(dev)
            net.zero_grad(set_to_none=True)
            saved = {}

            def _hook(mod, inp, out):
                out.retain_grad()
                saved['out'] = out

            hh = net.cycle_enc.register_forward_hook(_hook)
            pred = net(x, mask=m, baseline=b, cycle=cy, cond=cd, mode='dual')['prediction']
            loss = torch.nn.functional.mse_loss(pred.reshape(-1), y)
            loss.backward()
            hh.remove()

            g_w = float('nan')
            for name, p in net.cycle_enc.named_parameters():
                if name == 'proj.weight' and p.grad is not None:
                    g_w = float(p.grad.norm())
            go = saved.get('out')
            g_out = float(go.grad.norm(dim=-1).mean()) if (go is not None and
                                                           go.grad is not None) else float('nan')
            n_par = sum(p.numel() for p in net.cycle_enc.parameters())

            o = stats(out.detach().cpu().numpy())
            i_ = stats(c.detach().cpu().numpy())
            rec = dict(variant=v, subset=s, run=rid, n=len(c),
                       input=i_, output=o,
                       out_vec_norm=float(out.norm(dim=-1).mean()),
                       term_norm=n_term, h_norm=n_h,
                       ratio=n_term / n_h if n_h else float('nan'),
                       grad_out=g_out, grad_w=locals().get('g_w', float('nan')),
                       params=n_par)
            rows.append(rec)
            print('%-4s %-6s %10.3f %10.3f | %9.3f %9.3f %9.3f %9.3f | %8.3f %8.3f'
                  % (v, s, i_['mean'], i_['std'], o['mean'], o['std'],
                     o['min'], o['max'], n_term, n_h))

    p = OUT / 'tmp'
    p.mkdir(parents=True, exist_ok=True)
    (p / 'g2_scale.json').write_text(json.dumps(rows, indent=1))

    print('\n## detail: projection output per element')
    print('%-4s %-6s %9s %9s %9s %9s %9s %9s %8s'
          % ('arm', 'subset', 'mean', 'std', 'p05', 'p50', 'p95', 'vecnorm', '|term|/|h|'))
    for r in rows:
        o = r['output']
        print('%-4s %-6s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %8.4f'
              % (r['variant'], r['subset'], o['mean'], o['std'], o['p05'],
                 o['p50'], o['p95'], r['out_vec_norm'], r['ratio']))

    print('\n## gradient through the cycle path (one backward on 2048 train windows)')
    print('%-4s %-6s %12s %12s %10s' % ('arm', 'subset', '|dL/dout|', '|dL/dW_proj|', 'params'))
    for r in rows:
        print('%-4s %-6s %12.4e %12.4e %10d'
              % (r['variant'], r['subset'], r['grad_out'], r['grad_w'], r['params']))

    print('\nwrote %s' % (p / 'g2_scale.json'))


if __name__ == '__main__':
    main()
