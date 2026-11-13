"""Stage D trainer: engine-level damage recursion, three losses, four arms.

Why this is a separate module and not a flag on ``train_hc.py``: the Stage C loop
trains on independent windows, so the recursion ``D_t = sum_{u<=t} delta_u``
cannot be formed (a window carries no time index). Stage D must walk whole
trajectories, so the batching unit changes from "window" to "engine", and the
loss gains two engine-level terms. Keeping them in one function would mean two
different batching rules behind one name.

The four arms differ ONLY in what produces the RUL prediction and whether the
damage path is used as an auxiliary signal:

  B0  direct head, no damage path at all            L_RUL
  B1  direct head + damage path as auxiliary        L_RUL + lam_T L_term + lam_S L_smooth/S0
  B2  arrival only, fixed rate nu_t                 L_RUL + lam_T L_term + lam_S L_smooth/S0
  B3  arrival only, learned rate g_phi(., k)        same as B2

B2/B3 must not carry a bypass RUL head (task book section 7 D5): the regression
head's parameters are excluded from the optimiser, so it cannot act as one.

Numerics that are not negotiable:

  * The encoder runs under bf16 autocast, but the damage recursion runs in
    float32. Each increment is ~5e-3 and the path sums ~500 of them to reach 1;
    bf16's 8-bit mantissa would cost percent-level accuracy in the accumulated
    damage, which is exactly the quantity the threshold comparison reads.
  * ``S0`` is read from the shared preparation record, never recomputed per
    epoch: a moving normaliser makes the objective drift.
  * The training loss uses the raw operator output. Clipping at ``rul_cap``
    happens only in evaluation (D4.1/D4.6).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.metrics import per_engine_rmse, rmse, summarize
from . import config_hc as C
from .arrival import first_passage
from .data_d import engine_batch, train_lifetime_mean
from .hi_d import DSPT_HC_Structured
from .csr_pool import DSPT_CSR, remap_b0_state_dict, CSR_ALLOWED_MISSING

ARMS = ('B0', 'B1', 'B2', 'B3')
# M3-C (same-age ranking): the arm that has to show the structured HI is not a
# re-parameterised cycle timer. M3B's anchored increment is exactly lambda0 for
# every engine at step 0, so two engines at the same age get the same damage --
# measured on the 40-epoch probe: 32,424 same-age pairs, 100% of them tied. A tie
# carries no sensor information, and more budget cannot break it, because the
# timer is the optimum of the RUL loss alone. The ranking loss is the only term
# that can force a departure from it.
RANK_ARMS = ('M3C',)
# M3-B: the age-anchored increment. Same arrival machinery as B3, only the
# increment parameterisation changes (delta = lambda0 * exp(clamp(u))), so it is
# a one-variable ablation of B3 rather than a new pipeline. M3C shares that
# increment and adds the ranking loss, so M3C vs M3B isolates the loss.
ANCHORED_ARMS = ('M3B',) + RANK_ARMS
# M7 (SCR) is a formal arm, not a flag. run_id() keys on the arm, so a flag
# would collide with B3's directory and the guard would (correctly) refuse the
# second run. M7 builds and runs exactly like B3; the only difference is the
# alive penalty on top of the loss.
SCR_ARMS = ('M7',)
# M4 (CSR) rides the B0 backbone on purpose: B0 is the only arm proven strong,
# and attaching a module to B3 would make the result unattributable -- exactly
# the trap M3B fell into (it needs a B3 cell just to say what it beat). A formal
# arm, not a flag, so run_id()/ledger/report split automatically.
CSR_ARMS = ('M4',)
# M2 (CDE) hangs on the B0 backbone on purpose: B0 is the only arm proven
# strong, so a gain measured on it is a real gain. Hanging a module on B3
# would confound 'module useless' with 'backbone already bad'.
CDE_ARMS = ('M2',)
# M5 (DPC): multi-span degradation consistency. Training-only auxiliary head;
# the prediction path stays B0's reg_head(z), so inference is unchanged and the
# only difference from B0 is the auxiliary loss acting on the shared encoder.
DPC_ARMS = ('M5',)
# M6 (EFR): evidence-gated future degradation rate on the B3 backbone.
# nu_future = nu_age * exp(clamp(g_t * a_tk, -a, a)), g_t = sigmoid(gate(q_t)).
# The amplitude head is zero-init, so step 0 is exactly B3.
EFR_ARMS = ('M6',)
# Arms with no damage head in play: the head exists in the module (same build
# path as B0) but is never trained and never used by the prediction.
NOHEAD_ARMS = ('B0',) + CSR_ARMS + CDE_ARMS + DPC_ARMS
HI_ARMS = ('B1', 'B2', 'B3') + ANCHORED_ARMS + SCR_ARMS + EFR_ARMS
ARRIVAL_ARMS = ('B2', 'B3') + ANCHORED_ARMS + SCR_ARMS + EFR_ARMS
# Sprint arms (Baseline B and the age-only control). They are NOT part of the
# frozen 48-cell matrix -- adding them to ARMS would silently change it -- but
# they share the data interface, the censored hinge and the evaluator, which is
# what makes their numbers comparable to B0-B3.
SPRINT_ARMS = ('AGE', 'AAR')
ALL_ARMS = (ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS + CDE_ARMS
            + DPC_ARMS + EFR_ARMS)


class ResidualHead(nn.Module):
    """r_theta(z_t) for Baseline B. Last layer is zero-initialised.

    Zero init is the definition of the baseline, not a trick: at step 0 the
    prediction is exactly the age prior, so any improvement later is attributable
    to the sensor residual rather than to a lucky random head. Gradient still
    reaches the hidden layer from step 1 (dL/dW1 = W2^T dL/dpre with W2 = 0 at
    step 0, nonzero after).
    """

    def __init__(self, d_model, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z):
        return self.net(z).reshape(-1)


def fit_age_prior(ebt, cap, min_alive=5):
    """A[t] = mean over training engines alive at t of min(n_e - t, cap).

    This is the least-squares optimal prediction from age alone for the capped
    label, fitted on the TRAINING split only. Cycles with fewer than
    ``min_alive`` engines alive have no reliable mean, so the last supported
    value is carried forward (flat extrapolation) rather than extrapolating a
    trend that the data does not support.
    """
    lens = np.asarray(ebt['lengths'], np.int64)
    Tmax = int(lens.max())
    tot = np.zeros(Tmax + 2, np.float64)
    cnt = np.zeros(Tmax + 2, np.int64)
    for n in lens:
        r = np.minimum(n - np.arange(1, n + 1, dtype=np.float64), cap)
        tot[1:n + 1] += r
        cnt[1:n + 1] += 1
    A = np.zeros(Tmax + 2, np.float32)
    supported = np.nonzero(cnt >= min_alive)[0]
    if supported.size == 0:
        supported = np.nonzero(cnt > 0)[0]
    if supported.size == 0:
        raise RuntimeError('age prior: no cycle has an alive engine')
    last = float(tot[supported[-1]] / cnt[supported[-1]])
    # one pass: supported cycles use their own mean, everything after the last
    # supported cycle is flat, everything before is the first supported mean
    first = float(tot[supported[0]] / cnt[supported[0]])
    A[:] = last
    A[:supported[0]] = first
    for t in supported:
        A[t] = tot[t] / cnt[t]
    A[0] = A[1] if Tmax >= 1 else first
    return A


def run_id(arm, subset, seed):
    """The registry id AND the artifact directory name -- one name, never two.

    The registry pre-registers these ids; the run must write into the directory
    of the same name or the ledger points at nothing. Kept here as the single
    source: the dispatcher asks this module for the id instead of rebuilding the
    string, and the report reads cells through the same function.
    """
    return 'HC-D-%s-%s-s%d-D1' % (arm, subset, int(seed))


def code_fingerprint():
    """sha256 over the modules that decide the numbers in a Stage D run.

    Recorded per run so a metrics.json can be tied to one code state even though
    a40 has no git. config_hc is included because it carries the protocol
    constants; arrival/hi_d/data_d/train_d are the four that change numerics.
    """
    here = Path(__file__).parent
    names = ('arrival.py', 'hi_d.py', 'data_d.py', 'train_d.py', 'config_hc.py')
    h = hashlib.sha256()
    missing = []
    per_file = {}
    for n in names:
        p = here / n
        h.update(n.encode())
        if p.exists():
            # Hash the LF-normalised bytes, i.e. what the git blob holds. The
            # repo has core.autocrlf=true, so a Windows checkout has CRLF while
            # a Linux checkout has LF; hashing raw working-tree bytes gives one
            # commit two different fingerprints, which is the same "it changed
            # but nobody can say why" failure this field exists to prevent.
            b = p.read_bytes().replace(b'\r\n', b'\n')
            h.update(b)
            per_file[n] = hashlib.sha256(b).hexdigest()[:32]
        else:
            missing.append(n)
    git_sha = None
    try:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(here),
                           capture_output=True, text=True, timeout=5)
        git_sha = r.stdout.strip() or None
    except Exception:
        git_sha = None
    return dict(files=list(names), files_sha256=h.hexdigest()[:32],
                file_sha256=per_file, git_sha=git_sha, missing=missing)


# Snapshot at import, not when metrics.json is written. code_fingerprint() reads
# the files on disk, so calling it at the end of a run records whatever was
# synced into the tree *during* training rather than the code that was loaded at
# step 0. That is how four runs on one snapshot ended up recorded under two
# different fingerprints, and why an archived run showed a hash that matches no
# commit. With the snapshot taken here, a mid-run sync can no longer rewrite
# history: the run reports what it actually imported.
_FP_AT_IMPORT = code_fingerprint()


def _now():
    import datetime
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_d(arm, subset, K, l_mean, head_seed, dev, init_from, seed=101):
    """Build one arm. The head init is seeded and identical for B1/B2/B3.

    An unknown arm name is a hard error. The dispatch below has no final else,
    so before this check an unregistered or misspelled arm silently built B0 --
    the run then produced a genuine B0 result under another arm's run_id, which
    is the kind of failure that poisons the ledger rather than crashing.
    """
    if arm not in ALL_ARMS:
        raise ValueError('unknown arm %r; known: %s' % (arm, list(ALL_ARMS)))
    cfg = dict(C.hc_config('A1', subset, seed))
    bb = cfg['backbone']
    kw = dict(
        fixed_rate=(arm == 'B2'), K=int(K), cycle_mode='none',
        num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
        dropout=cfg['dropout'], max_span=cfg['span'],
        ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
        hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
        use_pad_mask=True, n_cond=cfg['n_cond'],
        l_mean=(None if (arm in NOHEAD_ARMS or arm in SPRINT_ARMS) else l_mean),
        anchored=(arm in ANCHORED_ARMS),
        head_seed=(None if arm in NOHEAD_ARMS else head_seed))
    if arm in CSR_ARMS:
        net = DSPT_CSR(csr_mode='cond', **kw)
    else:
        net = DSPT_HC_Structured(**kw)
    ck = torch.load(init_from, map_location='cpu', weights_only=True)
    sd = dict(ck['state_dict'] if 'state_dict' in ck else ck)
    if arm in CSR_ARMS:
        sd = remap_b0_state_dict(sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    allowed = CSR_ALLOWED_MISSING if arm in CSR_ARMS else ('head.',)
    bad = [m for m in missing if not m.startswith(allowed)]
    if bad or unexpected:
        raise RuntimeError('init mismatch: missing=%s unexpected=%s' % (bad, unexpected))
    if arm in CDE_ARMS:
        from .cde import CDECorrection, CDEEvidence
        net.cde = CDEEvidence(14, 32)
        net.cde_head = CDECorrection(32)
    if arm in EFR_ARMS:
        from .cde import CDEEvidence
        from .efr import EFRCorrection
        net.cde = CDEEvidence(14, 32)
        net.efr = EFRCorrection(net.d_model, 32, clip=1.0)
    if arm in DPC_ARMS:
        from .dpc import DPCHead
        rh = net.reg_head
        dim = int(rh.in_features if hasattr(rh, 'in_features') else rh[0].in_features)
        net.dpc = DPCHead(dim)
    return net.to(dev), sorted(missing)


def trainable(net, arm):
    """B0 never touches the damage head; B2/B3 never touch the regression head.

    The exclusion is the enforcement of D5's "no bypass RUL head" constraint:
    an unused head whose parameters are still optimised is a bypass waiting to
    be used. ``requires_grad_`` is set in BOTH directions -- leaving the excluded
    head at its default True would keep it in ``net.parameters()`` and it would
    silently be trained (and would still count as trainable in the audit).
    """
    out = []
    for name, p in net.named_parameters():
        keep = True
        if arm in NOHEAD_ARMS and name.startswith('head.'):
            keep = False
        if arm in ARRIVAL_ARMS and name.startswith('reg_head.'):
            keep = False
        if arm in SPRINT_ARMS:
            # AGE is a fitted curve: nothing to train. AAR trains the backbone
            # plus the residual head only -- the damage and regression heads are
            # not part of Baseline B and must not drift into being one.
            keep = (arm == 'AAR') and not (name.startswith('head.')
                                           or name.startswith('reg_head.'))
        p.requires_grad_(keep)
        if keep:
            out.append(p)
    return out


def module_counts(net, arm):
    def n(prefix):
        return int(sum(p.numel() for k, p in net.named_parameters()
                       if k.startswith(prefix)))
    return dict(backbone=n('') - n('head.') - n('reg_head.'),
                damage_head=n('head.'), reg_head=n('reg_head.'),
                trainable=int(sum(p.numel() for p in net.parameters() if p.requires_grad)))


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #
def engine_tensors(eb, i, n, dev):
    x = torch.from_numpy(np.asarray(eb['x'][i, :n], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][i, :n], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][i, :n], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][i, :n], np.int64)).to(dev)
    cy = torch.zeros(n, device=dev)
    r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
    return x, m, b, cy, cd, r


def forward_engine(net, arm, eb, i, n, dev, K):
    """One engine: z_t (fp32), damage path, and the RUL prediction."""
    if arm in SPRINT_ARMS:
        # Age prior (and, for AAR, a sensor residual on top of it). AGE never
        # touches the encoder -- it is a fitted curve, so running it through the
        # backbone would only cost time and add nothing.
        r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
        cyc = torch.from_numpy(np.asarray(eb['cycle_raw'][i, :n], np.float32)).to(dev)
        idx = cyc.long().clamp(0, net.age_prior.numel() - 1)
        pred = net.age_prior[idx]
        if arm == 'AAR':
            x, m, b, cy, cd, _ = engine_tensors(eb, i, n, dev)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=(dev.type == 'cuda')):
                z = net.pooled_feature(x, m, b, cy, cd)
            pred = pred + net.residual(z.float())
        return dict(pred=pred, r=r, n=n)
    x, m, b, cy, cd, r = engine_tensors(eb, i, n, dev)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                        enabled=(dev.type == 'cuda')):
        z = net.pooled_feature(x, m, b, cy, cd)
    # fp32 from here on: the recursion sums ~500 increments of ~5e-3.
    z = z.float()
    if arm in DPC_ARMS:
        # B0's prediction path plus the latent sequence the auxiliary needs.
        return dict(pred=net.reg_head(z).reshape(-1), r=r, n=n, z=z)
    if arm == 'B0':
        return dict(pred=net.reg_head(z).reshape(-1), r=r, n=n)
    delta = net.head.increment(z.unsqueeze(0)).reshape(-1)
    D = torch.cumsum(delta, dim=0)
    nu = net.current_rate(delta.unsqueeze(0)).reshape(-1)
    out = dict(delta=delta, D=D, nu=nu, r=r, n=n, z=z)
    if arm in CDE_ARMS:
        # M2 = B0 plus an explicit condition-matched evidence correction.
        # The correction head is zero-init, so step 0 is bit-identical to B0
        # and any later difference is attributable to the CDE evidence.
        q = net.cde(x, b, m)
        out['pred'] = net.reg_head(z).reshape(-1) + net.cde_head(q)
        out['q'] = q
    elif arm in EFR_ARMS:
        # B3's rate, multiplied by an evidence-gated correction that is 1 at
        # step 0. The arrival operator is the reviewed one, called as-is.
        q = net.cde(x, b, m)
        nu_age = net.head.future_rates(D, nu, z, K)
        out['pred'] = first_passage(D, net.efr.rates(nu_age, D, nu, z, q, K))
        out['q'] = q
    elif arm == 'B1':
        out['pred'] = net.reg_head(z).reshape(-1)
    else:
        out['pred'] = first_passage(D, net.head.future_rates(D, nu, z, K))
    return out


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def rul_loss(pred, r, cap):
    """Censored hinge (D4.1). Uncropped: no clamp of pred at cap."""
    capped = r >= cap
    return torch.where(capped, torch.clamp(cap - pred, min=0.0) ** 2,
                       (pred - r) ** 2)


def rank_pair_loss(fos, margin, gap):
    """Same-age ranking loss on the damage path (M3-C / ADC).

    Two training engines A, B alive at the same cycle t, with true remaining
    lives r_A > r_B (A is the healthier one), must satisfy

        D_A + margin <= D_B        penalty = relu(margin - (D_B - D_A))

    i.e. the engine closer to failure has to carry the larger accumulated damage.
    A timer that depends on age alone gives every engine the same D at the same
    t, so every pair is tied and the penalty is exactly ``margin`` -- that is the
    failure mode this term exists to break, and it is why the term is the only
    evidence that the structured HI is reading the sensors.

    Pairs are built only from the engines handed in, which are the engines of one
    training batch. A validation engine never reaches this function: ``fos`` is
    filled from ``ebt`` (split='train') in the training loop.

    ``gap`` is the minimum true-RUL difference that makes a pair separable. Below
    it the label itself cannot order the two engines, so penalising the model for
    not ordering them would be fitting noise.

    Returns (sum of penalties over compared points, number of compared points),
    or (None, 0) when this batch contains no separable pair.
    """
    tot, n = None, 0
    for a in range(len(fos)):
        for b in range(a + 1, len(fos)):
            ra, rb = fos[a]['r'], fos[b]['r']
            m = min(int(ra.numel()), int(rb.numel()))
            if m < 1:
                continue
            ra, rb = ra[:m], rb[:m]
            dr = ra - rb
            sel = dr.abs() >= gap
            if not bool(sel.any()):
                continue
            Da, Db = fos[a]['D'][:m], fos[b]['D'][:m]
            # orient so that ``hi`` is the engine closer to failure
            hi = torch.where(dr > 0, Db, Da)
            lo = torch.where(dr > 0, Da, Db)
            pen = torch.relu(margin - (hi - lo))[sel]
            tot = pen.sum() if tot is None else tot + pen.sum()
            n += int(pen.numel())
    return tot, n


def term_sums(net, arm, fo, cap, scr=False):
    """Per-engine SUMS and counts, never means.

    Why sums: the batch loss is a reduction over engines, and ||sum_i L_i|| is
    not sum_i ||L_i||. Reducing per engine first (mean) and averaging the norms
    afterwards measures something else entirely -- the advisor caught exactly
    this. Sums let the caller apply the same reduction the optimiser sees.

    The three terms reduce differently, and that is deliberate:
      rul       point-weighted  (every cycle is a sample)
      terminal  engine-equal    (one anchor per engine; a long engine must not
                                 outvote a short one)
      smooth    point-weighted  (a second difference per cycle)
    """
    pt = rul_loss(fo['pred'], fo['r'], cap)
    out = dict(rul_sum=pt.sum(),
               rul_n=torch.tensor(float(pt.numel()), device=pt.device))
    if arm in HI_ARMS:
        out['term_sum'] = (fo['D'][-1] - 1.0) ** 2
        out['term_n'] = torch.ones((), device=out['term_sum'].device)
        if fo['delta'].numel() > 1:
            d2 = (fo['delta'][1:] - fo['delta'][:-1]) ** 2
            out['smooth_sum'] = d2.sum()
            out['smooth_n'] = torch.tensor(float(d2.numel()), device=d2.device)
        else:
            out['smooth_sum'] = torch.zeros((), device=fo['delta'].device)
            out['smooth_n'] = torch.ones((), device=fo['delta'].device)
        if scr:
            # M7 SCR: while the engine is still alive the damage must stay below
            # the threshold. r_t == 0 is the failure cycle, which 'terminal'
            # already anchors, so the mask is strict.
            alive = fo['r'] > 0.0
            if bool(alive.any()):
                over = torch.relu(fo['D'][alive] - 1.0) ** 2
                out['alive_sum'] = over.sum()
                out['alive_n'] = torch.tensor(float(over.numel()), device=over.device)
    if arm in DPC_ARMS and 'z' in fo:
        d, k = net.dpc.loss(fo['z'])
        out['dpc_sum'] = d
        out['dpc_n'] = torch.tensor(1.0, device=d.device)
    return out


def aggregate(acc, s0):
    """Reduce accumulated sums to the three loss terms the optimiser sees."""
    L = dict(rul=acc['rul_sum'] / acc['rul_n'].clamp_min(1.0))
    if 'term_sum' in acc:
        L['terminal'] = acc['term_sum'] / acc['term_n'].clamp_min(1.0)
        L['smooth'] = acc['smooth_sum'] / acc['smooth_n'].clamp_min(1.0) / s0
    if 'alive_sum' in acc:
        L['alive'] = acc['alive_sum'] / acc['alive_n'].clamp_min(1.0)
    if 'dpc_sum' in acc:
        L['dpc'] = acc['dpc_sum'] / acc['dpc_n'].clamp_min(1.0)
    return L


def losses_for(net, arm, fo, cap, s0, lam_t, lam_s, lam_a=0.0):
    """Single-engine convenience wrapper (probes only). Batch code uses sums."""
    L = aggregate(term_sums(net, arm, fo, cap), s0)
    total = L['rul']
    if 'terminal' in L:
        total = total + lam_t * L['terminal'] + lam_s * L['smooth']
    if 'alive' in L:
        total = total + lam_a * L['alive']
    return total, L


# --------------------------------------------------------------------------- #
# preparation: l_mean, b0, S0, head fingerprint  (once per subset)
# --------------------------------------------------------------------------- #
def calibrate_s0(net, ebt, dev, K, n_batches=8, bs=8, seed=0, floor=1e-12):
    """S0 = E[(delta_{t+1} - delta_t)^2] on the initial weights, detached.

    Two things here are not style:

      * ``net.eval()`` is mandatory. With dropout on, the second difference is
        measured on a randomly masked network, S0 comes out ~2.7x too large and
        seed-dependent -- the same failure shape as incident 14. The assertion
        below turns that into a crash instead of a wrong normaliser.
      * No bf16 autocast. ``probes/d_acceptance.py`` measures S0 in fp32, and the
        two records must agree digit for digit or an auditor cannot tell whether
        the run used the calibrated value. bf16 is a training-speed choice.
    """
    if net.training:
        raise AssertionError('S0 must be measured with dropout off: call net.eval() first')
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    E = len(ebt['lengths'])
    vals = []
    with torch.no_grad():
        for _ in range(n_batches):
            for i in torch.randint(0, E, (bs,), generator=g).tolist():
                n = int(ebt['lengths'][i])
                x, m, b, cy, cd, _ = engine_tensors(ebt, i, n, dev)
                z = net.pooled_feature(x, m, b, cy, cd).float()
                d = net.head.increment(z.unsqueeze(0)).reshape(-1)
                if d.numel() > 1:
                    vals.append(float(((d[1:] - d[:-1]) ** 2).mean()))
    raw = float(np.mean(vals))
    return dict(raw=raw, used=max(raw, floor), floor=floor,
                floor_hit=bool(raw < floor), n_batches=n_batches, batch=bs,
                seed=int(seed), n_terms=len(vals), dropout='eval')


def prepare(subset, data_dir, K, head_seed, dev, init_from, out_dir):
    """Write the shared record every arm of this subset reads."""
    lrec = train_lifetime_mean(subset, data_dir)
    net, _ = build_d('B3', subset, K, lrec['mean'], head_seed, dev, init_from)
    net.eval()  # S0 is a property of the init, not of a dropout draw
    ebt = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='train')
    s0 = calibrate_s0(net, ebt, dev, K)
    rec = dict(subset=subset, K=int(K), head_seed=int(head_seed),
               l_mean=lrec['mean'], train_n=lrec['n'],
               train_units=lrec['units'], train_lengths=lrec['lengths'],
               b0=net.head.init_b0, softplus_b0=float(F.softplus(
                   torch.tensor(net.head.init_b0))),
               head_fingerprint=net.head.fingerprint(),
               s0=s0, created_utc=_now(),
               note='measured once per subset; B1/B2/B3 read the same numbers')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ('%s_shared.json' % subset)
    if p.exists():
        raise SystemExit('REFUSED: %s already exists; the shared record is written once' % p)
    p.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return p, rec


# --------------------------------------------------------------------------- #
# evaluation (D4.6 six quantities)
# --------------------------------------------------------------------------- #
def metrics_from_arrays(preds, ys_raw, units, cuts, cap, K):
    """The ONE place labels get capped. evaluate() calls this; tests call this.

    Splitting it out is deliberate: the advisor's admission condition is an
    actual-path check (r_raw=150, pred=125 -> capped error 0, raw error 25), and
    a check that re-implements the cap proves nothing about the code that runs.
    """
    preds = np.asarray(preds, np.float64)
    ys_raw = np.asarray(ys_raw, np.float64)
    units = np.asarray(units)
    cuts = np.asarray(cuts)
    ys = np.minimum(ys_raw, cap)          # frozen metric label
    clipped = np.clip(preds, 0.0, cap)
    unc = ys_raw < cap                    # censoring is defined on the raw RUL
    capd = ~unc
    m = dict(n=int(ys.size),
             # main metric and selection: capped labels, clipped predictions
             val_engine_mean_rmse=per_engine_rmse(units, ys, clipped),
             val_pooled_rmse=rmse(clipped, ys),
             frac_capped=float(capd.mean()) if ys.size else None,
             rmse_uncensored=(float(np.sqrt(np.mean((clipped[unc] - ys_raw[unc]) ** 2)))
                              if unc.any() else None),
             # uncapped truth vs the operator's own (unclipped) output, but only
             # where the truth is inside the horizon: beyond K the operator
             # cannot represent it and including those points flatters the model
             rmse_uncapped=(float(np.sqrt(np.mean((preds[ys_raw <= K] - ys_raw[ys_raw <= K]) ** 2)))
                            if (ys_raw <= K).any() else None),
             frac_r_gt_K=float((ys_raw > K).mean()) if ys.size else None,
             boundary_hit=float((preds[capd] >= cap).mean()) if capd.any() else None,
             exact_boundary=(float((np.abs(preds[capd] - cap) < 1e-6).mean())
                             if capd.any() else None),
             # Pre-clip diagnostics (2026-11-13). The clip hides failure
             # magnitude: M4 froze at val_eng = 69.527 for 13 epochs because
             # every raw prediction went negative and np.clip squashed it to 0,
             # and from the clipped side alone we could not tell -5 from -500.
             # Recording the raw side is what makes that diagnosable at all.
             pred_raw_min=float(preds.min()) if preds.size else None,
             pred_raw_mean=float(preds.mean()) if preds.size else None,
             pred_raw_max=float(preds.max()) if preds.size else None,
             frac_pred_nonpositive=(float((preds <= 0.0).mean())
                                    if preds.size else None),
             frac_pred_at_cap=float((preds >= cap).mean()) if preds.size else None,
             rmse_by_censoring=dict(
                 uncensored=(float(np.sqrt(np.mean((clipped[unc] - ys_raw[unc]) ** 2)))
                             if unc.any() else None),
                 censored=(float(np.sqrt(np.mean((clipped[capd] - ys[capd]) ** 2)))
                           if capd.any() else None)))
    m['per_cut'] = {int(c): dict(n=int((cuts == c).sum()),
                                 rmse=float(np.sqrt(np.mean((clipped[cuts == c] - ys[cuts == c]) ** 2))))
                    for c in np.unique(cuts)}
    m['_preds'] = preds
    m['_ys'] = ys
    m['_ys_raw'] = ys_raw
    m['_units'] = units
    m['_cuts'] = cuts
    return m


def evaluate(net, arm, eb, dev, K, cap):
    """Cut-cycle evaluation: 5 cuts per engine, engine-mean RMSE (frozen metric).

    LABELS ARE CAPPED FOR THE MAIN METRIC. The frozen protocol scores
    ``min(r, rul_cap)``; scoring a clipped prediction against the RAW ``n - t``
    charges 25 cycles of error to a sample with r=150 and prediction 125, where
    the protocol records zero. The advisor caught this; it moves both model
    selection and D-G1. Raw labels are kept separately for the uncapped
    diagnostics, which are reported apart and never used for selection.
    """
    net.eval()
    preds, ys_raw, units, cuts = [], [], [], []
    with torch.no_grad():
        for i in range(len(eb['lengths'])):
            n = int(eb['lengths'][i])
            fo = forward_engine(net, arm, eb, i, n, dev, K)
            ct = eb['cut_t'][i]
            cv = eb['cut_valid'][i]
            for j in range(len(ct)):
                if cv[j] <= 0:
                    continue
                t = int(ct[j])
                preds.append(float(fo['pred'][t - 1]))
                ys_raw.append(float(n - t))
                units.append(int(eb['engine_id'][i]))
                cuts.append(j)
    return metrics_from_arrays(preds, ys_raw, units, cuts, cap, K)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def term_grads(net, arm, ebt, dev, K, cap, s0, lam_t, lam_s, n_eng=3,
               scr=False, lam_a=0.0, lam_dpc=0.0):
    """D4.5: gradient norm of the BATCH-AGGREGATED term, not of a mean of norms.

    ``||grad sum_i L_i||`` is what the optimiser produces; averaging per-engine
    gradient norms (the earlier version) measures a different quantity and is
    wrong by the triangle inequality. The batch here is a fixed sample of
    ``n_eng`` engines reduced with the same rules as the training step:
    rul point-weighted, terminal engine-equal, smooth point-weighted.

    Reports both readings, because the spec is ambiguous:
      unweighted  ||grad L_i||            D4.5's literal wording
      weighted    ||grad lambda_i L_i||   what the optimiser sees
    The ratio used for the gate is the weighted one; the unweighted one is kept
    for the record.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    acc = {}
    for i in range(min(n_eng, len(ebt['lengths']))):
        n = int(ebt['lengths'][i])
        fo = forward_engine(net, arm, ebt, i, n, dev, K)
        # scr must be threaded through here, otherwise the SCR term never enters
        # grad_terms and the only record of the alive penalty is a single-batch
        # log line -- which cannot answer "was the constraint ever triggered".
        for k, v in term_sums(net, arm, fo, cap, scr=scr).items():
            acc[k] = acc[k] + v if k in acc else v
    L = aggregate(acc, s0)
    w = {'rul': 1.0, 'terminal': lam_t, 'smooth': lam_s, 'alive': lam_a,
         'dpc': lam_dpc}
    unweighted, weighted = {}, {}
    for k, v in L.items():
        gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        nrm = float(sum(g.norm() ** 2 for g in gs if g is not None)) ** 0.5
        unweighted[k] = nrm
        weighted[k] = nrm * w[k]
    return dict(unweighted=unweighted, weighted=weighted,
                values={k: float(v) for k, v in L.items()})


def _selfcheck_degenerate_guard():
    """Prove the degenerate-output guard CAN fire, inside the real entry point.

    A guard that only exists in the source is worth nothing -- that is the M4
    mistake in a different costume ("6 assertions passed, the real run died").
    This runs at the top of every training run, so if the diagnostic field or the
    threshold ever drifts, the cell refuses to start instead of burning the
    budget on a constant metric.

    numpy only: metrics_from_arrays is the ONE place labels get capped, so this
    exercises the same code the run uses.
    """
    ys = np.array([50.0, 80.0])
    units = np.array([0, 0])
    cuts = np.array([10, 10])
    bad = metrics_from_arrays(np.array([-5.0, -12.0]), ys, units, cuts, 125.0, 200)
    assert bad['frac_pred_nonpositive'] == 1.0, bad['frac_pred_nonpositive']
    assert abs(bad['pred_raw_min'] + 12.0) < 1e-12, bad['pred_raw_min']
    assert abs(bad['pred_raw_mean'] + 8.5) < 1e-12, bad['pred_raw_mean']
    assert bad['frac_pred_nonpositive'] > 0.5, 'the guard would NOT have fired'
    good = metrics_from_arrays(np.array([40.0, 90.0]), ys, units, cuts, 125.0, 200)
    assert good['frac_pred_nonpositive'] == 0.0, good['frac_pred_nonpositive']
    capped = metrics_from_arrays(np.array([125.0, 125.0]), ys, units, cuts, 125.0, 200)
    assert capped['frac_pred_at_cap'] == 1.0, capped['frac_pred_at_cap']
    return dict(bad_frac=bad['frac_pred_nonpositive'],
                bad_raw_min=bad['pred_raw_min'],
                bad_raw_mean=bad['pred_raw_mean'],
                good_frac=good['frac_pred_nonpositive'],
                cap_frac=capped['frac_pred_at_cap'])


def train_arm(arm, subset, seed, data_dir, shared_path, out_dir, dev,
              init_from, K, epochs=None, patience=None, batch=None, lr=None,
              weight_decay=None, lam=None, lam_t=None, lam_s=None, log=print,
              budget_override=None, scr=False, lam_a=None, lam_dpc=None,
              lam_rank=None, rank_gap=None, rank_margin=None):
    # Identity comes from the arm name. --scr survives only as a compatibility
    # flag and is rejected unless the arm is already M7 (see main()).
    scr = bool(scr or (arm in SCR_ARMS))
    gsc = _selfcheck_degenerate_guard()
    log('  degenerate guard selfcheck: bad_frac=%.2f raw_min=%.1f good_frac=%.2f '
        'cap_frac=%.2f' % (gsc['bad_frac'], gsc['bad_raw_min'],
                           gsc['good_frac'], gsc['cap_frac']))
    sh = json.loads(Path(shared_path).read_text(encoding='utf-8'))
    K = int(sh['K'] if K is None else K)
    s0 = float(sh['s0']['used'])
    l_mean = float(sh['l_mean'])
    # One shared pair for all four arms (advisor, 2026-11-13). Read from the
    # frozen calibration record rather than hard-coded, so the value that trained
    # a run is the value in the record and the provenance is explicit.
    # Startup guard: the formal matrix must read the FROZEN weights. Falling
    # back to 125^2 would silently train a different objective than the one the
    # advisor calibrated and froze, and nothing downstream would show it.
    frozen_sha = None
    if lam is None and lam_t is None and lam_s is None:
        fz = Path(shared_path).parent / 'd45_init.json'
        if not fz.exists():
            raise SystemExit('REFUSED: %s missing; the frozen c_T/c_S must exist '
                             '(no silent fallback to 125^2)' % fz)
        raw = fz.read_bytes()
        frozen_sha = hashlib.sha256(raw).hexdigest()
        fr = json.loads(raw)['frozen']
        lam_t, lam_s = float(fr['lam_T']), float(fr['lam_S'])
        lam_src = 'd45_init.json:frozen(c_T=%.6f,c_S=%.6f)' % (fr['c_T'], fr['c_S'])
    else:
        lam_src = 'explicit_cli'
    lam_t = float(C.RUL_CAP ** 2 if lam_t is None else lam_t)
    lam_s = float(C.RUL_CAP ** 2 if lam_s is None else lam_s)
    if lam is not None:
        lam_t = lam_s = float(lam)
    # M7 SCR weight. Kept in the same unit as lam_T (cycles^2) so the two
    # penalties are dimensionally comparable: L_alive is a mean squared damage
    # overshoot and is O(1e-2), so the default mirrors lam_T.
    lam_a = float(C.RUL_CAP ** 2 if lam_a is None else lam_a)
    lam_dpc = float(1.0 if lam_dpc is None else lam_dpc)
    # M3-C ranking. lam_rank is a SHARE of the initial RUL loss, not an absolute
    # weight: the M5 cell spent a whole run learning that a fixed weight on a
    # term whose own scale moves is not a test of anything (40000 -> the auxiliary
    # dominated, 0.05 -> five orders of magnitude too small).
    lam_rank = float(0.05 if lam_rank is None else lam_rank)
    rank_gap = float(10.0 if rank_gap is None else rank_gap)
    rank_margin = float(0.05 if rank_margin is None else rank_margin)
    epochs = int(C.EPOCHS if epochs is None else epochs)
    patience = int(C.PATIENCE if patience is None else patience)
    batch = int(C.BATCH if batch is None else batch)
    lr = float(C.LR if lr is None else lr)
    weight_decay = float(C.WEIGHT_DECAY if weight_decay is None else weight_decay)
    cap = float(C.RUL_CAP)

    out = Path(out_dir)
    if (out / 'metrics.json').exists() and not budget_override:
        raise SystemExit('REFUSED: %s already has metrics.json; a formal cell is never '
                         'overwritten' % out)
    set_seed(seed)
    t0 = time.time()
    dev = torch.device(dev)
    net, new_keys = build_d(arm, subset, K, l_mean, int(sh['head_seed']), dev,
                            init_from, seed=seed)
    ebt = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='train')
    ebv = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='val')
    prior_sha = None
    if arm in SPRINT_ARMS:
        prior = fit_age_prior(ebt, cap)
        net.register_buffer('age_prior', torch.from_numpy(prior).to(dev))
        if arm == 'AAR':
            net.residual = ResidualHead(net.d_model).to(dev)
        prior_sha = hashlib.sha256(np.ascontiguousarray(prior, '<f4').tobytes()
                                   ).hexdigest()[:16]
        log('  age prior: %d cycles, A(1)=%.3f A(Tmax)=%.3f sha=%s'
            % (prior.size - 2, float(prior[1]), float(prior[-3]), prior_sha))
    params = trainable(net, arm)
    nparam = module_counts(net, arm)
    # M5: normalise the auxiliary by its own initial value, exactly like S0.
    # Without this, lam_dpc is an absolute weight on a term whose scale moves:
    # measured 0.0052 -> 0.0960 over 15 epochs (18x), which took the gradient
    # share from 15% to 100% and stopped being a test of M5.
    dpc0 = rul0 = rank0 = None
    if arm in DPC_ARMS:
        was_training = net.training
        net.eval()
        with torch.no_grad():
            dvals, rvals = [], []
            for _i in range(min(8, len(ebt['lengths']))):
                _n = int(ebt['lengths'][_i])
                _fo = forward_engine(net, arm, ebt, _i, _n, dev, K)
                _d, _k = net.dpc.loss(_fo['z'])
                if _k:
                    dvals.append(float(_d))
                    rvals.append(float(rul_loss(_fo['pred'], _fo['r'], cap).mean()))
        net.train(was_training)
        dpc0 = max(1e-12, float(np.mean(dvals))) if dvals else 1.0
        rul0 = max(1e-12, float(np.mean(rvals))) if rvals else 1.0
        # Both scales are measured in the SAME init pass and detached. The loss
        # term is then lam_dpc * (L_dpc/dpc0) * rul0, so lam_dpc is literally
        # "fraction of the initial RUL loss" and does not drift when L_dpc moves.
        log('  dpc0=%.6f rul0=%.2f from %d engines -> lam_dpc is a SHARE of rul0'
            % (dpc0, rul0, len(dvals)))
    if arm in RANK_ARMS:
        # Same pattern as M5: measure both scales in ONE detached pass at init,
        # then the term is lam_rank * (L_rank/rank0) * rul0, so lam_rank is
        # literally "fraction of the initial RUL loss" and cannot drift.
        was_training = net.training
        net.eval()
        with torch.no_grad():
            _fos, _rvals = [], []
            _E = len(ebt['lengths'])
            _n_eng = min(24, _E)
            _idx = [int(round(k * (_E - 1) / max(1, _n_eng - 1)))
                    for k in range(_n_eng)]
            for _i in _idx:
                _n = int(ebt['lengths'][_i])
                _fo = forward_engine(net, arm, ebt, _i, _n, dev, K)
                _fos.append(dict(D=_fo['D'], r=_fo['r']))
                _rvals.append(float(rul_loss(_fo['pred'], _fo['r'], cap).mean()))
            _s, _c = rank_pair_loss(_fos, rank_margin, rank_gap)
            rank0 = (float(_s) / float(_c)) if _s is not None else 0.0
            rul0 = float(np.mean(_rvals)) if _rvals else 1.0
        net.train(was_training)
        if _c <= 0 or rank0 <= 0.0:
            raise AssertionError(
                'M3C: no separable same-age pair at init (%d engines, %d compared '
                'points, gap_r=%.1f); the ranking term would start inert and could '
                'not break the age tie' % (_n_eng, _c, rank_gap))
        log('  rank0=%.6f margin=%.4f rul0=%.2f from %d engines, %d compared '
            'points -> lam_rank is a SHARE of rul0'
            % (rank0, rank_margin, rul0, _n_eng, _c))
    opt = (torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
           if params else None)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
           if opt is not None else None)
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    E = len(ebt['lengths'])

    best, best_state, best_ep, history = float('inf'), None, -1, []
    diag = {}   # last-batch loss terms that are not in the gradient-ratio probe
    n_epochs = epochs if params else 1   # AGE is a fitted curve: one pass is the answer
    for ep in range(1, n_epochs + 1):
        net.train()
        order = torch.randperm(E, generator=g).tolist()
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        acc, pts, tot, seen = {}, 0, 0.0, 0
        batch_fos = []   # M3C: (D, r) per engine of the current batch, for ranking

        def _step(acc, pts, fos):
            """One optimiser step on the batch-aggregated loss."""
            L = aggregate(acc, s0)
            total = L['rul']
            if 'terminal' in L:
                total = total + lam_t * L['terminal'] + lam_s * L['smooth']
            if 'alive' in L:
                total = total + lam_a * L['alive']
                diag['alive'] = float(L['alive'].detach())
            if 'dpc' in L:
                total = total + lam_dpc * (L['dpc'] / dpc0) * rul0
                diag['dpc'] = float(L['dpc'].detach())
            if arm in RANK_ARMS and len(fos) > 1:
                rs, rn = rank_pair_loss(fos, rank_margin, rank_gap)
                if rs is not None:
                    L_rank = rs / float(rn)
                    total = total + lam_rank * (L_rank / rank0) * rul0
                    diag['rank'] = float(L_rank.detach())
                    diag['rank_points'] = int(rn)
            if opt is not None:
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            return float(total.detach()) * pts, pts

        for i in order:
            n = int(ebt['lengths'][i])
            fo = forward_engine(net, arm, ebt, i, n, dev, K)
            if arm in RANK_ARMS:
                batch_fos.append(dict(D=fo['D'], r=fo['r']))
            for k, v in term_sums(net, arm, fo, cap, scr=(arm in SCR_ARMS)).items():
                acc[k] = acc[k] + v if k in acc else v
            pts += n
            if pts >= batch:
                dl, dp = _step(acc, pts, batch_fos)
                tot += dl
                seen += dp
                acc, pts, batch_fos = {}, 0, []
        if pts > 0:
            dl, dp = _step(acc, pts, batch_fos)
            tot += dl
            seen += dp
        if sch is not None:
            sch.step()

        ev = evaluate(net, arm, ebv, dev, K, cap)
        # Degenerate-output sentinel (2026-11-13). A cell whose predictions all
        # sit on a clip boundary has a val metric that is a constant unrelated to
        # its input: M4 did exactly that for 13 epochs (val_eng = 69.527, 15
        # decimals unchanged) and cost a full cell before anyone noticed. Stop on
        # the first such epoch instead of waiting for patience, and say why.
        fz = ev.get('frac_pred_nonpositive')
        if fz is not None and fz > 0.5:
            log('  DEGENERATE ep%03d: %.1f%% of raw predictions are <= 0, so the '
                'metric measures the clip, not the model (raw min=%.3f mean=%.3f). '
                'Stopping -- this cell is not a module result.'
                % (ep, 100.0 * fz, ev['pred_raw_min'], ev['pred_raw_mean']))
            raise SystemExit(
                'DEGENERATE: %.1f%% of predictions at the clip boundary (ep%d); '
                'refusing to burn the remaining budget' % (100.0 * fz, ep))
        tg = (term_grads(net, arm, ebt, dev, K, cap, s0, lam_t, lam_s,
                         scr=(arm in SCR_ARMS), lam_a=(lam_a or 0.0),
                         lam_dpc=((lam_dpc or 0.0) * (rul0 or 1.0) / (dpc0 or 1.0)))
              if params else dict(unweighted={}, weighted={}, values={}))
        w = tg['weighted']
        mx = max(w.values()) if w else 0.0
        ratio = {k: (v / mx if mx > 0 else None) for k, v in w.items()}
        tl = tot / max(1, _epoch_points(ebt))
        history.append(dict(epoch=ep, train_loss=tl,
                            val_engine_mean_rmse=ev['val_engine_mean_rmse'],
                            val_pooled_rmse=ev['val_pooled_rmse'],
                            grad_terms_unweighted=tg['unweighted'],
                            grad_terms_weighted=w, grad_ratio=ratio))
        if ep % 5 == 0 or ep == 1:
            log('  ep%03d loss=%.4f val_eng=%.4f alive=%s dpc=%s rank=%s g_ratio=%s'
                % (ep, tl, ev['val_engine_mean_rmse'],
                   ('%.3e' % diag['alive']) if 'alive' in diag else '-',
                   ('%.4f' % diag['dpc']) if 'dpc' in diag else '-',
                   ('%.4f' % diag['rank']) if 'rank' in diag else '-',
                   {k: ('%.3e' % v if v is not None else None)
                    for k, v in ratio.items()}))
        if ev['val_engine_mean_rmse'] < best - 1e-4:
            best, best_ep = ev['val_engine_mean_rmse'], ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    ev = evaluate(net, arm, ebv, dev, K, cap)
    preds, ys = ev.pop('_preds'), ev.pop('_ys')
    ys_raw = ev.pop('_ys_raw')
    units = ev.pop('_units')
    cuts = ev.pop('_cuts')
    out = Path(out_dir)
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / 'ckpt' / 'best.pt')
    m = dict(arm=arm, subset=subset, seed=int(seed), K=K, s0=s0,
             lam_t=lam_t, lam_s=lam_s, lam_source=lam_src,
             l_mean=l_mean,
             # B0 has no damage head, so it has no head fingerprint: recording
             # the shared one would claim an init the arm never built.
             head_fingerprint=(None if (arm == 'B0' or arm in SPRINT_ARMS)
                               else sh['head_fingerprint']),
             age_prior_sha256=prior_sha,
             frozen_sha256=frozen_sha, code_fingerprint=_FP_AT_IMPORT,
             ref_batch_epoch_log_engines=[int(x) for x in ebt['engine_id'][:3]],
             ref_batch_calibration='first 8 training engines, index order (see d45_init.json)',
             params=nparam, epochs_run=len(history), best_ep=best_ep,
             best_val_engine_mean_rmse=best, elapsed_s=round(time.time() - t0, 1),
             select_on_val=True, budget=dict(epochs=epochs, patience=patience,
                                             batch_points=batch, lr=lr,
                                             weight_decay=weight_decay),
             budget_override=budget_override,
             rank_loss=(dict(lam_share=lam_rank, gap_r=rank_gap,
                             margin_d=rank_margin, rank0=rank0, rul0=rul0)
                        if arm in RANK_ARMS else None),
             created_utc=_now())
    m.update(ev)
    m['grad_terms_last_unweighted'] = history[-1]['grad_terms_unweighted']
    m['grad_terms_last_weighted'] = history[-1]['grad_terms_weighted']
    m['grad_ratio_last'] = history[-1]['grad_ratio']
    (out / 'metrics.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    (out / 'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    import pandas as pd
    pd.DataFrame(dict(engine_id=units, cut=cuts, y_cap=ys, y_raw=ys_raw,
                      y_pred=preds, split='val')).to_csv(out / 'preds.csv', index=False)
    log('DONE %s %s s%d val_eng=%.4f ep=%d params=%d %.1fs'
        % (arm, subset, seed, ev['val_engine_mean_rmse'], best_ep,
           nparam['trainable'], m['elapsed_s']))
    return m


def _epoch_points(ebt):
    return int(np.asarray(ebt['lengths']).sum())


# --------------------------------------------------------------------------- #
# four-arm execution config audit
# --------------------------------------------------------------------------- #
def audit(arms=ARMS, subset='FD001', data_dir=None, K=600, head_seed=101,
          dev='cpu', init_from=None, l_mean=None, root=None):
    """Build all four arms and report exactly what differs.

    The point is not to print a table -- it is to make the differences a fact of
    the built objects, not of the document. Anything that differs beyond the
    three intended axes (RUL source, HI auxiliary, rate mode) is a bug.

    ``l_mean`` should come from the shared preparation record: with a dummy
    value the head fingerprint will not match the acceptance record, and the
    audit would then be certifying a different init than the one that trains.
    """
    if l_mean is None:
        p = Path(root or '.') / ('%s_shared.json' % subset)
        if not p.exists():
            raise SystemExit('run `prepare` first, or pass l_mean: %s missing' % p)
        l_mean = float(json.loads(p.read_text(encoding='utf-8'))['l_mean'])
    rows = {}
    for arm in arms:
        net, new_keys = build_d(arm, subset, K, l_mean, head_seed, dev, init_from)
        trainable(net, arm)
        rows[arm] = dict(
            params=module_counts(net, arm),
            has_damage_head=arm != 'B0',
            damage_head_trained=arm in HI_ARMS,
            reg_head_trained=arm in ('B0', 'B1'),
            uses_arrival=arm in ARRIVAL_ARMS,
            fixed_rate=(arm == 'B2'),
            head_fingerprint=(None if arm == 'B0' else net.head.fingerprint()),
            new_tensors=len(new_keys),
            K=int(net.K))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'train', 'audit', 'id'])
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--arm', default='B3', choices=ALL_ARMS)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--K', type=int, default=600)
    ap.add_argument('--head-seed', type=int, default=101)
    ap.add_argument('--data-dir', default='/media/sangfor/vdb/user/zxk/deepseek-rul/data/CMAPSS')
    ap.add_argument('--init-from',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/tmp/'
                            'wave1_init_A1_s101.pt')
    ap.add_argument('--root',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/d')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lam', type=float, default=None,
                    help='sets both weights; overridden by --lam-t/--lam-s')
    ap.add_argument('--lam-t', type=float, default=None)
    ap.add_argument('--lam-s', type=float, default=None)
    ap.add_argument('--budget-override', default=None)
    ap.add_argument('--scr', action='store_true',
                    help='deprecated compatibility flag; M7 is selected by --arm M7')
    ap.add_argument('--lam-a', type=float, default=None)
    ap.add_argument('--lam-dpc', type=float, default=None,
                    help='M5: weight on the multi-span latent consistency loss')
    ap.add_argument('--lam-rank', type=float, default=None,
                    help='M3C: same-age ranking weight, as a SHARE of the initial RUL loss')
    ap.add_argument('--rank-gap', type=float, default=None,
                    help='M3C: minimum true-RUL difference (cycles) for a same-age pair')
    ap.add_argument('--rank-margin', type=float, default=None,
                    help='M3C: required damage gap between two same-age engines')
    a = ap.parse_args()
    if a.scr and a.arm not in SCR_ARMS:
        raise SystemExit('--scr is no longer an identity: use --arm M7 '
                         '(a flag would share %s\'s run directory)' % a.arm)
    dev = torch.device(a.device)
    root = Path(a.root)

    if a.stage == 'id':
        print(run_id(a.arm, a.subset, a.seed))
        return
    if a.stage == 'prepare':
        p, rec = prepare(a.subset, a.data_dir, a.K, a.head_seed, dev, a.init_from, root)
        print(json.dumps(dict(path=str(p), l_mean=rec['l_mean'], b0=rec['b0'],
                              softplus_b0=rec['softplus_b0'],
                              head_fingerprint=rec['head_fingerprint'],
                              s0=rec['s0']['used'],
                              s0_floor_hit=rec['s0']['floor_hit']), indent=1))
    elif a.stage == 'audit':
        rows = audit(subset=a.subset, data_dir=a.data_dir, K=a.K,
                     head_seed=a.head_seed, dev=dev, init_from=a.init_from,
                     root=root)
        print(json.dumps(rows, indent=1))
        fps = {k: v['head_fingerprint'] for k, v in rows.items() if k != 'B0'}
        if len(set(fps.values())) != 1:
            raise SystemExit('head init differs across B1/B2/B3: %s' % fps)
        print('\nhead init identical across B1/B2/B3: %s' % next(iter(fps.values())))
    else:
        shared = root / ('%s_shared.json' % a.subset)
        out = root / run_id(a.arm, a.subset, a.seed)
        train_arm(a.arm, a.subset, a.seed, a.data_dir, shared, out, dev,
                  a.init_from, a.K, epochs=a.epochs, lam=a.lam,
                  lam_t=a.lam_t, lam_s=a.lam_s,
                  budget_override=a.budget_override,
                  scr=a.scr, lam_a=a.lam_a, lam_dpc=a.lam_dpc,
                  lam_rank=a.lam_rank, rank_gap=a.rank_gap,
                  rank_margin=a.rank_margin)


if __name__ == '__main__':
    main()
