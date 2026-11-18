r"""FB (FB_PLAN.txt): flip brake on the model leg after K position changes in a day.
    python fb_brake.py run
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import say

REAL = 4.5


def brake_path(f5, f15, k, mult):
    M = C.M; n = f5.shape[0]; pos = np.zeros(n); flips = np.zeros(n); out = np.zeros((n, C.NM))
    for j in range(C.NM):
        if M.grid[j]:
            ok = ~np.isnan(f5[:, j]); t = C.target(f5[:, j], f15[:, j], C.T0)
            th = np.zeros(n) if mult is None else C.target(f5[:, j], f15[:, j], mult * C.T0)
            braked = np.where(t == pos, pos, th)
            new = np.where(flips >= k, braked, t)
            new = np.where(ok, new, pos)
            flips = flips + (new != pos); pos = new
        out[:, j] = pos
    return out


def run():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'FB_RESULTS.txt')
    base_model = C.path(M.g5, M.g15, 'set', M.grid)
    chk = brake_path(M.g5, M.g15, 10 ** 6, 1.5)
    assert np.array_equal(chk, base_model), 'brake path with K = inf must equal the deployed model leg'
    base_net = base_model + M.band
    ch = (np.abs(np.diff(np.c_[np.zeros(M.n), base_model], axis=1)) > 0)
    early = ch[:, :M.dec.searchsorted(126)].sum(1)[M.keep]
    V = {}
    for i, k in enumerate((4, 6, 8)):
        V[f'FB{i + 1} after {k} changes: entries need 1.5 x T0'] = brake_path(M.g5, M.g15, k, 1.5) + M.band
    for i, k in enumerate((4, 6, 8)):
        V[f'FB{i + 4} after {k} changes: exits only'] = brake_path(M.g5, M.g15, k, None) + M.band
    b45, tb = C.simm(base_net, REAL); b23, _ = C.simm(base_net, 2.35); kb = L.stats(b45); kb23 = L.stats(b23)
    say(); say("=" * 150); say(f"FB  FLIP BRAKE ON THE MODEL LEG AT {REAL} TICKS A SIDE  fb_brake.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n book 1 + 1"); say("=" * 150)
    say(f"   model-leg changes by 11:35 per day: median {np.median(early):.0f}, 95% {np.percentile(early, 95):.0f}, max {early.max():.0f}; days >= 11: {(early >= 11).mean() * 100:.1f}%")
    say(f"   FB0 baseline: 2013-19 Sharpe {kb['shA']:.2f} DD ${kb['ddA']:,.0f} | 2019-26 Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr DD ${kb['ddB']:,.0f} | {tb / (len(b45) / 252):,.0f} round trips/yr | (2.35 ticks: 2019-26 Sharpe {kb23['shB']:.2f})")
    for nm, net in V.items():
        s, tr = C.simm(net, REAL); s23, _ = C.simm(net, 2.35); k = L.stats(s); k23 = L.stats(s23)
        sel = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']; _, sb = L.halves(s); _, bb = L.halves(b45); d = (sb - bb).values; t = d.mean() / d.std() * np.sqrt(len(d))
        ok = sel and k['shB'] > kb['shB'] and k['ddB'] <= kb['ddB'] and t >= 2.0 and k23['shB'] >= kb23['shB']
        say(f"   {nm:42s} 2013-19 Sharpe {k['shA']:.2f} ({k['shA'] - kb['shA']:+.2f}) DD ${k['ddA']:,.0f} {'SEL' if sel else '   '}| 2019-26 Sharpe {k['shB']:.2f} ({k['shB'] - kb['shB']:+.2f}) "
            f"${k['yrB']:,.0f}/yr DD ${k['ddB']:,.0f} t {t:+.2f} | {tr / (len(s) / 252):,.0f} trips/yr | 2.35 ticks 2019-26 Sharpe {k23['shB']:.2f}" + (' CONFIRMED' if ok else ''))
    say(f"6 trials; run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run': run()
    else: print(__doc__)
