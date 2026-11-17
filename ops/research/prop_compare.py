r"""Which futures prop firm's 50K evaluation suits E5n (information, not a trial; 2026-11-17). Same bootstrap paths as e5p_predict
(fair model leg). Rules: Bulenox as stored from its help center; the others from 2026 third-party rule guides (conflicting in places -
verify on each firm's own site before buying). All: $3,000 target, threshold trailing the highest end-of-day balance, breached intraday,
locking at the start balance (except Bulenox's stored engine). No resets, no fees.
    python prop_compare.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5x_lab as X
import e5p_predict as PR

FIRMS = (('Bulenox 50K (stored rules)', None),
         ('Topstep 50K Combine', dict(dd=2000, dll=None, cons=0.55)),
         ('Apex 50K EOD', dict(dd=2000, dll=1000, cons=None)),
         ('Lucid Pro / Tradeify Growth 50K', dict(dd=2000, dll=None, cons=None)))


def run(BL, X_, W_, r, target=3000.0):
    n_p, n_d = X_.shape; bal = np.zeros(n_p); hi = np.zeros(n_p); best = np.zeros(n_p); alive = np.ones(n_p, bool); ok = np.zeros(n_p, bool); when = np.full(n_p, -1)
    for t in range(n_d):
        x, w = BL.day(X_[:, t], W_[:, t], r['dll']); thr = np.minimum(hi - r['dd'], 0.0)
        br = alive & (bal + w < thr); alive &= ~br
        bal = np.where(alive, bal + x, bal); best = np.where(alive, np.maximum(best, x), best); hi = np.where(alive, np.maximum(hi, bal), hi)
        hit = alive & (bal >= target) & ((r['cons'] is None) | (best <= (r['cons'] or 1) * bal)); when[hit & ~ok] = t + 1; ok |= hit; alive &= ~hit
    return ok, when


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0); O = X.SIDE
    L.OUT = os.path.join(HERE, 'PROP_COMPARE.txt')
    L.say(); L.say("=" * 150); L.say(f"PROP FIRMS FOR E5n  prop_compare.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  fair model leg; pass = reach +$3,000 before the threshold (no resets)"); L.say("=" * 150)
    L.say(f"   {'firm':34s} | {'ticks':>5s} | {'mix':5s} | pass within 21 sessions | within 63 | median sessions to pass")
    for tk in (2.35, 4.5):
        X.SIDE = lambda t, tk=tk: O(t - 2.35 + tk); mk = PR.strengths(M, fair, dep)[0][1]
        for cm, cb in ((0, 6), (1, 6), (1, 3)):
            u, w = mk(cm, cb); nt = (u != 0).astype(float); X_, W_, _ = BL.paths(63, u, w, nt, 1, BL.SEED_REPORT)
            for name, r in FIRMS:
                if r is None:
                    ok21, _ = BL.evaluation(X_[:, :21], W_[:, :21], a['dd'], a['dll'], a['target']); ok63, _ = BL.evaluation(X_, W_, a['dd'], a['dll'], a['target']); med = np.nan
                else:
                    ok63, when = run(BL, X_, W_, r); ok21 = ok63 & (when <= 21); med = np.median(when[ok63]) if ok63.any() else np.nan
                L.say(f"   {name:34s} | {tk:5.2f} | {cm} + {cb} | {ok21.mean():22.0%} | {ok63.mean():9.0%} | {med:6.0f}")
    X.SIDE = O; L.say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
