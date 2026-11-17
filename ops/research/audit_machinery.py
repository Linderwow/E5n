r"""AUDIT 2026-11-17: does the select-on-2013-19 / confirm-on-2019-26 machinery reject changes with no information, and does it
catch a real improvement? Changes to the model leg of the book (E5B amendment-A simulator, e5c_fast):
  NULL     two books, each with a different random 20% of the model leg's signals set to flat, compared with each other (both carry
           no information, same expected result) - 200 draws
  PLANTED  as NULL, but the challenger also sets to flat a share q of the signals whose next 5 minutes go AGAINST it (deliberate
           look-ahead, an oracle of known strength), q = 0.25%, 0.5%, 1% (200 draws each); plus q = 2% / 5% against the real base
  (first run 08:39 compared random cuts with the full base - not a fair null, cutting signals also cuts edge; superseded)
Selection (E5B stage 1): Sharpe 2013-19 above the base and max drawdown 2013-19 not above it.
Confirmation, two versions: (a) as e5c_fast stage 2 - Sharpe and max drawdown 2019-26 better and bootstrap P < 0.05 computed by
X.boot_p on the WHOLE series; (b) bootstrap P on 2019-26 only; (c) paired t on 2019-26 daily differences >= 2.0.
    python audit_machinery.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import G, say

X = L.X


def grade(s, base, kb):
    k = L.stats(s); sel = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']
    _, sb = L.halves(s); _, bb = L.halves(base)
    better = k['shB'] > kb['shB'] and k['ddB'] <= kb['ddB']
    pa = X.boot_p(s, base, reps=2000); pb = X.boot_p(sb, bb, reps=2000)
    d = (sb - bb).values; t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
    return dict(sel=sel, a=sel and better and pa < 0.05, b=sel and better and pb < 0.05, c=sel and t >= 2.0, t=t, dsh=k['shB'] - kb['shB'], dshA=k['shA'] - kb['shA'])


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'AUDIT_MACHINERY.txt'); rng = np.random.default_rng(1117)
    base_net = C.path(M.g5, M.g15, 'set', M.grid) + M.band; base, _ = C.simm(base_net); kb = L.stats(base)
    fwd = np.zeros_like(M.dpx)
    for k in range(5): fwd[:, :C.NM - k] += M.dpx[:, k:]
    tgt = C.target(M.g5, M.g15); cells = M.grid[None, :] & (tgt != 0)
    against = cells & (fwd * tgt < 0)
    say(); say("=" * 150); say(f"AUDIT OF THE EDGE-FINDING MACHINERY  audit_machinery.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}"); say("=" * 150)
    say(f"   base book: 2013-19 Sharpe {kb['shA']:.2f}, 2019-26 {kb['shB']:.2f}; model-leg signal cells {int(cells.sum()):,}, of which against the next 5 minutes {against.sum() / cells.sum():.0%}")

    def book(mask):
        f5 = np.where(mask, 0.0, M.g5); s, _ = C.simm(C.path(f5, M.g15, 'set', M.grid) + M.band); return s

    rows = []
    for q in (0.02, 0.05):
        for _ in range(10):
            r = grade(book(against & (rng.random(cells.shape) < q)), base, kb); r['kind'] = f'PLANTED q={q:.0%} vs base'; rows.append(r)
    for i in range(200):
        b0 = book(cells & (rng.random(cells.shape) < 0.20)); k0 = L.stats(b0)
        r = grade(book(cells & (rng.random(cells.shape) < 0.20)), b0, k0); r['kind'] = 'NULL'; rows.append(r)
        for q in (0.0025, 0.005, 0.01):
            m = (cells & (rng.random(cells.shape) < 0.20)) | (against & (rng.random(cells.shape) < q))
            r = grade(book(m), b0, k0); r['kind'] = f'PLANTED q={q:.2%}'; rows.append(r)
        if (i + 1) % 50 == 0: print(f"{i + 1}/200, {time.time() - t0:.0f} s", flush=True)
    R = pd.DataFrame(rows)
    say(f"   {'change':<16s} {'n':>4s} | {'selected 2013-19':>16s} | {'confirmed (a) as coded':>22s} {'(b) P on 2019-26':>17s} {'(c) paired t>=2':>16s} | {'mean 2019-26 t':>14s} {'mean dSharpe 2019-26':>20s}")
    for kind, g in R.groupby('kind', sort=False):
        say(f"   {kind:<16s} {len(g):>4d} | {g.sel.mean():>16.0%} | {g.a.mean():>22.0%} {g.b.mean():>17.0%} {g.c.mean():>16.0%} | {g.t.mean():>+14.2f} {g.dsh.mean():>+20.3f}")
    say("   reading: NULL rows should be selected about half the time (coin flip) and confirmed rarely (~5% or less);")
    say("   PLANTED rows should be confirmed more often as q grows - that is the machinery's power to find a real improvement.")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
