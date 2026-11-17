import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5d_more as D
from e5b_lab import G, say
X = L.X; ND = L.ND


def main():
    t0 = time.time(); L.setup(); m, b = G.model, G.band; base = m + b
    say(); say("=" * 160); say(f"E5B AMENDMENT C  e5e_more2.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  FOMC days, VWAP trend leg; $ per micro at today's price, live cost on net orders"); say("=" * 160)
    book = lambda t: L.sim(m, b, ticks=t)
    fomc = {pd.Timestamp(s.strip()) for s in open(os.path.join(L.X.PATHS.WEB, 'fomc_dates.txt')) if s.strip() and not s.startswith('#')}
    is_f = np.array([d in fomc for d in G.dates]); dec = G.dec
    A = G.keep & G.isA; bs = book(2.35); onA = bs[bs.index <= L.JOIN]; fA = onA[[d in fomc for d in onA.index]]
    say(f"2013-19: {len(fA)} FOMC days in the book: total ${fA.sum():+,.0f}, average ${fA.mean():+.0f} a day vs ${onA.mean():+.0f} on all days; worst ${fA.min():,.0f}; losing {np.mean(fA < 0):.0%}")
    mk = lambda lo, hi: np.where(is_f[:, None] & (dec[None, :] >= lo) & (dec[None, :] < hi), 0.0, 1.0)
    K = [dict(name='K1', desc='FOMC days: both legs flat from 13:55 to the close', fn=lambda t: L.sim(net=base * mk(265, 999), ticks=t)),
         dict(name='K2', desc='FOMC days: both legs flat 13:55-14:30', fn=lambda t: L.sim(net=base * mk(265, 300), ticks=t)),
         dict(name='K3', desc='FOMC days: no trading at all', fn=lambda t: L.sim(net=base * mk(0, 999), ticks=t))]
    D.judge('K', K, book, 'the netted book B0')
    vw = G.m(G.T.vwapd.values); VW = []
    for nm, cad, dz, lab in (('VW1', 5, 0.0, 'every 5 minutes'), ('VW2', 15, 0.0, 'every 15 minutes'), ('VW3', 30, 0.0, 'every 30 minutes'), ('VW4', 15, 0.0005, 'every 15 minutes, 0.05% dead zone')):
        pos = np.zeros(G.n); lp = np.zeros((G.n, ND))
        for j in range(ND):
            if dec[j] % cad == 0:
                v = vw[:, j]; pos = np.where(np.isnan(v), pos, np.where(v > dz, 1.0, np.where(v < -dz, -1.0, 0.0)))
            lp[:, j] = pos
        lp[~G.keep] = 0
        VW.append(dict(name=nm, desc=f'new leg: side of VWAP, {lab}', fn=(lambda lp: lambda t: L.sim(net=base + lp, ticks=t))(lp), leg=L.sim(net=lp)))
    D.judge('VW', VW, book, 'the netted book B0')
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
