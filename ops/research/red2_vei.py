r"""RED2 N8 (RED2_PLAN.txt): the VEI filter (daily ATR(10) / ATR(50) < 1.0) on E5n's band-leg entries, inside the book.
    python red2_vei.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import red_rules as R
from e5b_lab import G, say
from e5q_capture import days


def vei():
    out = {}
    for name in ('nq', 'mnqx'):
        d = R.load(name); g = pd.DataFrame({'d': d['tday'], 'h': d['h'], 'l': d['l'], 'c': d['c']}).groupby('d').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
        pc = g.c.shift(1); tr = np.maximum.reduce([g.h - g.l, (g.h - pc).abs(), (g.l - pc).abs()])
        a10 = pd.Series(np.asarray(tr), index=g.index).ewm(alpha=1 / 10, adjust=False).mean(); a50 = pd.Series(np.asarray(tr), index=g.index).ewm(alpha=1 / 50, adjust=False).mean()
        v = (a10 / a50).shift(1)
        for k, x in v.items(): out.setdefault(name, {})[int(k)] = x
    return out


def main():
    t0 = time.time(); C.load(); M = C.M; V = vei(); L.OUT = os.path.join(HERE, 'RED2_RESULTS.txt')
    ep = np.array([int(pd.Timestamp(x).value // 86_400_000_000_000) for x in M.dates]); isA = np.asarray(G.isA)
    v = np.array([V['nq' if isA[i] else 'mnqx'].get(int(ep[i]), np.nan) for i in range(M.n)]); ok = np.isfinite(v) & (v < 1.0)
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); band = M.band
    p = band.copy(); prev = np.zeros(M.n); skip = np.zeros(M.n, bool)
    for j in range(band.shape[1]):
        bj = band[:, j]; new = bj != prev; start = new & (bj != 0); skip = np.where(new, start & ~ok, skip); p[:, j] = np.where(skip, 0.0, bj); prev = bj
    keep = np.asarray(M.keep); A_ = keep & isA; B_ = keep & ~isA
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    ub, _ = days(fair + band); u, _ = days(fair + p)
    say(); say("=" * 160); say(f"RED2 N8  VEI FILTER ON E5n's BAND LEG  red2_vei.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  daily ATR(10)/ATR(50) < 1.0 to take a band entry; sessions passing {ok[keep].mean():.0%}"); say("=" * 160)
    for lab, x in (('BASELINE (E5n as built)', ub), ('N8 band entries only when VEI < 1.0', u)):
        say(f"   {lab:40s} | 2013-19 Sharpe {sh(x[A_]):.2f} ${x[A_].mean() * 252:+7,.0f}/yr DD ${dd(x[A_]):6,.0f} | 2019-26 Sharpe {sh(x[B_]):.2f} ${x[B_].mean() * 252:+7,.0f}/yr DD ${dd(x[B_]):6,.0f}")
    sel = sh(u[A_]) >= sh(ub[A_]) + 0.05
    if sel:
        dl = u[B_] - ub[B_]; t = dl.mean() / dl.std() * np.sqrt(len(dl))
        say(f"   SELECTED on 2013-19; 2019-26 paired t {t:+.2f} (bar 2.0) -> {'CONFIRMED' if t >= 2.0 and sh(u[B_]) > sh(ub[B_]) else 'not confirmed'}")
    else: say("   not selected on 2013-19 (needs Sharpe >= baseline + 0.05)")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
