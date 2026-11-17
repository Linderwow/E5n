r"""E5Q amendment 2: robustness gauntlet for the surviving profit-capture rule (target = k band half-widths on the band leg).
    python e5q_robust.py
"""
import os, sys, time
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
from e5b_lab import G, say


def main():
    t0 = time.time(); C.load(); M = C.M
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0)
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    ctx = NS(atr=slot('atr'), vw=slot('vw'), half=(slot('ub') - slot('lb')) / 2, mdl=fair)
    keep = np.asarray(M.keep); isA = np.asarray(G.isA); A_ = keep & isA; B_ = keep & ~isA; dates = M.dates
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    tgt = lambda k: (lambda S: (S.op >= k * S.W, np.zeros_like(S.mfe, bool)))
    base = M.band; ub, _ = Q.days(fair + base); vb, _ = Q.days(base); db, _ = Q.days(dep + base)
    say(); say("=" * 160)
    say(f"PROFIT CAPTURE GAUNTLET  e5q_robust.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5Q_PLAN.txt amendment 2) the survivor Q2 = band-leg profit target at 2 band half-widths")
    say("=" * 160)
    say(f"   baseline: book Sharpe {sh(ub[A_]):.2f} / {sh(ub[B_]):.2f} (2013-19 / 2019-26), ${ub[A_].mean() * 252:+,.0f} / ${ub[B_].mean() * 252:+,.0f} a year, max DD ${dd(ub[A_]):,.0f} / ${dd(ub[B_]):,.0f}")
    P = {}; ok1 = True
    say("K1 PLATEAU - target in band half-widths:")
    for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        p = Q.overlay(tgt(k), ctx); u, _ = Q.days(fair + p); P[k] = (p, u); d = u - ub
        hit = (np.abs(p - base).sum(1) > 0)[keep].sum() / (keep.sum() / 252)
        say(f"   {k:3.1f}: book Sharpe {sh(u[A_]):.2f} / {sh(u[B_]):.2f}, ${u[A_].mean() * 252:+7,.0f} / ${u[B_].mean() * 252:+7,.0f} a year, max DD ${dd(u[A_]):,.0f} / ${dd(u[B_]):,.0f}, "
            f"paired t {tt(d[A_]):+.2f} / {tt(d[B_]):+.2f}, target hit on {hit:.0f} days a year")
        if k in (1.5, 2.5) and not (sh(u[A_]) >= sh(ub[A_]) and sh(u[B_]) >= sh(ub[B_])): ok1 = False
    say(f"   K1 {'PASS' if ok1 else 'FAIL'}: both neighbours (1.5 and 2.5) at or above the baseline in both halves" if ok1 else "   K1 FAIL: a neighbour (1.5 or 2.5) falls below the baseline in a half")
    p2, u2 = P[2.0]; d = pd.Series((u2 - ub)[keep], index=dates[keep]); yr = d.groupby(d.index.year).sum()
    say("K2 YEARS - paired difference (Q2 - baseline, book) by year: " + "  ".join(f"{y} ${v:+,.0f}" for y, v in yr.items()))
    dB = d[d.index > L.JOIN]; yB = dB.groupby(dB.index.year).sum(); wo = dB[dB.index.year != yB.idxmax()].sum()
    ok2 = (yr > 0).sum() >= 8 and wo > 0
    say(f"   positive in {(yr > 0).sum()} of {len(yr)} years; 2019-26 total ${dB.sum():+,.0f}, without its best year ({yB.idxmax()}, ${yB.max():+,.0f}) ${wo:+,.0f} -> K2 {'PASS' if ok2 else 'FAIL'}")
    top = dB.nlargest(5); rest = dB.drop(top.index).sum(); ok3 = rest >= 0
    say(f"K3 DROP-5 - 2019-26 difference ${dB.sum():+,.0f}; its 5 best days ${top.sum():+,.0f} ({', '.join(f'{i:%Y-%m-%d} ${v:+,.0f}' for i, v in top.items())}); without them ${rest:+,.0f} -> K3 {'PASS' if ok3 else 'FAIL'}")
    nz = dB[dB != 0]; say(f"   days where the target changed the result in 2019-26: {len(nz)} ({(nz > 0).sum()} better, {(nz < 0).sum()} worse; average better ${nz[nz > 0].mean():+,.0f}, average worse ${nz[nz < 0].mean():+,.0f})")
    v2, _ = Q.days(p2); dv = v2 - vb; ok4 = tt(dv[B_]) >= 1.5 and dv[A_].mean() * 252 >= -300
    say(f"K4 BAND LEG ALONE - difference ${dv[A_].mean() * 252:+,.0f} / ${dv[B_].mean() * 252:+,.0f} a year, paired t {tt(dv[A_]):+.2f} / {tt(dv[B_]):+.2f}; band-leg max DD ${dd(v2[A_]):,.0f} vs ${dd(vb[A_]):,.0f} / "
        f"${dd(v2[B_]):,.0f} vs ${dd(vb[B_]):,.0f} -> K4 {'PASS' if ok4 else 'FAIL'}")
    d2, _ = Q.days(dep + p2); ok5 = sh(d2[A_]) >= sh(db[A_]) and sh(d2[B_]) >= sh(db[B_])
    say(f"K5 MODEL LEG AS BACKTESTED - book Sharpe {sh(d2[A_]):.2f} vs {sh(db[A_]):.2f} / {sh(d2[B_]):.2f} vs {sh(db[B_]):.2f} -> K5 {'PASS' if ok5 else 'FAIL'}")
    allok = ok1 and ok2 and ok3 and ok4 and ok5
    say(f"GAUNTLET: {'ALL PASS - candidate (shadow forward test + optional setting, off by default)' if allok else 'FAILED (' + ', '.join(n for n, o in (('K1', ok1), ('K2', ok2), ('K3', ok3), ('K4', ok4), ('K5', ok5)) if not o) + ') - near miss, not adopted; shadow column only'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
