r"""CX confirmation (CX_PLAN.txt): the exact book at 4.5 ticks with the confirmed band filter, its neighbours, years and risk.
    python cx_confirm.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5u_bandml as UB
import cx_search as CS
from e5b_lab import G, say

TK = 4.5


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'CX_RESULTS.txt'); isA = np.asarray(G.isA); keep = np.asarray(M.keep)
    BT = UB.trades(); A = isA[BT.r.values]; DA = BT[A & keep[BT.r.values]]
    feats = ['gap', 'vol120', 'atrp15', 'obv15', 'df15', 'half']; _, names, edges = CS.states(DA, feats, DA)
    S = CS.apply_states(BT, names, edges); col = {nm: i for i, nm in enumerate(names)}
    st = lambda f, k: S[:, col[(f, k)]] > 0
    model = C.path(M.g5, M.g15, 'set', M.grid)
    base, _ = C.simm(model + M.band, TK); kb = L.stats(base); rat = lambda k, h: k['yr' + h] / k['dd' + h]
    say(); say("=" * 150); say(f"CX CONFIRMATION  cx_confirm.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  exact book, 1 + 1, {TK} ticks; cut-offs from 2013-19 band trades: "
        + ", ".join(f"{f} {np.round(edges[f], 5).tolist()}" for f in ('gap', 'vol120'))); say("=" * 150)
    say(f"   B0 baseline                       2013-19 Sharpe {kb['shA']:.2f} ratio {rat(kb, 'A'):.2f} DD ${kb['ddA']:,.0f} | 2019-26 Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr ratio {rat(kb, 'B'):.2f} DD ${kb['ddB']:,.0f}")
    V = {'F1 CONFIRMED: gap mid & vol120 low': st('gap', 1) & st('vol120', 0),
         'n1 neighbour: vol120 low (any gap)': st('vol120', 0), 'n2 neighbour: gap mid (any vol)': st('gap', 1),
         'n3 neighbour: vol120 low & atrp15 low': st('vol120', 0) & st('atrp15', 0), 'n4 neighbour: gap mid & atrp15 low': st('gap', 1) & st('atrp15', 0)}
    for nm, sk in V.items():
        p = UB.path_with(BT, np.where(sk, 0.0, 1.0)); s, _ = C.simm(model + p, TK); k = L.stats(s)
        _, sb = L.halves(s); _, bb = L.halves(base); d = (sb - bb).values; t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        ok = k['shB'] > kb['shB'] and rat(k, 'B') > rat(kb, 'B') and k['ddB'] <= kb['ddB'] and t >= 2.0
        sk_tr = BT[sk & keep[BT.r.values]]; yrs = sk_tr.groupby(pd.DatetimeIndex(M.dates[sk_tr.r.values]).year).usd.sum()
        say(f"   {nm:38s} skips {int(sk.sum()):4d} trades | 2013-19 Sharpe {k['shA']:.2f} ratio {rat(k, 'A'):.2f} DD ${k['ddA']:,.0f} | 2019-26 Sharpe {k['shB']:.2f} "
            f"${k['yrB']:,.0f}/yr ratio {rat(k, 'B'):.2f} DD ${k['ddB']:,.0f} paired t {t:+.2f}" + (' CONFIRMED IN THE BOOK' if ok else ''))
        say(f"      skipped trades' $ by year (2.35-tick costs): " + ", ".join(f"{y}: {v:+,.0f}" for y, v in yrs.items()))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
