r"""SO (SO_PLAN.txt): scale-out - 2 contracts per leg, the second takes profit, the first runs as now; 4.5 ticks a side.
    python so_scaleout.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from types import SimpleNamespace as NS
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
from e5b_lab import G, say

TK = 4.5


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'SO_RESULTS.txt')
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band.copy()
    ctx = NS(atr=slot('atr'), vw=slot('vw'), half=(slot('ub') - slot('lb')) / 2, mdl=model)
    tg = lambda a: (lambda S: (S.op >= a * S.A, S.op * 0 > 1)); qt = lambda k: (lambda S: (S.op >= k * S.W, S.op * 0 > 1))
    def capped(leg, rule):
        keepb = M.band; M.band = leg
        try: return Q.overlay(rule, ctx)
        finally: M.band = keepb
    bT = {a: capped(band, tg(a)) for a in (2, 3, 4)}; bQ = capped(band, qt(2)); mT = {a: capped(model, tg(a)) for a in (1, 2)}
    V = {'B1 band 2nd contract: target 2 ATR': 2 * model + band + bT[2], 'B2 band 2nd contract: target 3 ATR': 2 * model + band + bT[3],
         'B3 band 2nd contract: target 4 ATR': 2 * model + band + bT[4], 'B4 band 2nd contract: target 2 band half-widths': 2 * model + band + bQ,
         'M1 model 2nd contract: target 1 ATR': model + mT[1] + 2 * band, 'M2 model 2nd contract: target 2 ATR': model + mT[2] + 2 * band,
         'X1 band 3 ATR + model 1 ATR': model + mT[1] + band + bT[3], 'X2 band Q2 + model 2 ATR': model + mT[2] + band + bQ}
    base, _ = C.simm(2 * model + 2 * band, TK); kb = L.stats(base); one, _ = C.simm(model + band, TK); k1 = L.stats(one)
    rat = lambda k, h: k['yr' + h] / k['dd' + h] if k['dd' + h] > 0 else np.nan
    say(); say("=" * 150); say(f"SO  SCALE-OUT, 2 CONTRACTS A LEG  so_scaleout.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {TK} ticks a side; $/yr per $ max DD = ratio"); say("=" * 150)
    say(f"   E5n now (1 + 1)            2013-19 Sharpe {k1['shA']:.2f} ${k1['yrA']:,.0f}/yr DD ${k1['ddA']:,.0f} | 2019-26 Sharpe {k1['shB']:.2f} ${k1['yrB']:,.0f}/yr DD ${k1['ddB']:,.0f} ratio {rat(k1, 'B'):.2f}")
    say(f"   BASELINE 2 + 2, no targets 2013-19 Sharpe {kb['shA']:.2f} ${kb['yrA']:,.0f}/yr DD ${kb['ddA']:,.0f} | 2019-26 Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr DD ${kb['ddB']:,.0f} ratio {rat(kb, 'B'):.2f}")
    for nm, net in V.items():
        s, _ = C.simm(net, TK); k = L.stats(s); sel = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']
        _, sb = L.halves(s); _, bb = L.halves(base); d = (sb - bb).values; t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        ok = sel and k['shB'] > kb['shB'] and k['ddB'] <= kb['ddB'] and t >= 2.0
        say(f"   {nm:44s} 2013-19 Sharpe {k['shA']:.2f} ({k['shA'] - kb['shA']:+.2f}) DD ${k['ddA']:,.0f} {'SEL' if sel else '   '}| 2019-26 Sharpe {k['shB']:.2f} "
            f"({k['shB'] - kb['shB']:+.2f}) ${k['yrB']:,.0f}/yr DD ${k['ddB']:,.0f} ratio {rat(k, 'B'):.2f} t {t:+.2f}" + (' CONFIRMED' if ok else ''))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
