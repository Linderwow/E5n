r"""EXIT (EXIT_PLAN.txt): exits aimed at drawdown, 4.5 ticks a side.
    python exit_dd.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from types import SimpleNamespace as NS
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
import e5x_lab as X
from e5b_lab import G, say

TK = 4.5


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'EXIT_RESULTS.txt'); U = X.U; side = X.SIDE(TK)
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band.copy()
    ctx = NS(atr=slot('atr'), vw=slot('vw'), half=(slot('ub') - slot('lb')) / 2, mdl=model)
    no = lambda S: S.op * 0 > 1
    stop = lambda a: (lambda S: (S.op <= -a * S.A, no(S)))
    trail = lambda a: (lambda S: (S.op <= S.mfe - a * S.A, no(S)))
    be = lambda a: (lambda S: ((S.mfe >= a * S.A) & (S.op <= 0), no(S)))
    def ov(leg, rule):
        kb = M.band; M.band = leg
        try: return Q.overlay(rule, ctx)
        finally: M.band = kb
    def daycap(net, cap):
        step = net * M.dpx / M.o930[:, None] * U; tr = np.abs(np.diff(np.c_[np.zeros(M.n), net], axis=1)); cum = np.cumsum(step, 1) - side * U * np.cumsum(tr, 1)
        hit = cum <= -cap; first = np.where(hit.any(1), hit.argmax(1), net.shape[1]); out = net.copy()
        for i in np.flatnonzero(first < net.shape[1]): out[i, first[i] + 1:] = 0.0
        return out
    V = {'BX1 band hard stop 1.5 ATR': model + ov(band, stop(1.5)), 'BX2 band hard stop 2 ATR': model + ov(band, stop(2)),
         'BX3 band hard stop 2.5 ATR': model + ov(band, stop(2.5)), 'BX4 band trail 2 ATR from the start': model + ov(band, trail(2)),
         'BX5 band trail 2.5 ATR from the start': model + ov(band, trail(2.5)), 'BX6 band breakeven after 1 ATR': model + ov(band, be(1)),
         'MX1 model day cap -$600': daycap(model, 600) + band, 'MX2 model day cap -$1,000': daycap(model, 1000) + band,
         'MX3 model stop 2 ATR': ov(model, stop(2)) + band}
    V['DX1 book day stop -$1,000'] = daycap(model + band, 1000)
    V['CX1 model day cap -$1,000 + band stop 2 ATR'] = daycap(model, 1000) + ov(band, stop(2))
    base, _ = C.simm(model + band, TK); kb = L.stats(base)
    rat = lambda k, h: k['yr' + h] / k['dd' + h]
    say(); say("=" * 150); say(f"EXIT  EXITS AIMED AT DRAWDOWN  exit_dd.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n 1 + 1, {TK} ticks; ratio = $/yr per $ max DD"); say("=" * 150)
    say(f"   B0 baseline                                  2013-19 DD ${kb['ddA']:,.0f} ratio {rat(kb, 'A'):.2f} Sharpe {kb['shA']:.2f} | 2019-26 DD ${kb['ddB']:,.0f} ratio {rat(kb, 'B'):.2f} Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr")
    for nm, net in V.items():
        s, _ = C.simm(net, TK); k = L.stats(s)
        sel = k['ddA'] < kb['ddA'] and rat(k, 'A') > rat(kb, 'A')
        ok = sel and k['ddB'] <= 0.9 * kb['ddB'] and rat(k, 'B') > rat(kb, 'B') and k['shB'] >= kb['shB'] - 0.05
        say(f"   {nm:44s} 2013-19 DD ${k['ddA']:,.0f} ratio {rat(k, 'A'):.2f} Sharpe {k['shA']:.2f} {'SEL' if sel else '   '}| 2019-26 DD ${k['ddB']:,.0f} ratio {rat(k, 'B'):.2f} "
            f"Sharpe {k['shB']:.2f} ${k['yrB']:,.0f}/yr | all-years DD ${k['dd']:,.0f}" + (' CONFIRMED' if ok else ''))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
