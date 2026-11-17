r"""BR (BR_PLAN.txt): stop / target brackets (1:2, 1:3) on each leg, 4.5 ticks a side.
    python br_brackets.py
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
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'BR_RESULTS.txt')
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band.copy()
    ctx = NS(atr=slot('atr'), vw=slot('vw'), half=(slot('ub') - slot('lb')) / 2, mdl=model)
    br = lambda s, r: (lambda S: ((S.op <= -s * S.A) | (S.op >= r * s * S.A), S.op * 0 > 1))
    def ov(leg, rule):
        kb = M.band; M.band = leg
        try: return Q.overlay(rule, ctx)
        finally: M.band = kb
    V = {}
    for s in (1.0, 1.5, 2.0):
        for r in (2, 3): V[f'BB band stop {s:g} ATR, target {r * s:g} ATR (1:{r})'] = model + ov(band, br(s, r))
    for s in (1.0, 1.5):
        for r in (2, 3): V[f'MB model stop {s:g} ATR, target {r * s:g} ATR (1:{r})'] = ov(model, br(s, r)) + band
    base, _ = C.simm(model + band, TK); kb = L.stats(base); rat = lambda k, h: k['yr' + h] / k['dd' + h]
    atr = slot('atr'); keep = np.asarray(M.keep); px = M.o930[:, None]
    risk = lambda s: float(np.nanmedian((s * atr / px * 27875 * 2)[keep][:, M.grid]))
    say(); say("=" * 150); say(f"BR  STOP / TARGET BRACKETS  br_brackets.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n 1 + 1, {TK} ticks; ratio = $/yr per $ max DD"); say("=" * 150)
    say("   $ at risk per micro at the stop (median, MNQ at 27,875): " + ", ".join(f"{s:g} ATR = ${risk(s):,.0f}" for s in (1.0, 1.5, 2.0)))
    say(f"   B0 baseline                                    2013-19 DD ${kb['ddA']:,.0f} ratio {rat(kb, 'A'):.2f} Sharpe {kb['shA']:.2f} | 2019-26 DD ${kb['ddB']:,.0f} ratio {rat(kb, 'B'):.2f} Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr")
    for nm, net in V.items():
        s_, _ = C.simm(net, TK); k = L.stats(s_)
        sel = k['ddA'] < kb['ddA'] and rat(k, 'A') > rat(kb, 'A')
        ok = sel and k['ddB'] <= 0.9 * kb['ddB'] and rat(k, 'B') > rat(kb, 'B') and k['shB'] >= kb['shB'] - 0.05
        say(f"   {nm:46s} 2013-19 DD ${k['ddA']:,.0f} ratio {rat(k, 'A'):.2f} Sharpe {k['shA']:.2f} {'SEL' if sel else '   '}| 2019-26 DD ${k['ddB']:,.0f} ratio {rat(k, 'B'):.2f} "
            f"Sharpe {k['shB']:.2f} ${k['yrB']:,.0f}/yr" + (' CONFIRMED' if ok else ''))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
