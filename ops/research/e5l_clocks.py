r"""Clock diagnostics (information, not trials): the model leg on clocks offset by 0-4 minutes, and the 15-minute check on every clock.
    python e5l_clocks.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import say


def main():
    t0 = time.time(); C.load(); M = C.M
    sets = (('deployed models (trained on the :00/:05 clock)', M.g5, M.g15), ('models retrained on every-minute rows', M.a5, M.a15))
    clk = lambda o: (M.dec % 5 == o) & (M.dec >= 15)
    say(); say("=" * 150); say(f"CLOCK DIAGNOSTICS  e5l_clocks.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (information, not trials) $ per micro at today's price, live cost"); say("=" * 150)
    say("1. The model leg alone, deciding every 5 minutes on a clock offset by 0-4 minutes (offset 0 = as deployed):")
    for lab, f5, f15 in sets:
        say(lab + ":")
        for o in range(5): say(L.line(f"   offset {o} min", C.simm(C.path(f5, f15, 'set', clk(o)))[0]))
        say(L.line("   average of the 5 clocks", C.simm(sum(C.path(f5, f15, 'set', clk(o)) for o in range(5)) / 5)[0]))
    say("2. The 15-minute check on every clock: model leg and book, with / without the check:")
    nochk = lambda f5: np.where(np.isnan(f5), np.nan, np.sign(f5)); up = dn = both = tot = 0
    for lab, f5, f15 in sets:
        for o in range(5):
            pw = C.path(f5, f15, 'set', clk(o)); po = C.path(f5, nochk(f5), 'set', clk(o))
            kw, ko, kbw, kbo = (L.stats(C.simm(x)[0]) for x in (pw, po, pw + M.band, po + M.band))
            tot += 1; up += kbw['sh'] > kbo['sh']; dn += kbw['dd'] < kbo['dd']; both += kbw['sh'] > kbo['sh'] and kbw['dd'] < kbo['dd']
            say(f"   {lab[:38]:38s} offset {o}: leg Sharpe {kw['sh']:.2f} / {ko['sh']:.2f} | leg max DD ${kw['dd']:6,.0f} / ${ko['dd']:6,.0f} | leg $/yr {kw['yr']:6,.0f} / {ko['yr']:6,.0f} | "
                f"BOOK Sharpe {kbw['sh']:.2f} / {kbo['sh']:.2f}, max DD ${kbw['dd']:6,.0f} / ${kbo['dd']:6,.0f}")
    say(f"   with the check the book's Sharpe is higher on {up} of {tot} combinations, its max drawdown lower on {dn} of {tot}, both on {both} of {tot}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
