import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5d_more as D
from e5b_lab import G, say


def main():
    t0 = time.time(); L.setup(); m, b = G.model, G.band
    say(); say("=" * 160); say(f"E5B AMENDMENT D (band part)  e5f_band_ens.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  band-leg copies on staggered clocks / look-backs; $ per micro at today's price, live cost on net orders"); say("=" * 160)
    book = lambda t: L.sim(m, b, ticks=t)
    c2 = frozenset(range(615, 946, 30)); l2 = frozenset((735, 765, 795))
    b_clock = L.band_path(checks=c2, lunch=l2); b_look = L.band_path(look=28); b_both = L.band_path(look=28, checks=c2, lunch=l2)
    say(L.line('band leg as deployed (alone)', L.sim(m, b, 0, 1))); say(L.line('band leg on the :15/:45 clock (alone)', L.sim(m, b_clock, 0, 1)))
    say(L.line('band leg with a 28-session look-back (alone)', L.sim(m, b_look, 0, 1))); say(L.line('band leg, :15/:45 clock and 28 sessions (alone)', L.sim(m, b_both, 0, 1)))
    one = L.sim(m, b, 0, 1)
    for lab, x in (('second clock', b_clock), ('28-session look-back', b_look)):
        say(f"   daily correlation of the deployed band leg with the {lab} copy: {one.corr(L.sim(m, x, 0, 1)):.2f}")
    B = [dict(name='B1', desc='band leg = half on :00/:30, half on :15/:45', fn=lambda t: L.sim(net=m + 0.5 * b + 0.5 * b_clock, ticks=t)),
         dict(name='B2', desc='band leg = half with 14 sessions, half with 28', fn=lambda t: L.sim(net=m + 0.5 * b + 0.5 * b_look, ticks=t)),
         dict(name='B3', desc='band leg = four quarter-size copies (2 clocks x 2 look-backs)', fn=lambda t: L.sim(net=m + 0.25 * (b + b_clock + b_look + b_both), ticks=t))]
    D.judge('B', B, book, 'the netted book B0')
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
