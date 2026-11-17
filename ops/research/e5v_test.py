r"""E5V checks on hand-made bars: gap detection, unfilled zones, resting stop / target fills, the snipe entry and the one-position rule.
    python e5v_test.py
"""
import os, sys
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import e5v_struct as ST

R = []


def rec(n, ok, d=''):
    R.append(bool(ok)); print(('PASS  ' if ok else 'FAIL  ') + n + (f'  [{d}]' if d else ''))


def era(o, h, l, c, t0=20000 * 1440 + 600):
    E = NS(); n = len(o); E.t = t0 + np.arange(n, dtype=np.int64); E.shift = np.zeros(n)
    E.o, E.h, E.l, E.c = (np.array(x, float) for x in (o, h, l, c)); E.md = E.t % 1440; E.dd = E.t // 1440
    E.bars = {1: ST.agg(E, 1)}; E.bars[1].atr = np.full(n, 1.0); E.fvg = {1: ST.fvgs(E.bars[1], 1)}
    return E


def main():
    o = [100, 100, 104, 106, 106, 105, 105]; h = [101, 104, 106, 107, 107, 106, 106]; l = [99, 100, 103, 105, 104, 104, 100]; c = [100, 104, 106, 106, 105, 105, 101]
    E = era(o, h, l, c); F = E.fvg[1]
    rec('a bullish gap is found where bar 3 low (103) is above bar 1 high (101), known at bar 3 close', len(F.t) >= 1 and F.typ[0] == 1 and F.lo[0] == 101 and F.hi[0] == 103 and F.t[0] == E.t[2] + 1,
        str(list(zip(F.t - E.t[0], F.typ, F.lo, F.hi))))
    W = NS(t=E.t, smin=np.minimum.accumulate(E.l[::-1])[::-1], smax=np.maximum.accumulate(E.h[::-1])[::-1])
    rec('the gap counts as filled once a later bar trades through its far edge (low 100 <= 101)', not (ST.after(W, F.t[:1], 'min') > F.lo[:1])[0])
    W2 = NS(t=E.t[:6], smin=np.minimum.accumulate(E.l[:6][::-1])[::-1], smax=np.maximum.accumulate(E.h[:6][::-1])[::-1])
    rec('and as unfilled before that', (ST.after(W2, F.t[:1], 'min') > F.lo[:1])[0])
    o = [100, 100, 100, 100]; h = [100.5, 103, 100, 100]; l = [99.5, 97, 100, 100]; c = [100, 100, 100, 100]; E = era(o, h, l, c)
    x = ST.sim_exit(E, int(E.t[0]), int(E.t[0]) + 4, 1.0, 102.0, 98.0)
    rec('stop and target inside one minute: the stop fills, at the stop price', x is not None and x[1] == 98.0 and x[2] == 'stop' and x[0] == E.t[1], str(x))
    h2 = [100.5, 102.0, 102.25, 100]; l2 = [99.5, 99, 99, 99]; E = era(o, h2, l2, c)
    x = ST.sim_exit(E, int(E.t[0]), int(E.t[0]) + 4, 1.0, 102.0, 90.0)
    rec('a resting target needs one tick of trade-through (102.00 touched = no fill, 102.25 = filled at 102)', x is not None and x[0] == E.t[2] and x[1] == 102.0 and x[2] == 'target', str(x))
    o3 = [100, 97, 100, 100]; l3 = [99.5, 96, 100, 100]; h3 = [100.5, 97.5, 100, 100]; E = era(o3, h3, l3, c)
    x = ST.sim_exit(E, int(E.t[0]), int(E.t[0]) + 4, 1.0, np.nan, 98.0)
    rec('a stop gapped through fills at the minute open (97), not at the stop', x is not None and x[1] == 97.0, str(x))
    E.c = np.array([100, 97, 100, 100], float)
    x = ST.sim_exit(E, int(E.t[0]), int(E.t[0]) + 4, 1.0, np.nan, 98.0, market=True)
    rec('market-order mode: a close beyond the level leaves at the next open', x is not None and x[1] == 100.0 and x[0] == E.t[2] - 1, str(x))
    x = ST.sim_exit(E, int(E.t[0]), int(E.t[0]) + 4, 1.0, np.nan, np.nan, (np.array([E.t[1]]), np.array([96.5])))
    rec('a trailing stop only acts from the minute its zone is known', x is not None and x[0] == E.t[1] and x[1] == 96.5, str(x))
    import e5v_snipe as SN
    T = np.array([[0, 10, 20, 1, 100, 101, 1], [1, 15, 18, 1, 100, 101, 1], [2, 21, 30, 1, 100, 101, 1]], float)
    rec('one position at a time: a signal while a trade is open is skipped', list(SN.one_at_a_time(T)) == [True, False, True])
    ok = all(R); print('ALL PASS' if ok else f'{R.count(False)} FAILED'); return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
