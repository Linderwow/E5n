r"""DDL (DDL_PLAN.txt): drawdown levers from the book's own equity curve.
    python ddl_levers.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import say


def sizes(x, rule, p):
    v = x.values; eq = np.cumsum(v); n = len(v); s = np.ones(n)
    prev_eq = np.r_[0.0, eq[:-1]]; peak = np.maximum.accumulate(np.r_[0.0, eq])[:-1]; dd = peak - prev_eq
    if rule in ('ma', 'mah'):
        ma = pd.Series(prev_eq).rolling(p, min_periods=p).mean().values
        s = np.where(np.isnan(ma) | (prev_eq >= ma), 1.0, 0.0 if rule == 'ma' else 0.5)
    elif rule == 'pause': s = np.where(dd >= p, 0.0, 1.0)
    elif rule == 'half': s = np.where(dd >= p, 0.5, 1.0)
    elif rule == 'up': s = np.where(dd >= p, 1.5, 1.0)
    elif rule in ('week', 'month'):
        key = x.index.to_period('W' if rule == 'week' else 'M'); acc = 0.0; cur = None; stop = False
        for i in range(n):
            if key[i] != cur: cur = key[i]; acc = 0.0; stop = False
            s[i] = 0.0 if stop else 1.0; acc += s[i] * v[i]
            if acc <= -p: stop = True
    return pd.Series(s * v, index=x.index)


def k(x):
    a, b = L.halves(x); f = lambda y: (y.mean() * 252 / L.mdd(y) if L.mdd(y) > 0 else np.inf, L.X.sharpe(y), L.mdd(y), y.mean() * 252)
    return f(a), f(b)


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'DDL_RESULTS.txt')
    mod, _ = C.simm(C.path(M.g5, M.g15, 'set', M.grid)); band, _ = C.simm(M.band); books = {'backtest': mod + band, 'fair': mod * 2 / 3 + band}
    trials = [('E1', 'ma', 10), ('E2', 'ma', 20), ('E3', 'ma', 50), ('E4', 'mah', 20), ('P1', 'pause', 1000), ('P2', 'pause', 2000), ('P3', 'pause', 3000),
              ('P4', 'half', 1000), ('P5', 'half', 2000), ('W1', 'week', 500), ('W2', 'week', 1000), ('M1', 'month', 1500), ('U1', 'up', 2000)]
    say(); say("=" * 150); say(f"DDL  DRAWDOWN LEVERS FROM THE BOOK'S OWN EQUITY CURVE  ddl_levers.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  1 + 1 micro, live cost; Calmar = $/yr / max DD"); say("=" * 150)
    base = {nm: k(b) for nm, b in books.items()}
    fmt = lambda t: f"Calmar {t[0]:4.2f} Sharpe {t[1]:4.2f} DD ${t[2]:5,.0f} ${t[3]:6,.0f}/yr"
    say(f"   B0 baseline           2013-19 {fmt(base['backtest'][0])} | 2019-26 {fmt(base['backtest'][1])} | fair 2019-26 {fmt(base['fair'][1])}")
    conf = []
    for nm, rule, p in trials:
        r = {b: k(sizes(x, rule, p)) for b, x in books.items()}; A, B = r['backtest']; FA, FB = r['fair']; bA, bB = base['backtest']; fB = base['fair'][1]
        sel = A[0] > bA[0] and A[1] >= bA[1]
        ok = sel and B[0] > bB[0] and B[1] > bB[1] and B[2] <= bB[2] and FB[0] > fB[0] and FB[1] > fB[1] and FB[2] <= fB[2]
        if ok: conf.append(nm)
        say(f"   {nm} {rule:5s} {p:>5}     2013-19 {fmt(A)} {'SEL' if sel else '   '}| 2019-26 {fmt(B)} | fair 2019-26 {fmt(FB)} {'CONFIRMED' if ok else ''}")
    say(f"   CONFIRMED: {', '.join(conf) if conf else 'none'}  (trial count 10,503)"); say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
