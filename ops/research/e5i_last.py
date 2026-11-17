import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5d_more as D
from e5b_lab import G, say
X = L.X; ND = L.ND; U = L.U


def streak_stop(m, nmax, ticks):
    c1 = X.SIDE(ticks) * U; out = np.zeros_like(m); live = np.ones(G.n, bool); pos = np.zeros(G.n); ent = np.zeros(G.n); run = np.zeros(G.n)
    for j in range(ND):
        want = np.where(live, m[:, j], 0.0); ch = want != pos; closing = ch & (pos != 0)
        pnl = pos * (G.px[:, j] - ent) / G.o930 * U - 2 * c1
        run = np.where(closing, np.where(pnl <= 0, run + 1, 0.0), run)
        live &= ~(run >= nmax); want = np.where(live, want, 0.0)
        ent = np.where(want != pos, G.px[:, j], ent); pos = want; out[:, j] = pos
    return out


def range_stop(m, mult):
    T = G.T; px = G.px; hi = np.maximum.accumulate(px, axis=1); lo = np.minimum.accumulate(px, axis=1)
    hi = np.maximum(hi, G.o930[:, None]); lo = np.minimum(lo, G.o930[:, None])
    dayrng = pd.Series(np.maximum(px.max(1), np.maximum(G.exit, G.o930)) - np.minimum(px.min(1), np.minimum(G.exit, G.o930)))
    avg = dayrng.shift(1).rolling(14, min_periods=10).mean().values
    with np.errstate(invalid='ignore'): big = (hi - lo) > mult * avg[:, None]
    halted = np.maximum.accumulate(big, axis=1)
    return np.where(halted, 0.0, m)


def main():
    t0 = time.time(); L.setup(); m, b = G.model, G.band; base = m + b
    say(); say("=" * 160); say(f"E5B AMENDMENT F  e5i_last.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  model-leg day protection, to-the-close leg; $ per micro at today's price, live cost on net orders"); say("=" * 160)
    book = lambda t: L.sim(m, b, ticks=t)
    MR = [dict(name=f'MR{k}', desc=f'model leg stops for the day after {n} losing trades in a row', fn=(lambda n: lambda t: L.sim(net=streak_stop(m, n, t) + b, ticks=t))(n)) for k, n in ((1, 3), (2, 4), (3, 5))]
    MR += [dict(name=f'MR{k}', desc=f"model leg stops once the day's range exceeds {mult} x its 14-day average", fn=(lambda mp: lambda t: L.sim(net=mp + b, ticks=t))(range_stop(m, mult))) for k, mult in ((4, 2.0), (5, 1.5))]
    D.judge('MR', MR, book, 'the netted book B0')
    fc = np.load(os.path.join(X.CACHE, 'e5c_gc.npy')); dec_m = np.arange(5, 386); fc = fc[:, (dec_m % 5 == 0) & (dec_m >= 15)]
    a = np.abs(fc[G.keep]); med = pd.Series(np.nanmedian(a, axis=1)).expanding(min_periods=20).median().shift(1).values; mf = np.full(G.n, np.nan); mf[np.flatnonzero(G.keep)] = med
    NC = []
    for nm, cad, k, once, lab in (('NC1', 30, 1.0, False, 'every 30 minutes, |forecast| above its usual size'), ('NC2', 30, 2.0, False, 'every 30 minutes, |forecast| above 2 x its usual size'),
                                  ('NC3', 30, 1.0, True, 'one decision at 10:00, held to the close')):
        pos = np.zeros(G.n); lp = np.zeros((G.n, ND))
        for j in range(ND):
            dj = G.dec[j]
            if (dj == 30) if once else (dj >= 30 and dj % cad == 0):
                v = fc[:, j]
                with np.errstate(invalid='ignore'): t = np.where(v > k * mf, 1.0, np.where(v < -k * mf, -1.0, 0.0))
                pos = np.where(np.isnan(v) | np.isnan(mf), pos, t)
            lp[:, j] = pos
        NC.append(dict(name=nm, desc=f'new leg: to-the-close forecast, {lab}', fn=(lambda lp: lambda t: L.sim(net=base + lp, ticks=t))(lp), leg=L.sim(net=lp)))
    D.judge('NC', NC, book, 'the netted book B0')
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
