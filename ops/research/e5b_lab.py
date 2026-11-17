r"""E5B - improving the two-leg book (E5B_PLAN.txt). Both legs on one 5-minute timeline, orders netted.
    python e5b_lab.py audit     the joint simulator against each leg's own research series
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_band as B
import e5n_band_test as BT
import e5n_train as TR

OUT = os.path.join(HERE, 'E5B_RESULTS.txt'); ND = 75; U = X.U; T0 = X.T0; JOIN = X.JOIN
LINES = []


class G:
    pass


def say(s=''):
    print(s, flush=True); LINES.append(s)


def flush():
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")
    LINES.clear()


def setup(band=True):
    X.setup(); T = X.NQ.T; n = len(T) // ND; G.T = T; G.n = n
    G.dates = pd.DatetimeIndex(T.date.values[::ND]); G.keep = np.asarray(G.dates.year >= 2013); G.isA = np.asarray(G.dates <= JOIN)
    G.row = {d: i for i, d in enumerate(G.dates)}
    m = lambda a: np.asarray(a, float).reshape(n, ND)
    G.m = m; G.f5 = m(X.F_NQ)
    for nm, fn in (('f15', 'x5_nq'), ('f30', 'c5_nq'), ('f60', 'c4_nq')): setattr(G, nm, m(np.load(os.path.join(X.CACHE, fn + '.npy'))))
    G.px = m(T.entry.values); G.exit = T.exitp.values.reshape(n, ND)[:, -1]; G.o930 = T.open930.values[::ND]
    G.dpx = np.diff(np.c_[G.px, G.exit], axis=1); G.dec = T.dec.values[:ND].astype(int)
    G.model = model_path()
    if band:
        G.df5 = {'NQ': BT.load('NQ'), 'MNQ': BT.load('MNQ')}
        G.band, G.diag = band_path(diag=True)


def model_path(f=None, thr=T0, check='f15', mask=None, maxrun=None, runmask=None, g=None):
    g = g or G
    f = g.f5 if f is None else f; chk = getattr(g, check) if isinstance(check, str) else check
    pos = np.zeros(g.n); out = np.zeros((g.n, ND)); run = np.zeros(g.n); prev = np.zeros(g.n)
    for j in range(ND):
        fj = f[:, j]
        with np.errstate(invalid='ignore'):
            t = np.where(fj > thr, 1.0, np.where(fj < -thr, -1.0, 0.0))
            if chk is not None: t = np.where(chk[:, j] * t > 0, t, 0.0)
        if mask is not None: t = t * mask[:, j]
        ok = ~np.isnan(fj)
        if maxrun is not None:
            run = np.where(ok, np.where((t == prev) & (t != 0), run + 1, np.where(t != 0, 1.0, 0.0)), run); prev = np.where(ok, t, prev)
            lim = run > maxrun if runmask is None else (run > maxrun) & (runmask[:, j] > 0)
            t = np.where(lim, 0.0, t)
        pos = np.where(ok, t, pos); out[:, j] = pos
    return out


def band_path(diag=False, **hooks):
    out = np.zeros((G.n, ND)); D = {} if diag else None
    for nm, sel in (('NQ', lambda d: d <= JOIN), ('MNQ', lambda d: d > JOIN)):
        marks = []; B.run(G.df5[nm], live=True, warmup=B.LOOK, bars=marks, **hooks)
        mk = pd.concat(marks, ignore_index=True); day = mk.time.dt.normalize(); dec = (mk.time.dt.hour * 60 + mk.time.dt.minute - 570).values
        i = day.map(G.row).values; ok = ~np.isnan(i) & np.asarray(sel(day)) & (dec >= 15) & (dec <= 385)
        ii = i[ok].astype(int); jj = (dec[ok] - 15) // 5
        out[ii, jj] = mk.target.values[ok]
        if diag:
            for c in ('q', 'close', 'ub', 'lb', 'vw', 'ema', 'atr'):
                a = D.setdefault(c, np.full((G.n, ND), np.nan)); a[ii, jj] = mk[c].values[ok]
    return (out, D) if diag else out


def sim(m=None, b=None, cm=1.0, cb=1.0, ticks=2.35, net=None, parts=False, scale=None, g=None):
    g = g or G
    net = cm * m + cb * b if net is None else net
    if scale is not None: net = net * scale[:, None]
    pnl = (net * g.dpx).sum(1); sides = np.abs(np.diff(np.c_[np.zeros(g.n), net, np.zeros(g.n)], axis=1)).sum(1)
    r = (pnl / g.o930 - sides * X.SIDE(ticks)) * U
    s = pd.Series(r[g.keep], index=g.dates[g.keep])
    return (s, float(sides[g.keep].sum() / 2), net) if parts else s


def grid(book, f5, f15):
    class Q: pass
    T = book.T; q = Q(); q.n = len(T) // ND; mm = lambda a: np.asarray(a, float).reshape(q.n, ND)
    q.dates = pd.DatetimeIndex(T.date.values[::ND]); q.keep = np.asarray(q.dates.year >= 2013); q.f5 = mm(f5); q.f15 = mm(f15)
    q.px = mm(T.entry.values); q.exit = T.exitp.values.reshape(q.n, ND)[:, -1]; q.o930 = T.open930.values[::ND]; q.dpx = np.diff(np.c_[q.px, q.exit], axis=1)
    return q


def mdd(x):
    eq = np.cumsum(np.asarray(x, float)); return float((np.maximum(np.maximum.accumulate(eq), 0) - eq).max()) if len(eq) else 0.0


def halves(s):
    return s[s.index <= JOIN], s[s.index > JOIN]


def stats(s):
    a, b = halves(s)
    return dict(yr=s.mean() * 252, sh=X.sharpe(s), dd=mdd(s), shA=X.sharpe(a), shB=X.sharpe(b), ddA=mdd(a), ddB=mdd(b), yrA=a.mean() * 252, yrB=b.mean() * 252, worst=float(s.min()))


def line(lab, s, extra=''):
    k = stats(s)
    return (f"{lab:58s} Sharpe {k['shA']:5.2f} / {k['shB']:5.2f} / {k['sh']:5.2f} | $/yr {k['yrA']:7,.0f} / {k['yrB']:7,.0f} / {k['yr']:7,.0f} | "
            f"max DD {k['ddA']:6,.0f} / {k['ddB']:6,.0f} / {k['dd']:6,.0f} | worst day {k['worst']:7,.0f}{extra}")


def audit():
    t0 = time.time(); setup()
    say(); say("=" * 150); say(f"E5B joint simulator audit  e5b_lab.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  $ per micro at today's price, 2.35 ticks + $0.61 a side on NET orders"); say("=" * 150)
    e15, _ = X.NQ.run(X.F_NQ, X.r_agree(G.f15.reshape(-1)), 2.35); mine = sim(G.model, G.band, 1, 0)
    d1 = float(np.abs(mine.reindex(e15.index) - e15).max())
    say(f"model leg (with the 15-minute check) vs e5x_lab: max |daily diff| ${d1:.2e} -> {'PASS' if d1 < 1e-6 else 'FAIL'}")
    x3 = pd.read_csv(os.path.join(X.CACHE, 'x3_daily.csv'), index_col=0, parse_dates=True); bl = sim(G.model, G.band, 0, 1)
    c = pd.concat([bl.rename('mine'), (x3.r_close * U).rename('ref')], axis=1, join='inner')
    say(f"band leg on the model's price grid vs its own research series: {len(c):,} sessions, correlation {c.mine.corr(c.ref):.4f}, "
        f"$/yr {c.mine.mean() * 252:,.0f} vs {c.ref.mean() * 252:,.0f} (research costed 1 tick + $0.50 a side; here 2.35 ticks + $0.61; live rules skip early-close days)")
    book, trips, _ = sim(G.model, G.band, parts=True); sep = sim(G.model, G.band, 1, 0) + sim(G.model, G.band, 0, 1)
    say(line('BOOK netted (base B0)', book, f" | {trips / (len(book) / 252):,.0f} net round trips/yr"))
    say(line('  legs costed separately (no netting)', sep)); say(line('  model leg with check', sim(G.model, G.band, 1, 0))); say(line('  band leg', sim(G.model, G.band, 0, 1)))
    say(f"columns: 2013-19 / 2019-26 / all. run {time.time() - t0:.0f} s"); flush()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'audit': audit()
    else: print(__doc__)
