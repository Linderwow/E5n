r"""E5F - free order-flow pilot on the trade data already owned (E5F_PLAN.txt).
    python e5f_pilot.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import databento as db
import e5x_lab as X

OUT = os.path.join(HERE, 'E5F_RESULTS.txt'); LINES = []
FILES = {'MNQ': (os.path.join(X.PATHS.TRADES, 'mnq_trades_2026-01-01_2026-04-16.dbn.zst'), '2026-01-02', '2026-04-09'),
         'MES': (os.path.join(X.PATHS.TRADES, 'MES_trades_2025-09_2026-06.dbn.zst'), '2025-09-02', '2026-05-29')}


def say(s=''):
    print(s, flush=True); LINES.append(s)


def minute_flow(path):
    parts = []
    for df in db.DBNStore.from_file(path).to_df(count=5_000_000):
        t = df.index.tz_convert('America/New_York').tz_localize(None).floor('min'); sz = df['size'].values.astype(float)
        buy = np.where(df.side.values == 'B', sz, 0.0); sell = np.where(df.side.values == 'A', sz, 0.0); big = sz >= 10
        parts.append(pd.DataFrame({'m': t, 'buy': buy, 'sell': sell, 'lbuy': buy * big, 'lsell': sell * big, 'chg': df.price.values}).groupby('m').agg(
            buy=('buy', 'sum'), sell=('sell', 'sum'), lbuy=('lbuy', 'sum'), lsell=('lsell', 'sum'), first=('chg', 'first'), last=('chg', 'last')))
    g = pd.concat(parts); g = g.groupby(level=0).agg(buy=('buy', 'sum'), sell=('sell', 'sum'), lbuy=('lbuy', 'sum'), lsell=('lsell', 'sum'), first=('first', 'first'), last=('last', 'last'))
    return g.sort_index()


def window(g, ends, w, a, b):
    m = g.index.values.astype('datetime64[m]'); ca = np.r_[0.0, np.cumsum(g[a].values)]; cb = np.r_[0.0, np.cumsum(g[b].values)]
    j1 = np.searchsorted(m, ends, side='left'); j0 = np.searchsorted(m, ends - np.timedelta64(w, 'm'), side='left')
    A_ = ca[j1] - ca[j0]; B_ = cb[j1] - cb[j0]
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(A_ + B_ > 0, (A_ - B_) / (A_ + B_), np.nan)


def ols_t(y, f, x, day):
    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(x); y, f, x, day = y[ok], f[ok], x[ok], day[ok]
    x = (x - x.mean()) / x.std(); Xm = np.c_[np.ones(len(y)), f / f.std(), x]
    bread = np.linalg.inv(Xm.T @ Xm); beta = bread @ Xm.T @ y; e = y - Xm @ beta
    meat = np.zeros((3, 3))
    for d in np.unique(day):
        s = Xm[day == d].T @ e[day == d]; meat += np.outer(s, s)
    V = bread @ meat @ bread; return float(beta[2] / np.sqrt(V[2, 2])), int(ok.sum()), float(np.corrcoef(x, y)[0, 1])


def main():
    t0 = time.time(); X.setup()
    say(); say("=" * 150); say(f"E5F ORDER-FLOW PILOT  e5f_pilot.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  trades already owned; information test only (E5F_PLAN.txt)"); say("=" * 150)
    res = {}
    for mk, (path, a, b) in FILES.items():
        g = minute_flow(path); g['net'] = g.buy - g.sell; g['ret'] = g['last'] - g['first']
        san = float(g.net.corr(g.ret))
        bk, f = (X.NQ, X.F_NQ) if mk == 'MNQ' else (X.ES, X.F_ES)
        T = bk.T; sel = (T.date >= a).values & (T.date <= b).values
        T = T[sel].reset_index(drop=True); f = f[sel]; y = T.y.values; day = T.date.values.astype('datetime64[D]').astype(np.int64)
        ends = T.date.values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]')
        meas = {'OF1': window(g, ends, 1, 'buy', 'sell'), 'OF5': window(g, ends, 5, 'buy', 'sell'), 'OF15': window(g, ends, 15, 'buy', 'sell'), 'LT5': window(g, ends, 5, 'lbuy', 'lsell')}
        say(f"{mk} {a} .. {b}: {len(g):,} minutes of trades, {T.date.nunique()} sessions, {len(T):,} E5n decisions; sanity: buyer-initiated minus seller-initiated volume vs the same "
            f"minute's price change, correlation {san:+.2f} (must be clearly positive)")
        res[mk] = {}
        for nm, x in meas.items():
            tt, n, ic = ols_t(y, f, x, day); pl = []
            for s in (1, 2, 3):
                xs = pd.Series(x).shift(75 * s).values; pl.append(ols_t(y, f, xs, day)[0])
            res[mk][nm] = (tt, pl)
            say(f"   {nm:5s} t {tt:+.2f} beyond E5n's forecast (n {n:,}; plain correlation with the next 5 minutes {ic:+.3f}) | placebos (1 / 2 / 3 sessions earlier) t " + ' / '.join(f"{p:+.2f}" for p in pl))
    hit = [nm for nm in ('OF1', 'OF5', 'OF15', 'LT5') if abs(res['MNQ'][nm][0]) >= 2.5 and np.sign(res['MES'][nm][0]) == np.sign(res['MNQ'][nm][0]) and abs(res['MES'][nm][0]) >= 1.5
           and all(abs(p) < 2.0 for p in res['MNQ'][nm][1] + res['MES'][nm][1])]
    say(f"VERDICT: " + (f"POSITIVE PILOT - {', '.join(hit)} add information beyond E5n's forecast on both markets" if hit else "NEGATIVE PILOT - no measure passes on both markets"))
    say(f"run {time.time() - t0:.0f} s")
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    main()
