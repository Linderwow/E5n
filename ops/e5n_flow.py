r"""Order-flow pilot on E5n's own tick log (E5F_PLAN.txt; re-run rule: at about 120 sessions, buy the trade history only if the large-trade
imbalance reaches t >= 2.5 beyond E5n's forecast with every placebo |t| < 2). Called by the nightly job; it keeps per-day minute aggregates.
    python e5n_flow.py status            sessions recorded so far
    python e5n_flow.py run               run the pilot now on whatever has been recorded (also: the recorder's tapes when asked)
    python e5n_flow.py tapes YYYY-MM-DD  aggregate one recorder tape day (pipeline check on the recorder's format)
"""
import os, sys, glob, json, datetime as dt
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_paths as PATHS

NT_E5N = os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8', 'e5n'); TICKS = os.path.join(NT_E5N, 'ticks')
REC = PATHS.TAPE; DATA = os.path.join(HERE, 'data'); AGG = os.path.join(DATA, 'flow_minutes'); OUT = os.path.join(DATA, 'e5n_flow_pilot.json')
NEED = 120; EVERY = 20; LARGE = 10; MEASURES = ('OF1', 'OF5', 'OF15', 'LT5')


def classify(price, bid, ask):
    side = np.zeros(len(price))
    with np.errstate(invalid='ignore'):
        side[np.isfinite(ask) & (price >= ask)] = 1.0; side[np.isfinite(bid) & (price <= bid) & (side == 0)] = -1.0
    tick = np.sign(np.diff(price, prepend=price[0])); tick = pd.Series(np.where(tick == 0, np.nan, tick)).ffill().fillna(0.0).values
    return np.where(side != 0, side, tick)


def minutes_from(df, t):
    sz = df['size'].values.astype(float); side = classify(df.price.values.astype(float), df.bid.values.astype(float), df.ask.values.astype(float))
    buy = np.where(side > 0, sz, 0.0); sell = np.where(side < 0, sz, 0.0); big = sz >= LARGE; m = t.dt.floor('min')
    g = pd.DataFrame({'m': m, 'buy': buy, 'sell': sell, 'lbuy': buy * big, 'lsell': sell * big, 'p': df.price.values.astype(float)}).groupby('m').agg(
        buy=('buy', 'sum'), sell=('sell', 'sum'), lbuy=('lbuy', 'sum'), lsell=('lsell', 'sum'), first=('p', 'first'), last=('p', 'last'))
    return g


def own_days():
    return sorted({os.path.basename(f)[:10] for f in glob.glob(os.path.join(TICKS, 'MNQ_*', '*.csv'))})


def aggregate_own(day):
    out = os.path.join(AGG, f'{day}.parquet')
    files = sorted(glob.glob(os.path.join(TICKS, 'MNQ_*', f'{day}.csv')))
    if not files: return None
    if os.path.exists(out) and os.path.getmtime(out) >= max(os.path.getmtime(f) for f in files): return pd.read_parquet(out)
    parts = []
    for f in files:
        x = pd.read_csv(f, on_bad_lines='skip')
        for c in ('price', 'size', 'bid', 'ask'): x[c] = pd.to_numeric(x[c], errors='coerce')
        x = x.dropna(subset=['price', 'size']); t = pd.to_datetime(x.time_et, errors='coerce'); ok = t.notna().values
        parts.append(minutes_from(x[ok].reset_index(drop=True), t[ok].reset_index(drop=True)))
    g = pd.concat(parts).groupby(level=0).sum() if len(parts) > 1 else parts[0]
    os.makedirs(AGG, exist_ok=True); g.to_parquet(out); return g


def aggregate_tape(day):
    f = sorted(glob.glob(os.path.join(REC, 'MNQ_*', f'{day}_tape.parquet')))
    if not f: return None
    parts = []
    import pyarrow.parquet as pq
    for b in pq.ParquetFile(f[-1]).iter_batches(batch_size=5_000_000, columns=['kind', 'feed_utc', 'price', 'size', 'bid', 'ask']):
        ch = b.to_pandas()
        ch = ch[ch.kind == 'T'].reset_index(drop=True)
        if not len(ch): continue
        t = pd.to_datetime(ch.feed_utc).dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None)
        parts.append(minutes_from(ch, t))
    if not parts: return None
    g = pd.concat(parts); g = g.groupby(level=0).agg(buy=('buy', 'sum'), sell=('sell', 'sum'), lbuy=('lbuy', 'sum'), lsell=('lsell', 'sum'), first=('first', 'first'), last=('last', 'last'))
    return g


def window(g, ends, w, a, b):
    m = g.index.values.astype('datetime64[m]'); ca = np.r_[0.0, np.cumsum(g[a].values)]; cb = np.r_[0.0, np.cumsum(g[b].values)]
    j1 = np.searchsorted(m, ends, side='left'); j0 = np.searchsorted(m, ends - np.timedelta64(w, 'm'), side='left')
    A_ = ca[j1] - ca[j0]; B_ = cb[j1] - cb[j0]
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(A_ + B_ > 0, (A_ - B_) / (A_ + B_), np.nan)


def ols_t(y, f, x, day):
    ok = np.isfinite(y) & np.isfinite(f) & np.isfinite(x)
    if ok.sum() < 200 or np.std(x[ok]) == 0: return float('nan'), int(ok.sum())
    y, f, x, day = y[ok], f[ok], x[ok], day[ok]; x = (x - x.mean()) / x.std(); Xm = np.c_[np.ones(len(y)), f / f.std(), x]
    bread = np.linalg.inv(Xm.T @ Xm); beta = bread @ Xm.T @ y; e = y - Xm @ beta; meat = np.zeros((3, 3))
    for d in np.unique(day):
        s = Xm[day == d].T @ e[day == d]; meat += np.outer(s, s)
    V = bread @ meat @ bread; return float(beta[2] / np.sqrt(V[2, 2])), int(ok.sum())


def pilot(T, f5, minutes):
    g = minutes.sort_index(); g = g[~g.index.duplicated()]; days = set(g.index.normalize())
    keep = np.array([pd.Timestamp(d) in days for d in pd.to_datetime(T.date)]); T = T[keep].reset_index(drop=True); f5 = np.asarray(f5)[keep]
    if not len(T): return dict(sessions=0)
    y = T.y.values; day = pd.to_datetime(T.date).values.astype('datetime64[D]').astype(np.int64)
    ends = pd.to_datetime(T.date).values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]')
    meas = {'OF1': window(g, ends, 1, 'buy', 'sell'), 'OF5': window(g, ends, 5, 'buy', 'sell'), 'OF15': window(g, ends, 15, 'buy', 'sell'), 'LT5': window(g, ends, 5, 'lbuy', 'lsell')}
    net = (g.buy - g.sell); san = float(net.corr(g['last'] - g['first']))
    res = dict(sessions=int(T.date.nunique()), decisions=int(len(T)), sanity=round(san, 3), measures={})
    for nm, x in meas.items():
        tt, n = ols_t(y, f5, x, day); pl = [ols_t(y, f5, pd.Series(x).shift(75 * s).values, day)[0] for s in (1, 2, 3)]
        res['measures'][nm] = dict(t=round(tt, 2), n=n, placebo=[round(p, 2) for p in pl])
    lt = res['measures']['LT5']; pls = [p for m_ in res['measures'].values() for p in m_['placebo'] if p == p]
    res['buy_history'] = bool(res['sessions'] >= NEED and san > 0.3 and lt['t'] == lt['t'] and lt['t'] >= 2.5 and all(abs(p) < 2.0 for p in pls))
    res['verdict'] = ('BUY THE TRADE HISTORY - large-trade imbalance t >= 2.5 beyond E5n on its own ticks, placebos clean' if res['buy_history']
                      else (f"collecting - {res['sessions']} of {NEED} sessions" if res['sessions'] < NEED else 'do not buy - the large-trade imbalance does not reach t 2.5 on E5n\'s own ticks'))
    return res


def nightly(T, f5, now):
    days = own_days(); full = [d for d in days if pd.Timestamp(d).weekday() < 5]
    for d in full: aggregate_own(d)
    prev = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    sess = len([d for d in full if pd.Timestamp(d) in set(pd.to_datetime(T.date).dt.normalize())])
    if sess >= NEED and (prev.get('sessions', 0) == 0 or sess >= prev.get('sessions', 0) + EVERY or not prev.get('run')):
        mins = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(AGG, '*.parquet')))])
        res = pilot(T, f5, mins); res.update(run=now.strftime('%Y-%m-%d %H:%M'))
        os.makedirs(DATA, exist_ok=True); json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1); return res
    return dict(prev, sessions_recorded=sess, need=NEED, verdict=prev.get('verdict') if prev.get('run') else f"collecting - {sess} of {NEED} sessions of E5n ticks")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'status': print(f"E5n tick days: {len(own_days())} ({', '.join(own_days()[-5:])})")
    elif a == 'tapes' and len(sys.argv) > 2:
        g = aggregate_tape(sys.argv[2]); print(g.head() if g is not None else 'no tape'); print(f"{0 if g is None else len(g)} minutes")
    else: print(__doc__)
