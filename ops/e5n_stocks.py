r"""E5n stock shadow test (E5S_PLAN.txt amendment 4): the model leg driven by the average of the deployed forecast and a model that also
sees the Nasdaq-100 stocks, scored every evening on unseen sessions, with stock columns from the full market (SIP) and from the free
real-time exchange (IEX). Nothing is traded.
    python e5n_stocks.py train     fit the frozen stock-aware model on data to 2026-10-15 -> data\e5n_stock_model.pkl (+ parity checks)
    python e5n_stocks.py backfill  fetch the stock bars and write the shadow ledger for every unseen session so far
"""
import os, sys, glob, json, pickle, time, datetime as dt
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'research'); sys.path.insert(0, HERE); sys.path.insert(0, RES)
import numpy as np, pandas as pd
import e5n_paths as PATHS
import e5n_features as F
import e5n_calendar as CAL
import e5s_alpaca as AL

DATA = os.path.join(HERE, 'data'); MODEL = os.path.join(DATA, 'e5n_stock_model.pkl'); LEDGER = os.path.join(DATA, 'e5n_shadow_stocks.csv')
DIRS = {'sip': PATHS.STOCK_BARS, 'iex': PATHS.IEX_BARS}
SYMS = [s for s in AL.NOW + AL.PAST if s != 'QQQ']
AGG = ['bk_r5', 'bk_r15', 'bk_ro', 'up5', 'up15', 'disp5', 'upv5', 'upv15']
TRAIN_END = pd.Timestamp('2026-10-15'); FORWARD = pd.Timestamp('2026-10-16'); MIN_N = 30
M5 = np.timedelta64(5, 'm'); M15 = np.timedelta64(15, 'm')
COLS = ['date', 'base_usd', 'sip_usd', 'iex_usd', 'band_usd', 'book_base', 'book_sip', 'book_iex', 'sip_cover', 'iex_cover', 'sip_stocks', 'iex_stocks', 'recorded']


def asof(t_idx, vals, when, stale):
    j = np.searchsorted(t_idx, when, side='right') - 1; ok = j >= 0; j = np.clip(j, 0, None)
    out = np.where(ok, vals[j], np.nan); age = (when - t_idx[j]).astype('timedelta64[m]').astype(float)
    return np.where(ok & (age <= stale), out, np.nan)


def series(sym, d):
    files = sorted(glob.glob(os.path.join(d, f'{sym}_*.parquet')))
    if not files: return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.empty: return None
    for c in ('open', 'high', 'low', 'close', 'volume'): df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df.time)
    df = df.sort_values('time').drop_duplicates('time'); df = df[(df.close > 0) & (df.open > 0)].reset_index(drop=True)
    if len(df) < 2: return None
    ts = df.time.values.astype('datetime64[m]'); te = ts + M5; day = ts.astype('datetime64[D]')
    first = np.r_[True, day[1:] != day[:-1]]; dv = df.close.values * df.volume.values; cum = np.cumsum(dv)
    tp = np.r_[te, ts[first]]; vp = np.r_[np.log(df.close.values), np.log(df.open.values[first])]; vc = np.r_[cum, (cum - dv)[first]]
    o = np.argsort(tp, kind='stable')
    dd = pd.Series(dv).groupby(day).sum(); prev = dd.shift(1)
    return dict(t=tp[o], lc=vp[o], cum=vc[o], t_open=ts[first], lo=np.log(df.open.values[first]), prev_days=prev.index.values.astype('datetime64[D]'), prev_dv=prev.values)


def aggregates(T, d, syms=SYMS, min_n=MIN_N):
    dates = pd.to_datetime(T.date).values.astype('datetime64[m]'); now = dates + (570 + T.dec.values).astype('timedelta64[m]'); opn = dates + np.timedelta64(570, 'm')
    n = len(T); days = pd.to_datetime(T.date).values.astype('datetime64[D]')
    A = {k: np.zeros(n) for k in ('sw5', 'w5', 'sw15', 'w15', 'swo', 'wo', 'u5', 'n5', 'u15', 'n15', 's1', 's2', 'dvu5', 'dv5', 'dvu15', 'dv15')}; used = 0
    for s in syms:
        S = series(s, d)
        if S is None: continue
        used += 1; p0 = asof(S['t'], S['lc'], now, 10)
        r5 = p0 - asof(S['t'], S['lc'], now - M5, 10); r15 = p0 - asof(S['t'], S['lc'], now - M15, 10); ro = p0 - asof(S['t_open'], S['lo'], opn, 1)
        c0 = asof(S['t'], S['cum'], now, 10); d5 = c0 - asof(S['t'], S['cum'], now - M5, 10); d15 = c0 - asof(S['t'], S['cum'], now - M15, 10)
        m_ = len(S['prev_days']); j = np.clip(np.searchsorted(S['prev_days'], days), 0, m_ - 1)
        w = np.where(S['prev_days'][j] == days, S['prev_dv'][j], np.nan)
        for r, sw, ww in ((r5, 'sw5', 'w5'), (r15, 'sw15', 'w15'), (ro, 'swo', 'wo')):
            ok = np.isfinite(r) & np.isfinite(w) & (w > 0); A[sw] += np.where(ok, r * w, 0.0); A[ww] += np.where(ok, w, 0.0)
        f5 = np.isfinite(r5); f15 = np.isfinite(r15)
        A['n5'] += f5; A['u5'] += f5 & (np.nan_to_num(r5) > 0); A['n15'] += f15; A['u15'] += f15 & (np.nan_to_num(r15) > 0)
        A['s1'] += np.where(f5, r5, 0.0); A['s2'] += np.where(f5, r5 * r5, 0.0)
        g5 = f5 & np.isfinite(d5) & (d5 > 0); g15 = f15 & np.isfinite(d15) & (d15 > 0)
        A['dv5'] += np.where(g5, d5, 0.0); A['dvu5'] += np.where(g5 & (np.nan_to_num(r5) > 0), d5, 0.0); A['dv15'] += np.where(g15, d15, 0.0); A['dvu15'] += np.where(g15 & (np.nan_to_num(r15) > 0), d15, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        e5 = A['n5'] >= min_n; e15 = A['n15'] >= min_n
        agg = {'bk_r5': np.where(e5 & (A['w5'] > 0), A['sw5'] / A['w5'], np.nan) - T.r5.values, 'bk_r15': np.where(e15 & (A['w15'] > 0), A['sw15'] / A['w15'], np.nan) - T.r15.values,
               'bk_ro': np.where(e5 & (A['wo'] > 0), A['swo'] / A['wo'], np.nan) - T.ropen.values, 'up5': np.where(e5, A['u5'] / A['n5'], np.nan), 'up15': np.where(e15, A['u15'] / A['n15'], np.nan),
               'disp5': np.where(e5, np.sqrt(np.maximum(A['s2'] / A['n5'] - (A['s1'] / A['n5']) ** 2, 0.0)), np.nan),
               'upv5': np.where(e5 & (A['dv5'] > 0), A['dvu5'] / A['dv5'], np.nan), 'upv15': np.where(e15 & (A['dv15'] > 0), A['dvu15'] / A['dv15'], np.nan)}
    out = pd.DataFrame(agg, index=T.index); out.attrs['stocks'] = used; out.attrs['n5'] = A['n5']
    return out


def train():
    t0 = time.time(); import e5x_lab as X, e5b_lab as L, e5s_lab as S, e5n_train as TR
    L.setup(band=False); L2, ML, MR, IL = TR.research(); T = X.NQ.T
    new = aggregates(T, DIRS['sip']); ref, _, _, _ = S.stock_features(SYMS, min_n=MIN_N)
    a, b = new[AGG].values, ref[AGG].values; both = np.isfinite(a) & np.isfinite(b)
    d = float(np.max(np.abs(a[both] - b[both]))) if both.any() else float('nan'); same_nan = bool((np.isfinite(a) == np.isfinite(b)).all())
    print(f"parity: stock columns from this module vs the research: max |diff| {d:.1e}, same missing cells {same_nan}", flush=True)
    assert d < 1e-12 and same_nan, 'stock columns differ from the research'
    Xm = np.c_[T[F.FEATS].values, new[AGG].values]; y = T.y.values; ok = (T.date <= TRAIN_END).values & np.isfinite(y)
    m = MR.model(); m.fit(Xm[ok], y[ok])
    os.makedirs(DATA, exist_ok=True)
    with open(MODEL, 'wb') as fh: pickle.dump(dict(model=m, feats=F.FEATS + AGG, train_end=str(TRAIN_END.date()), rows=int(ok.sum()), built=dt.datetime.now().isoformat(timespec='minutes')), fh)
    print(f"stock-aware model trained on {int(ok.sum()):,} rows to {TRAIN_END.date()} -> {MODEL}; {time.time() - t0:.0f} s", flush=True)


def fetch(feed, start, end):
    k, s = AL.keys()
    if not (k and s): return 'no Alpaca key'
    d = DIRS[feed]; os.makedirs(d, exist_ok=True); rows = []
    stop = min(pd.Timestamp(end) + pd.Timedelta(days=1), pd.Timestamp.now('UTC').tz_localize(None) - pd.Timedelta(minutes=20))
    for i in range(0, len(SYMS), 50):
        params = dict(symbols=','.join(SYMS[i:i + 50]), timeframe='5Min', start=start.strftime('%Y-%m-%dT00:00:00Z'), end=stop.strftime('%Y-%m-%dT%H:%M:%SZ'),
                      limit=10000, adjustment='split', feed=feed, sort='asc')
        while True:
            j = AL.get(params, k, s)
            for sym, bars in (j.get('bars') or {}).items():
                for b in bars: rows.append((sym, b['t'], b['o'], b['h'], b['l'], b['c'], b['v'], b.get('n', np.nan), b.get('vw', np.nan)))
            tok = j.get('next_page_token')
            if not tok: break
            params['page_token'] = tok
    if not rows: return f'{feed}: no bars returned for {start:%Y-%m-%d} .. {end:%Y-%m-%d}'
    df = pd.DataFrame(rows, columns=['sym', 't', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vwap'])
    df['time'] = pd.to_datetime(df.t, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); m = df.time.dt.hour * 60 + df.time.dt.minute
    df = df[(m >= 570) & (m < 960)]
    for (sym, y), g in df.groupby([df.sym, df.time.dt.year]):
        p = os.path.join(d, f'{sym}_{y}.parquet'); g = g[['time', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vwap']]
        if os.path.exists(p):
            old = pd.read_parquet(p)
            if len(old): g = pd.concat([old, g], ignore_index=True)
        g = g.drop_duplicates('time', keep='last').sort_values('time'); g.to_parquet(p, index=False)
    return ''


def last_day(feed):
    f = sorted(glob.glob(os.path.join(DIRS[feed], 'AAPL_*.parquet')))
    if not f: return None
    x = pd.read_parquet(f[-1], columns=['time']); return pd.Timestamp(x.time.max()).normalize() if len(x) else None


def leg_days(T, f, g15, thr):
    out = {}
    for k, g in T.assign(f=f, g=g15).groupby('k', sort=True):
        pos = 0; pts = 0.0; sides = 0; last = None; n = 0
        for r in g.itertuples():
            if not np.isfinite(r.f): continue
            n += 1; new = 1 if r.f > thr else (-1 if r.f < -thr else 0)
            if new != 0 and not (np.isfinite(r.g) and r.g * new > 0): new = 0
            if pos != 0: pts += pos * (r.entry - last); last = r.entry
            if new != pos: sides += abs(new - pos); pos = new; last = r.entry
        if pos != 0: pts += pos * (g.exitp.iat[-1] - last); sides += 1
        if n: out[pd.Timestamp(g.date.iat[0]).date()] = (round(pts, 2), sides / 2)
    return out


def shadow(S, M, M15, band, warmup, now=None, fetch_data=True):
    if not os.path.exists(MODEL): return None, 'stock-aware model not trained (python e5n_stocks.py train)'
    now = now or dt.datetime.now(); notes = []
    if fetch_data:
        end = pd.Timestamp(now.date()) if now.hour * 60 + now.minute >= 16 * 60 + 20 else pd.Timestamp(now.date()) - pd.Timedelta(days=1)
        for feed, first in (('sip', None), ('iex', FORWARD - pd.Timedelta(days=7))):
            ld = last_day(feed); start = (ld + pd.Timedelta(days=1)) if ld is not None else first
            trading = start is not None and any(d.weekday() < 5 and d.date() not in CAL.closed_days(d.year) for d in pd.date_range(start, end))
            if trading:
                try:
                    e = fetch(feed, start, end)
                    if e: notes.append(e)
                except SystemExit as ex: notes.append(f'{feed} download failed: {ex}')
    T = F.features(S)
    T['mend'] = np.array([CAL.month_end_flag(d.date()) for d in S.D.date], float)[T.k.values]
    T = T[(T.k >= warmup) & S.present[T.k.values, T.dec.values - 1] & (T.date >= FORWARD)].reset_index(drop=True)
    if not len(T): return {}, '; '.join(notes)
    X0 = T[F.FEATS].values; f5, _ = M.predict(T.date.values, X0); f15, _ = M15.predict(T.date.values, X0)
    md = pickle.load(open(MODEL, 'rb')); m = md['model']; assert md['feats'] == F.FEATS + AGG
    res = {'base': leg_days(T, f5, f15, M.threshold)}; cover = {}; stocks = {}
    for feed in ('sip', 'iex'):
        ag = aggregates(T, DIRS[feed]); has = np.isfinite(ag[AGG].values).any(axis=1)
        s3 = m.predict(np.c_[X0, ag[AGG].values]); blend = np.where(has & np.isfinite(f5), (f5 + s3) / 2, f5)
        res[feed] = leg_days(T, blend, f15, M.threshold)
        cv = pd.Series(has, index=T.index).groupby(T.date).mean(); cover[feed] = {pd.Timestamp(k).date(): float(v) for k, v in cv.items()}; stocks[feed] = ag.attrs['stocks']
    out = {}
    for d in sorted(res['base']):
        b = band.get(d, dict(pts=0.0, trips=0.0)); usd = lambda p: round(p[0] * 2.0 - p[1] * 2 * F.SIDE_USD, 2)
        bu = round(b['pts'] * 2.0 - b['trips'] * 2 * F.SIDE_USD, 2); r = dict(date=d, band_usd=bu)
        for feed in ('base', 'sip', 'iex'): r[f'{feed}_usd'] = usd(res[feed].get(d, (0.0, 0.0))); r[f'book_{feed}'] = round(r[f'{feed}_usd'] + bu, 2)
        r['sip_cover'] = round(cover['sip'].get(d, 0.0), 3); r['iex_cover'] = round(cover['iex'].get(d, 0.0), 3); r['sip_stocks'] = stocks['sip']; r['iex_stocks'] = stocks['iex']
        out[d] = r
    return out, '; '.join(notes)


def append_ledger(rows, now):
    old = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    have = set(pd.to_datetime(old.date).dt.date) if len(old) else set()
    new = [dict(rows[d], recorded=now.strftime('%Y-%m-%d %H:%M')) for d in sorted(rows) if d not in have and rows[d]['sip_cover'] > 0.5 and rows[d]['iex_cover'] > 0.5]
    if new: pd.DataFrame(new)[COLS].to_csv(LEDGER, mode='a', header=not os.path.exists(LEDGER), index=False)
    L = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    return L, [str(r['date']) for r in new]


def summary(L):
    if not len(L): return dict(sessions=0)
    tt = lambda x: float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else 0.0
    ds, di = L.sip_usd - L.base_usd, L.iex_usd - L.base_usd
    return dict(sessions=int(len(L)), base_usd=round(float(L.base_usd.sum()), 2), sip_usd=round(float(L.sip_usd.sum()), 2), iex_usd=round(float(L.iex_usd.sum()), 2),
                sip_lead=round(float(ds.sum()), 2), sip_lead_t=round(tt(ds), 2), iex_lead=round(float(di.sum()), 2), iex_lead_t=round(tt(di), 2),
                stop_line_126=-1000.0, verdict_at=252)


def backfill():
    import e5n_daily as D
    from e5n_model import ModelSet
    now = dt.datetime.now(); M = ModelSet(D.MODELS); M15 = ModelSet(D.MODELS15)
    a, keep = D.all_bars(); bars = D.minute_bars(keep); S = D.Sessions(bars); cont, offs = D.continuous(a, keep); bd, marks = D.band_days(cont)
    rows, note = shadow(S, M, M15, bd, D.WARMUP, now)
    if rows is None: print(note); return
    L, added = append_ledger(rows, now); print(f"shadow ledger: {len(L)} sessions, added {len(added)}" + (f"; notes: {note}" if note else ''))
    print(json.dumps(summary(L), indent=1))


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'train': train()
    elif a == 'backfill': backfill()
    else: print(__doc__)
