r"""E5S - new information for the model leg from free data (E5S_PLAN.txt).
    python e5s_lab.py cross      S1: other futures already on this PC
    python e5s_lab.py mega       S2a, S2b: the 8 largest Nasdaq-100 stocks (Alpaca 5-minute bars, cache\stocks_alpaca_5m)
    python e5s_lab.py all        S3: every downloaded Nasdaq-100 stock
"""
import os, sys, glob, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import G, say
X = L.X; TR = L.TR; F = X.F; ND = L.ND
OUT = os.path.join(HERE, 'E5S_RESULTS.txt'); SHIFTS = (1, 2, 3, 5, 8); STOCKS = X.PATHS.STOCK_BARS
MARKETS = ('mym', 'm2k', 'mgc', 'mcl', 'm6e', 'mbt')
MEGA = {'AAPL': .12, 'MSFT': .11, 'NVDA': .08, 'AMZN': .07, 'GOOGL': .07, 'META': .045, 'AVGO': .04, 'TSLA': .035}
M5 = np.timedelta64(5, 'm'); M15 = np.timedelta64(15, 'm')


def flush():
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(L.LINES) + "\n")
    L.LINES.clear()


def times():
    T = G.T
    return (T.date.values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]')), T.date.values.astype('datetime64[m]') + np.timedelta64(570, 'm')


def asof(t_idx, vals, when, stale):
    j = np.searchsorted(t_idx, when, side='right') - 1; ok = j >= 0; j = np.clip(j, 0, None)
    out = np.where(ok, vals[j], np.nan); age = (when - t_idx[j]).astype('timedelta64[m]').astype(float)
    return np.where(ok & (age <= stale), out, np.nan)


def cross_features():
    now, opn = times(); cols = {}
    for mk in MARKETS:
        df = pd.read_csv(os.path.join(TR.RESEARCH, 'regime_work', 'bars5m', mk + '_5m.csv')); t = pd.to_datetime(df.time).values.astype('datetime64[m]'); o = np.argsort(t, kind='stable')
        t = t[o]; lc = np.log(df.close.values[o].astype(float)); p0 = asof(t, lc, now, 10)
        for w in (5, 15, 60): cols[f'{mk}_r{w}'] = p0 - asof(t, lc, now - np.timedelta64(w, 'm'), 10)
        cols[f'{mk}_ro'] = p0 - asof(t, lc, opn, 30)
    return pd.DataFrame(cols)


def stock_series(sym):
    files = sorted(glob.glob(os.path.join(STOCKS, f'{sym}_*.parquet')))
    if not files: return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if df.empty: return None
    for c in ('open', 'high', 'low', 'close', 'volume'): df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df.time)
    df = df.sort_values('time').drop_duplicates('time'); df = df[(df.close > 0) & (df.open > 0)].reset_index(drop=True)
    if len(df) < 1000: return None
    ts = df.time.values.astype('datetime64[m]'); te = ts + M5; day = ts.astype('datetime64[D]')
    first = np.r_[True, day[1:] != day[:-1]]; dv = df.close.values * df.volume.values; cum = np.cumsum(dv)
    tp = np.r_[te, ts[first]]; vp = np.r_[np.log(df.close.values), np.log(df.open.values[first])]; vc = np.r_[cum, (cum - dv)[first]]
    o = np.argsort(tp, kind='stable')
    dd = pd.Series(dv).groupby(day).sum(); prev = dd.shift(1)
    return dict(t=tp[o], lc=vp[o], cum=vc[o], t_open=ts[first], lo=np.log(df.open.values[first]), prev_days=prev.index.values.astype('datetime64[D]'), prev_dv=prev.values)


def stock_features(symbols, weights=None, per_stock=False, min_n=4):
    now, opn = times(); T = G.T; n = len(T); z = lambda: np.zeros(n); days = T.date.values.astype('datetime64[D]')
    A = {k: z() for k in ('sw5', 'w5', 'sw15', 'w15', 'swo', 'wo', 'u5', 'n5', 'u15', 'n15', 's1', 's2', 'dvu5', 'dv5', 'dvu15', 'dv15')}; cols = {}; used = []
    for s in symbols:
        S = stock_series(s)
        if S is None: continue
        used.append(s); p0 = asof(S['t'], S['lc'], now, 10)
        r5 = p0 - asof(S['t'], S['lc'], now - M5, 10); r15 = p0 - asof(S['t'], S['lc'], now - M15, 10); ro = p0 - asof(S['t_open'], S['lo'], opn, 1)
        c0 = asof(S['t'], S['cum'], now, 10); d5 = c0 - asof(S['t'], S['cum'], now - M5, 10); d15 = c0 - asof(S['t'], S['cum'], now - M15, 10)
        if weights is not None: w = np.full(n, float(weights[s]))
        else:
            m_ = len(S['prev_days']); j = np.clip(np.searchsorted(S['prev_days'], days), 0, m_ - 1)
            w = np.where(S['prev_days'][j] == days, S['prev_dv'][j], np.nan)
        for r, sw, ww in ((r5, 'sw5', 'w5'), (r15, 'sw15', 'w15'), (ro, 'swo', 'wo')):
            ok = np.isfinite(r) & np.isfinite(w) & (w > 0); A[sw] += np.where(ok, r * w, 0.0); A[ww] += np.where(ok, w, 0.0)
        f5 = np.isfinite(r5); f15 = np.isfinite(r15)
        A['n5'] += f5; A['u5'] += f5 & (np.nan_to_num(r5) > 0); A['n15'] += f15; A['u15'] += f15 & (np.nan_to_num(r15) > 0)
        A['s1'] += np.where(f5, r5, 0.0); A['s2'] += np.where(f5, r5 * r5, 0.0)
        g5 = f5 & np.isfinite(d5) & (d5 > 0); g15 = f15 & np.isfinite(d15) & (d15 > 0)
        A['dv5'] += np.where(g5, d5, 0.0); A['dvu5'] += np.where(g5 & (np.nan_to_num(r5) > 0), d5, 0.0); A['dv15'] += np.where(g15, d15, 0.0); A['dvu15'] += np.where(g15 & (np.nan_to_num(r15) > 0), d15, 0.0)
        if per_stock: cols[f'{s}_r5'] = r5; cols[f'{s}_r15'] = r15; cols[f'{s}_ro'] = ro
    with np.errstate(invalid='ignore', divide='ignore'):
        e5 = A['n5'] >= min_n; e15 = A['n15'] >= min_n
        agg = {'bk_r5': np.where(e5 & (A['w5'] > 0), A['sw5'] / A['w5'], np.nan) - T.r5.values, 'bk_r15': np.where(e15 & (A['w15'] > 0), A['sw15'] / A['w15'], np.nan) - T.r15.values,
               'bk_ro': np.where(e5 & (A['wo'] > 0), A['swo'] / A['wo'], np.nan) - T.ropen.values, 'up5': np.where(e5, A['u5'] / A['n5'], np.nan), 'up15': np.where(e15, A['u15'] / A['n15'], np.nan),
               'disp5': np.where(e5, np.sqrt(np.maximum(A['s2'] / A['n5'] - (A['s1'] / A['n5']) ** 2, 0.0)), np.nan),
               'upv5': np.where(e5 & (A['dv5'] > 0), A['dvu5'] / A['dv5'], np.nan), 'upv15': np.where(e15 & (A['dv15'] > 0), A['dvu15'] / A['dv15'], np.nan)}
    return pd.DataFrame(agg), pd.DataFrame(cols), used, A['n5']


def placebo(Fx, shift):
    a = Fx.values.reshape(G.n, ND, -1); out = np.full_like(a, np.nan); out[shift:] = a[:-shift]
    return out.reshape(len(Fx), -1)


def walk(Xall, y, yr, MR, years):
    pred = np.full(len(y), np.nan)
    for yy in years:
        tr = np.where((yr < yy) & np.isfinite(y))[0]; te = yr == yy
        use = np.flatnonzero(np.isfinite(Xall[tr]).sum(0) >= 2000)
        m = MR.model(); m.fit(Xall[tr][:, use], y[tr]); pred[te] = m.predict(Xall[te][:, use])
    return pred


def score(pred, y, halves):
    f = pred.reshape(G.n, ND); mp = L.model_path(f=f); leg = L.sim(mp, G.band, 1, 0); book = L.sim(mp, G.band); out = {}
    dts = np.repeat(G.dates.values, ND)
    for nm, (a, b) in zip(('h1', 'h2'), halves):
        sel = (dts >= np.datetime64(a)) & (dts <= np.datetime64(b)) & np.isfinite(pred) & np.isfinite(y)
        lg = leg[(leg.index >= a) & (leg.index <= b)]; bk = book[(book.index >= a) & (book.index <= b)]
        out[nm] = dict(ic=float(np.corrcoef(pred[sel], y[sel])[0, 1]), sh=X.sharpe(lg), yr=lg.mean() * 252, bsh=X.sharpe(bk), n=int(sel.sum()))
    return out


def run(tag, title, Fx, first=2021, split=2024, note=''):
    t0 = time.time(); L2, ML, MR, IL = TR.research(); T = G.T; y = T.y.values; yr = T.date.dt.year.values; X0 = T[F.FEATS].values; years = range(first, 2027)
    halves = ((pd.Timestamp(f'{first}-01-01'), pd.Timestamp(f'{split - 1}-12-31')), (pd.Timestamp(f'{split}-01-01'), pd.Timestamp('2026-12-31')))
    say(); say("=" * 150); say(f"E5S {tag}  e5s_lab.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {title}; out-of-sample {first}-26, halves {first}-{split - 1} / {split}-26; $ per micro at today's price, live cost"); say("=" * 150)
    cov = Fx.notna().any(axis=1).values; d = pd.DatetimeIndex(T.date.values)
    say(f"{Fx.shape[1]} new columns; rows with data from {d[cov].min():%Y-%m-%d} to {d[cov].max():%Y-%m-%d}; share of {first}-26 decisions with data {cov[(yr >= first)].mean():.1%}{note}")
    base = np.where(yr >= first, X.F_NQ, np.nan); res = {'base (deployed models)': score(base, y, halves)}
    res['with the new data'] = score(walk(np.c_[X0, Fx.values], y, yr, MR, years), y, halves)
    for s in SHIFTS: res[f'placebo ({s} sessions earlier)'] = score(walk(np.c_[X0, placebo(Fx, s)], y, yr, MR, years), y, halves)
    for k, r in res.items():
        say(f"   {k:30s} accuracy {r['h1']['ic']:+.4f} / {r['h2']['ic']:+.4f} | model-leg Sharpe {r['h1']['sh']:5.2f} / {r['h2']['sh']:5.2f} | model-leg $/yr {r['h1']['yr']:7,.0f} / {r['h2']['yr']:7,.0f} | "
            f"book Sharpe {r['h1']['bsh']:5.2f} / {r['h2']['bsh']:5.2f}")
    v, b = res['with the new data'], res['base (deployed models)']; pl = [r for k, r in res.items() if k.startswith('placebo')]
    c1 = all(v[h]['ic'] > b[h]['ic'] and v[h]['ic'] > max(p[h]['ic'] for p in pl) for h in ('h1', 'h2'))
    c2 = all(v[h]['sh'] > b[h]['sh'] and v[h]['sh'] > float(np.median([p[h]['sh'] for p in pl])) for h in ('h1', 'h2'))
    c3 = all(v[h]['bsh'] >= b[h]['bsh'] for h in ('h1', 'h2'))
    beats = all(v[h]['ic'] > max(p[h]['ic'] for p in pl) for h in ('h1', 'h2'))
    say(f"   C1 accuracy above the base and all placebos in both halves: {'yes' if c1 else 'no'} (above all placebos alone: {'yes' if beats else 'no'}) | "
        f"C2 model-leg Sharpe above the base and the placebo median: {'yes' if c2 else 'no'} | C3 book Sharpe not below the base: {'yes' if c3 else 'no'}")
    say(f"   VERDICT {tag}: " + ('FORWARD-TEST CANDIDATE (stress checks next)' if c1 and c2 and c3 else ('no information in this source' if not beats else 'carries information the placebos do not, but not enough to beat the deployed model - not adopted')))
    say(f"run {time.time() - t0:.0f} s"); flush()
    return res


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a not in ('cross', 'mega', 'all'): print(__doc__); return
    L.setup()
    if a == 'cross':
        run('S1', 'other futures on this PC (Dow, Russell, gold, crude, euro, bitcoin)', cross_features())
    elif a == 'mega':
        agg, per, used, n5 = stock_features(list(MEGA), weights=MEGA, per_stock=True, min_n=4); note = f"; stocks: {', '.join(used)}"
        six = agg[['bk_r5', 'bk_r15', 'bk_ro', 'up5', 'up15', 'disp5']]
        run('S2a', 'the 8 largest Nasdaq-100 stocks - 6 summary columns', six, first=2018, split=2022, note=note)
        run('S2b', 'the 8 largest Nasdaq-100 stocks - 30 columns', pd.concat([six, per], axis=1), first=2018, split=2022, note=note)
    else:
        import e5s_alpaca as AL
        syms = [s for s in AL.NOW + AL.PAST if s != 'QQQ']; agg, per, used, n5 = stock_features(syms, min_n=30)
        yr = G.T.date.dt.year.values; note = f"; {len(used)} stocks with data; stocks counted per decision: median {np.median(n5[yr >= 2018]):.0f}, minimum {n5[yr >= 2018].min():.0f}"
        run('S3', 'every downloaded Nasdaq-100 stock - 8 summary columns', agg, first=2018, split=2022, note=note)


if __name__ == '__main__':
    main()
