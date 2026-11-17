r"""E5Y Y10 + Y11 (E5Y_PLAN.txt): do fast-trading stocks lead the index at 15 minutes, and do the members' lagged 5-minute returns add to
E5n's forecast? 151 Nasdaq-100 stocks, 5-minute bars 2016-2026.
    python e5y_stocks.py
"""
import os, sys, glob, json, time, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5s_lab as ES
from e5b_lab import G, say
X = L.X; U = L.U; ND = L.ND; WEB = X.PATHS.WEB


def caps(symbols):
    f = os.path.join(WEB, 'mktcap.csv')
    if os.path.exists(f): return pd.read_csv(f, index_col=0).cap.to_dict()

    def one(s):
        try:
            raw = subprocess.run(['curl', '-s', '--max-time', '25', '-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '-H', 'Accept: application/json',
                                  f"https://api.nasdaq.com/api/quote/{s}/summary?assetclass=stocks"], capture_output=True, timeout=40).stdout
            v = ((json.loads(raw.decode('utf-8', 'replace')).get('data') or {}).get('summaryData') or {}).get('MarketCap', {}).get('value', '')
            return s, float(str(v).replace(',', '').replace('$', '')) if v not in ('', 'N/A') else np.nan
        except Exception: return s, np.nan
    with ThreadPoolExecutor(6) as ex: out = dict(ex.map(one, symbols))
    pd.DataFrame({'cap': out}).to_csv(f); return out


def main():
    t0 = time.time(); C.load(); n = G.n; now, opn = ES.times(); yr = np.repeat(G.dates.year.values, ND); dts = np.repeat(G.dates.values, ND)
    syms = sorted({os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(ES.STOCKS, '*_*.parquet'))}); cap = caps(syms)
    R5 = np.full((n * ND, len(syms)), np.nan, np.float32); R15 = np.full((n * ND, len(syms)), np.nan, np.float32); TO = np.full((n, len(syms)), np.nan); CAPD = np.full((n, len(syms)), np.nan)
    day_idx = {d: i for i, d in enumerate(G.dates.values.astype('datetime64[D]'))}
    for k, s in enumerate(syms):
        z = ES.stock_series(s)
        if z is None: continue
        p0 = ES.asof(z['t'], z['lc'], now, 10); R5[:, k] = p0 - ES.asof(z['t'], z['lc'], now - np.timedelta64(5, 'm'), 10); R15[:, k] = p0 - ES.asof(z['t'], z['lc'], now - np.timedelta64(15, 'm'), 10)
        df = pd.concat([pd.read_parquet(f, columns=['time', 'close', 'volume']) for f in sorted(glob.glob(os.path.join(ES.STOCKS, f'{s}_*.parquet')))], ignore_index=True)
        df['d'] = pd.to_datetime(df.time).values.astype('datetime64[D]'); g = df.groupby('d').agg(v=('volume', 'sum'), c=('close', 'last'))
        c_ = cap.get(s, np.nan)
        if c_ == c_ and len(g):
            sh = c_ / g.c.iloc[-1]; to = (g.v.rolling(126, min_periods=60).mean() / sh).shift(1); cp = (g.c * sh).shift(1)
            ii = np.array([day_idx.get(d, -1) for d in g.index.values]); ok = ii >= 0; TO[ii[ok], k] = to.values[ok]; CAPD[ii[ok], k] = cp.values[ok]
    have = np.isfinite(R5).any(0); say(); say("=" * 160)
    say(f"WEB SCOUT BATCH 1, STOCK RULES  e5y_stocks.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5Y_PLAN.txt Y10, Y11) {int(have.sum())} stocks with data, market caps for {int(np.isfinite(list(cap.values())).sum())}; loaded in {time.time() - t0:.0f} s")
    say("=" * 160)
    px = G.px; fwd15 = np.full((n, ND), np.nan); fwd15[:, :-3] = px[:, 3:] / px[:, :-3] - 1; fwd15[:, -3:] = (G.exit[:, None] / px[:, -3:] - 1)
    idx15 = np.full((n, ND), np.nan); idx15[:, 3:] = np.log(px[:, 3:] / px[:, :-3])
    sig = np.full((n, ND), np.nan)
    for i in range(n):
        to = TO[i]; ok = np.isfinite(to) & np.isfinite(CAPD[i])
        if ok.sum() < 40: continue
        top = np.flatnonzero(ok)[np.argsort(-to[ok])[:15]]; w = CAPD[i, top]; r = R15[i * ND:(i + 1) * ND][:, top].astype(float)
        m = np.isfinite(r); pr = np.where(m.sum(1) >= 8, np.nansum(np.where(m, r, 0.0) * w, axis=1) / np.maximum((m * w).sum(1), 1e-9), np.nan); sig[i] = pr - idx15[i]
    dec15 = (G.dec % 15 == 0) & (G.dec >= 30) & (G.dec <= 375); pos = np.zeros((n, ND)); vd = np.flatnonzero(np.isfinite(sig).any(1))
    for i in vd:
        prior = vd[(vd < i)][-126:]
        if len(prior) < 60: continue
        sd = np.nanstd(sig[prior][:, dec15]); z = sig[i] / sd; cur = 0.0
        for g_ in range(ND):
            if dec15[g_]: cur = 1.0 if z[g_] > 1 else -1.0 if z[g_] < -1 else 0.0
            pos[i, g_] = cur if G.dec[g_] >= 30 else 0.0
    dp = np.diff(np.c_[px, G.exit], axis=1); c1 = X.SIDE(2.35) * U; tr = np.abs(np.diff(np.c_[np.zeros(n), pos, np.zeros(n)], axis=1)).sum(1)
    gross = (pos * dp).sum(1) / G.o930 * U; net = gross - c1 * tr; d_ = G.dates; tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0
    say("Y10 FAST STOCKS LEAD (15 highest-turnover stocks minus the index, 15 minutes; long / short the index for the next 15 minutes beyond 1 standard deviation):")
    for lab, m in (('discovery 2016-21', (d_ >= '2016-07-01') & (d_ <= '2021-12-31')), ('holdout 2022-26', d_ >= '2022-01-01')):
        m = np.asarray(m) & np.isin(np.arange(n), vd); x = net[m]; g_ = gross[m]; s_ = pd.Series(x, index=d_[m]); yy = s_.groupby(s_.index.year).sum()
        say(f"   {lab}: {tr[m].sum() / 2 / (m.sum() / 252):,.0f} round trips a year, ${g_.mean() * 252:+,.0f}/yr before costs (t {tt(g_):+.2f}), ${x.mean() * 252:+,.0f}/yr after (t {tt(x):+.2f}), losing years {int((yy < 0).sum())} of {len(yy)}")
    ic = np.corrcoef(sig[np.isfinite(sig) & np.isfinite(fwd15) & dec15[None, :]], fwd15[np.isfinite(sig) & np.isfinite(fwd15) & dec15[None, :]])[0, 1]
    say(f"   correlation of the signal with the index's next 15 minutes, all years: {ic:+.4f}  -> Y10 {'PASS' if False else 'see t above (bar 2.8 / 2.0)'}")
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    y = G.T.y.values.astype(float); f5 = G.f5.reshape(-1); cols = np.flatnonzero(have); Xs = R5[:, cols]

    def walk(Xa):
        pred = np.full(len(y), np.nan)
        for yy in range(2018, 2027):
            tr_ = (yr < yy) & (yr >= 2016) & np.isfinite(y) & (np.isfinite(Xa).sum(1) >= 40); te = (yr == yy) & (np.isfinite(Xa).sum(1) >= 40)
            if tr_.sum() < 20000 or not te.any(): continue
            A = Xa[tr_].astype(float); sd = np.nanstd(A, axis=0); sd[~(sd > 0)] = 1.0; A = np.nan_to_num(A / sd); B = np.nan_to_num(Xa[te].astype(float) / sd)
            m = LassoCV(alphas=12, cv=TimeSeriesSplit(3), max_iter=2000, n_jobs=4).fit(A, y[tr_]); p = m.predict(B)
            ins = m.predict(A); k = np.nanstd(f5[tr_]) / ins.std() if ins.std() > 0 else 0.0; pred[te] = p * k
        return pred
    halves = (('2018-01-01', '2021-12-31'), ('2022-01-01', '2026-12-31'))
    say(); say("Y11 MEMBERS' LAGGED 5-MINUTE RETURNS (lasso, refit each January) - what it adds to E5n's own forecast; out of sample 2018-21 | 2022-26:")
    base = ES.score(np.where(np.isfinite(f5), f5, np.nan), y, halves); pl = walk(Xs); sc_l = ES.score(pl, y, halves)
    blend = np.where(np.isfinite(pl), (f5 + pl) / 2, f5); sc_b = ES.score(blend, y, halves)
    row = lambda lab, s: say(f"   {lab:44s} | accuracy {s['h1']['ic']:+.4f} / {s['h2']['ic']:+.4f} | model leg Sharpe {s['h1']['sh']:.2f} / {s['h2']['sh']:.2f} | book Sharpe {s['h1']['bsh']:.2f} / {s['h2']['bsh']:.2f}")
    row("E5n's forecast alone", base); row("lasso on the stocks alone", sc_l); row("average of the two", sc_b)
    pb = []
    for shf in (1, 2, 3, 5, 8):
        a = Xs.reshape(n, ND, -1); o = np.full_like(a, np.nan); o[shf:] = a[:-shf]; pp = walk(o.reshape(len(y), -1)); s = ES.score(np.where(np.isfinite(pp), (f5 + pp) / 2, f5), y, halves); pb.append(s)
    say(f"   placebo blends (the stocks' returns taken from 1-8 sessions earlier): accuracy {min(s['h1']['ic'] for s in pb):+.4f}..{max(s['h1']['ic'] for s in pb):+.4f} / {min(s['h2']['ic'] for s in pb):+.4f}..{max(s['h2']['ic'] for s in pb):+.4f} | "
        f"book Sharpe {min(s['h1']['bsh'] for s in pb):.2f}..{max(s['h1']['bsh'] for s in pb):.2f} / {min(s['h2']['bsh'] for s in pb):.2f}..{max(s['h2']['bsh'] for s in pb):.2f}")
    ok = all(sc_b[h]['ic'] > base[h]['ic'] and sc_b[h]['ic'] > max(s[h]['ic'] for s in pb) and sc_b[h]['bsh'] > base[h]['bsh'] for h in ('h1', 'h2'))
    say(f"   -> Y11 {'PASSES (beats E5n alone and every placebo in both halves)' if ok else 'does not beat E5n alone and every placebo in both halves: dead'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
