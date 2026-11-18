r"""GP (GP_PLAN.txt): breadth confirmations, 08:30 shock days and VIX switches on the exact E5n book.
    python gp_gaps.py run
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5u_bandml as UB
import e5s_lab as S
import red_rules as RR
from e5b_lab import G, say

TK = 4.5
ND = L.ND
WEB = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'web')


def breadth():
    syms = sorted({f.split('_')[0] for f in os.listdir(S.STOCKS) if f.endswith('.parquet')} - {'QQQ'})
    agg, _, used, n5 = S.stock_features(syms, min_n=30)
    say(f"   breadth: {len(used)} stocks; decision rows with >= 30 stocks: {(n5 >= 30).mean():.1%} of all, "
        f"{(n5[np.repeat(np.asarray(G.dates >= '2016-01-04'), ND)] >= 30).mean():.1%} since 2016")
    return agg


def shocks():
    out = {}
    for nm, sel in (('nq', lambda d: d <= L.JOIN), ('mnqx', lambda d: d > L.JOIN)):
        d = RR.load(nm); w = (d['md'] >= 510) & (d['md'] <= 514); day = d['t'][w] // 1440
        df = pd.DataFrame({'day': day, 'h': d['h'][w], 'l': d['l'][w]}).groupby('day').agg(h=('h', 'max'), l=('l', 'min'))
        rng = (df.h - df.l); med = rng.shift(1).rolling(20, min_periods=15).median(); ratio = rng / med
        idx = pd.to_datetime(df.index.values * 86400, unit='s')
        for dt, v in zip(idx, ratio.values):
            if sel(dt): out[pd.Timestamp(dt)] = v
    return pd.Series(out).sort_index()


def vix_prev(dates):
    v = pd.read_csv(os.path.join(WEB, 'VIX_History.csv')); v.index = pd.to_datetime(v.DATE, format='%m/%d/%Y'); c = v.CLOSE.astype(float)
    j = np.searchsorted(c.index.values, dates.values, side='left') - 1
    return np.where(j >= 0, c.values[np.clip(j, 0, None)], np.nan)


def model_masked(drop):
    M = C.M; g5 = M.g5.copy(); g5[drop] = 0.0
    return C.path(g5, M.g15, 'set', M.grid)


def run():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'GP_RESULTS.txt')
    say(); say("=" * 150); say(f"GP  GAPS IN THE OWNED DATA  gp_gaps.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  exact book 1 + 1, {TK} ticks a side"); say("=" * 150)
    gi = np.clip((M.dec - 15) // 5, 0, ND - 1); assert (G.T.dec.values[gi[M.grid]] == M.dec[M.grid]).all(), 'decision rows misaligned'
    rows = np.arange(M.n)[:, None] * ND + gi[None, :]
    model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band; BT = UB.trades()
    brow = BT.r.values * ND + (BT.j0.values - 10) // 5; bdec = BT.j0.values + 5; bd = BT.d.values
    tgt = C.target(M.g5, M.g15); V = {}
    agg = breadth()
    for code, cols in (('1', ['up15']), ('2', ['upv15']), ('3', ['up15', 'upv15'])):
        disb = np.zeros(len(BT), bool); dism = np.zeros_like(M.g5, bool)
        for c in cols:
            a = agg[c].values; ab = a[brow]; am = a[rows]
            disb |= np.isfinite(ab) & (bd * (ab - 0.5) < 0)
            dism |= np.isfinite(am) & (tgt * (am - 0.5) < 0) & M.grid[None, :]
        lab = ' + '.join(cols)
        V[f'B{code} band entries need breadth ({lab})'] = ('br', model + UB.path_with(BT, np.where(disb, 0.0, 1.0)), int(disb.sum()))
        V[f'M{code} model leg needs breadth ({lab})'] = ('br', model_masked(dism) + band, int((dism & (tgt != 0)).sum()))
    sh = shocks(); sd = pd.Series(sh.reindex(M.dates).values, index=M.dates); isk = (sd >= 2.5).values
    per = pd.Series(isk, index=M.dates).groupby(M.dates.year).sum()
    say("   08:30 shock days a year: " + ", ".join(f"{y}: {int(v)}" for y, v in per.items()) + f" | days without the 08:30 window: {int(np.isnan(sd.values).sum())}")
    early = (M.dec < 60)[None, :]
    V['E1 shock days: model leg flat before 10:30'] = ('all', model_masked(isk[:, None] & early & M.grid[None, :]) + band, int(isk.sum()))
    skE = isk[BT.r.values] & (bdec < 60)
    V['E2 shock days: no band entries before 10:30'] = ('all', model + UB.path_with(BT, np.where(skE, 0.0, 1.0)), int(skE.sum()))
    net = model + band; net = np.where(isk[:, None], 0.0, net)
    V['E3 shock days: no trading'] = ('all', net, int(isk.sum()))
    vx = vix_prev(M.dates); A_ = np.asarray(M.dates <= L.JOIN) & np.asarray(M.dates.year >= 2013); lo, hi = np.nanpercentile(vx[A_], [100 / 3, 200 / 3])
    say(f"   VIX terciles from 2013-19 sessions: low < {lo:.2f}, high > {hi:.2f}; sessions without a VIX close {int(np.isnan(vx).sum())}")
    skV = (vx < lo)[BT.r.values]
    V['V1 band entries off when VIX in its low third'] = ('all', model + UB.path_with(BT, np.where(skV, 0.0, 1.0)), int(skV.sum()))
    V['V2 model leg off when VIX in its high third'] = ('all', model_masked((vx > hi)[:, None] & M.grid[None, :]) + band, int((vx > hi).sum()))
    base, _ = C.simm(model + band, TK)
    win = {'all': (base.index >= '2013-01-01') & (base.index <= L.JOIN), 'br': (base.index >= '2016-01-04') & (base.index <= L.JOIN)}
    B_ = base.index > L.JOIN
    shp = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    for k_, m_ in win.items():
        say(f"   baseline select window ({'2013-19' if k_ == 'all' else '2016-19'}): Sharpe {shp(base[m_]):.2f} ${base[m_].mean() * 252:,.0f}/yr DD ${L.mdd(base[m_]):,.0f}")
    say(f"   baseline 2019-26: Sharpe {shp(base[B_]):.2f} ${base[B_].mean() * 252:,.0f}/yr DD ${L.mdd(base[B_]):,.0f}")
    res = {}
    for nm, (w, net, cnt) in V.items():
        s, _ = C.simm(net, TK); m_ = win[w]
        sel = shp(s[m_]) > shp(base[m_]) and L.mdd(s[m_]) <= L.mdd(base[m_]); res[nm] = (s, sel)
        say(f"   {nm:48s} affects {cnt:6,d} | select Sh {shp(s[m_]):.2f} ({shp(s[m_]) - shp(base[m_]):+.2f}) DD ${L.mdd(s[m_]):,.0f} {'SEL' if sel else '   '}| "
            f"2019-26 Sh {shp(s[B_]):.2f} ({shp(s[B_]) - shp(base[B_]):+.2f}) ${s[B_].mean() * 252:,.0f}/yr ({(s[B_] - base[B_]).mean() * 252:+,.0f}) DD ${L.mdd(s[B_]):,.0f}")
    selected = [nm for nm, (_, sel) in res.items() if sel]; bar = 2.0 if len(selected) == 1 else 2.4
    say(); say(f"   CONFIRMATION on 2019-26 for {len(selected)} selected (bar paired t >= {bar}):" if selected else "   CONFIRMATION: nothing selected.")
    for nm in selected:
        s = res[nm][0]; d = (s[B_] - base[B_]).values; t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        ok = shp(s[B_]) > shp(base[B_]) and L.mdd(s[B_]) <= L.mdd(base[B_]) and t >= bar
        yrs = (s - base)[s.index >= '2013-01-01'].groupby(s.index[s.index >= '2013-01-01'].year).sum()
        say(f"   {nm}: paired t {t:+.2f} -> {'CONFIRMED' if ok else 'DEAD'} | $ by year vs baseline: " + ", ".join(f"{y}: {v:+,.0f}" for y, v in yrs.items()))
    say(f"11 trials; run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run': run()
    else: print(__doc__)
