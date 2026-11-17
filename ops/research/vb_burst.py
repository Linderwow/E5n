r"""VB (VB_PLAN.txt): waiting out volume bursts.
    python vb_burst.py cost     Part 1 on the tick store
    python vb_burst.py run      Part 2 on the research grid
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import red_rules as R
from e5b_lab import G, say

BANDS = ((0, 1.5), (1.5, 3), (3, 5), (5, np.inf))


def vol_grid(dates):
    out = np.full((len(dates), 390), np.nan); row = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    for nm in ('nq', 'mnqx'):
        d = R.load(nm); md = d['md']; ok = (md >= 570) & (md < 960)
        day = pd.to_datetime(d['tday'][ok] * 86400, unit='s').normalize(); pos = md[ok] - 570; v = d['v'][ok]
        idx = np.array([row.get(x, -1) for x in day]); m = idx >= 0; out[idx[m], pos[m]] = v[m]
    avg = pd.DataFrame(out).shift(1).rolling(20, min_periods=10).mean().values
    with np.errstate(invalid='ignore', divide='ignore'): return out / avg


def cost():
    import ut_tests as U, ut_timing2 as T2
    t0 = time.time(); dlog = pd.read_csv(U.DLOG, usecols=['date', 'time_et', 'net']); dlog = dlog[dlog.date >= T2.START].copy()
    dlog['prev'] = dlog.groupby('date').net.shift(1).fillna(0); od = dlog[(dlog.net != dlog.prev) & (dlog.time_et < '16:00')].copy()
    od['dir'] = np.sign(od.net - od.prev); od['T'] = pd.to_datetime(od.date + ' ' + od.time_et)
    TK = T2.tick_days(); ep = pd.Timestamp('1970-01-01'); rows = []
    hist = {}
    for day in sorted(TK):
        ms, px, vo = TK[day]; o = int((day + pd.Timedelta(hours=9, minutes=30) - ep).total_seconds() * 1000)
        b = np.searchsorted(ms, o + np.arange(391) * 60_000); hist[day] = np.diff(np.r_[0, np.cumsum(vo)][b])
    days = sorted(hist)
    for i, day in enumerate(days):
        prev = [hist[d] for d in days[max(0, i - 20):i]]
        if len(prev) < 10: continue
        avg = np.mean(prev, axis=0); ms, px, vo = TK[day]
        for r in od[od.date == f"{day:%Y-%m-%d}"].itertuples():
            T = int((r.T - ep).total_seconds() * 1000); col = int((r.T - day).total_seconds() // 60) - 570 - 1
            at = lambda t_: px[np.searchsorted(ms, t_, side='right') - 1]
            rows.append((hist[day][col] / avg[col] if avg[col] > 0 else np.nan, r.dir * (at(T + 400) - at(T)) / 0.25, r.dir * (at(T + 60_000) - at(T)) / 0.25))
    X = pd.DataFrame(rows, columns=['ratio', 'c04', 'c60']).dropna()
    say(); say("=" * 150); say(f"VB PART 1  COST OF E5n's ORDERS BY VOLUME BURST  vb_burst.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  tick store, {len(X):,} orders; ticks in the order's direction"); say("=" * 150)
    for lo, hi in BANDS:
        g = X[(X.ratio >= lo) & (X.ratio < hi)]
        if len(g): say(f"   volume ratio {lo:>3} - {hi:<4}: {len(g):5d} orders ({len(g) / len(X):4.0%}) | close -> +0.4 s {g.c04.mean():+5.2f} ticks | close -> +60 s {g.c60.mean():+6.2f} ticks")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


def run():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'VB_RESULTS.txt')
    vr = vol_grid(M.dates); colr = vr[:, M.dec - 1]
    model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band
    base, _ = C.simm(model + band); kb = L.stats(base)
    def hold(net, mask):
        out = net.copy()
        for j in range(1, net.shape[1]):
            ch = (net[:, j] != net[:, j - 1]) & mask[:, j]; out[:, j] = np.where(ch, out[:, j - 1], net[:, j])
            if j + 1 < net.shape[1]: pass
        return out
    def skip_new(thr):
        f5 = M.g5.copy(); sig = M.grid[None, :] & (colr >= thr); f5 = np.where(sig, np.nan, f5); return C.path(f5, M.g15, 'set', M.grid)
    def delay1(thr):
        flag = M.grid[None, :] & (colr >= thr); out = model.copy()
        for j in range(1, model.shape[1]):
            ch = flag[:, j] & (model[:, j] != model[:, j - 1]); out[:, j] = np.where(ch, model[:, j - 1], model[:, j])
        return out
    def band_skip(thr):
        b = band.copy(); prev = np.zeros(M.n); skip = np.zeros(M.n, bool)
        for j in range(b.shape[1]):
            bj = band[:, j]; new = bj != prev; start = new & (bj != 0)
            hi = np.nanmax(vr[:, max(0, M.dec[j] - 5):M.dec[j]], axis=1) if M.dec[j] >= 5 else np.zeros(M.n)
            skip = np.where(new, start & (np.nan_to_num(hi) >= thr), skip); b[:, j] = np.where(skip, 0.0, bj); prev = bj
        return b
    V = {'V1 model: no new position at volume ratio >= 3': skip_new(3) + band, 'V2 model: no new position at volume ratio >= 5': skip_new(5) + band,
         'V3 model: changes at ratio >= 3 one minute later': delay1(3) + band, 'V4 band: no entry when the entry bar volume ratio >= 3': model + band_skip(3)}
    say(); say("=" * 150); say(f"VB PART 2  WAITING OUT VOLUME BURSTS  vb_burst.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n book, 2.35 ticks; decisions with ratio >= 3: {np.mean(colr[:, M.grid[M.dec - 5] if False else (M.dec % 5 == 0) & (M.dec >= 15)] >= 3):.1%}"); say("=" * 150)
    say(L.line('B0 baseline', base))
    for nm, net in V.items():
        s, _ = C.simm(net); k = L.stats(s); sel = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']
        _, sb = L.halves(s); _, bb = L.halves(base); d = (sb - bb).values; t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        ok = sel and k['shB'] > kb['shB'] and k['ddB'] <= kb['ddB'] and t >= 2.0
        say(f"   {nm:55s} 2013-19 Sharpe {k['shA']:.2f} ({k['shA'] - kb['shA']:+.2f}) DD ${k['ddA']:,.0f} {'SEL' if sel else '   '} | 2019-26 Sharpe {k['shB']:.2f} ({k['shB'] - kb['shB']:+.2f}) "
            f"DD ${k['ddB']:,.0f} ${k['yrB']:,.0f}/yr t {t:+.2f} {'CONFIRMED' if ok else ''}")
    say(f"trial count 10,507; run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'cost': cost()
    elif a == 'run': run()
    else: print(__doc__)
