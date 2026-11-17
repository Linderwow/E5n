r"""BZ (BZ_PLAN.txt): bounce zones on 5 / 15 / 30 / 60 / 240-minute and daily bars - swing levels, flips, channels, weekly levels, round
numbers, stacked zones - traded on 1- and 5-minute limit / rejection / sweep triggers, every combination graded at live cost.
    python bz_zones.py test            hand-made-bar checks
    python bz_zones.py discover        NQ 2010-07 .. 2019-05, all 4,374 cells -> BZ_RESULTS.txt + data\research\bz_cells_nq.csv
    python bz_zones.py discover12      amendment 1: zones on 1- and 2-minute bars, triggers on 1 / 2 minutes
    python bz_zones.py holdout         the cells that cleared the discovery bar on MNQ 2019-26, ES 2010-19, MES 2019-26
"""
import os, sys, time, math
from multiprocessing import Pool
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_paths as PATHS

RES = os.path.join(HERE, 'BZ_RESULTS.txt'); TICK = 0.25; PREF = 27875.0; NUSD = 2 * PREF; COST = 2 * (2.35 * 0.5 + 0.61); K_SW = 3
TFS = (5, 15, 30, 60, 240, 1440); TFNAME = {1: '1m', 2: '2m', 5: '5m', 15: '15m', 30: '30m', 60: '60m', 240: '240m', 1440: 'daily'}
WINDOWS = {'nq': ('2010-07-01', '2019-05-03'), 'mnqx': ('2019-05-06', '2026-10-15'), 'es': ('2010-07-01', '2019-05-03'), 'mes': ('2019-05-06', '2026-10-15')}
TRIG = ('T1 limit', 'T2 reject', 'T3 sweep'); EXITS = ('1R', '2R', 'no target'); CTX = ('all', 'with 60m trend', 'against 60m trend'); SIDES = ('long', 'short', 'both')
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


class D:
    pass


def load(name):
    z = np.load(PATHS.bars_1m(name)); d = D()
    d.t = z['t'].astype('datetime64[m]').astype(np.int64); d.shift = z['shift'].astype(float)
    d.ro = z['o'].astype(float); d.rh = z['h'].astype(float); d.rl = z['l'].astype(float); d.rc = z['c'].astype(float)
    d.o, d.h, d.l, d.c = d.ro + d.shift, d.rh + d.shift, d.rl + d.shift, d.rc + d.shift
    d.md = d.t % 1440; d.tday = (d.t + 360) // 1440; d.rth = (d.md >= 570) & (d.md < 960); d.n = len(d.t)
    last = np.full(d.n, -1, np.int64); idx = np.flatnonzero(d.rth); dd = d.tday[idx]
    ends = idx[np.r_[np.flatnonzero(dd[1:] != dd[:-1]), len(idx) - 1]]; ed = d.tday[ends]
    pos = np.searchsorted(ed, d.tday); ok = (pos < len(ed)); pos = np.clip(pos, 0, len(ed) - 1); ok &= ed[pos] == d.tday
    last[ok] = ends[pos[ok]]; d.send = last
    return d


def zbars(d, tf):
    if tf == 1440: key = d.tday
    else: key = (d.t + (360 if tf == 240 else 0)) // tf
    st = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]; en = np.r_[st[1:] - 1, d.n - 1]
    b = D(); b.o = d.o[st]; b.h = np.maximum.reduceat(d.h, st); b.l = np.minimum.reduceat(d.l, st); b.c = d.c[en]
    b.tstart = d.t[st]; b.tend = d.t[en] + 1
    pc = np.r_[b.c[0], b.c[:-1]]; tr = np.maximum.reduce([b.h - b.l, np.abs(b.h - pc), np.abs(b.l - pc)])
    b.atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values; b.n = len(b.c)
    b.ci = np.searchsorted(b.tend, d.t, side='right') - 1
    return b


def levels(b, maxkeep=60, maxage=3000):
    n = b.n; out = {k: np.full(n, np.nan) for k in ('ssw', 'rsw', 'sfl', 'rfl')}
    ch = {k: np.full((n, 3), np.nan) for k in ('asc_lo', 'asc_up', 'dsc_up', 'dsc_lo')}
    sup, res, fsup, fres = [], [], [], []; lows, highs = [], []; asc = None; dsc = None; h, l, c = b.h, b.l, b.c; ts = b.tstart.astype(float)
    for m in range(n):
        j = m - K_SW
        if j >= K_SW:
            if h[j] > h[j - K_SW:j].max() and h[j] >= h[j + 1:m + 1].max():
                res.append((h[j], j)); highs.append((h[j], j))
                if len(highs) >= 2:
                    (p1, j1), (p2, j2) = highs[-2], highs[-1]
                    if p2 < p1 and j2 > j1:
                        sl = (p2 - p1) / (ts[j2] - ts[j1]); lo = l[j1:j2 + 1].min(); jl = j1 + int(np.argmin(l[j1:j2 + 1]))
                        dsc = (p1, ts[j1], sl, lo - (p1 + sl * (ts[jl] - ts[j1])))
            if l[j] < l[j - K_SW:j].min() and l[j] <= l[j + 1:m + 1].min():
                sup.append((l[j], j)); lows.append((l[j], j))
                if len(lows) >= 2:
                    (p1, j1), (p2, j2) = lows[-2], lows[-1]
                    if p2 > p1 and j2 > j1:
                        sl = (p2 - p1) / (ts[j2] - ts[j1]); hi = h[j1:j2 + 1].max(); jh = j1 + int(np.argmax(h[j1:j2 + 1]))
                        asc = (p1, ts[j1], sl, hi - (p1 + sl * (ts[jh] - ts[j1])))
            lows = lows[-2:]; highs = highs[-2:]
        cm = c[m]
        if sup:
            brk = [s for s in sup if cm < s[0]]
            if brk: sup = [s for s in sup if cm >= s[0]]; fres.extend(brk)
        if res:
            brk = [s for s in res if cm > s[0]]
            if brk: res = [s for s in res if cm <= s[0]]; fsup.extend(brk)
        if fsup: fsup = [s for s in fsup if cm >= s[0]]
        if fres: fres = [s for s in fres if cm <= s[0]]
        sup = [s for s in sup[-maxkeep:] if m - s[1] < maxage]; res = [s for s in res[-maxkeep:] if m - s[1] < maxage]
        fsup = [s for s in fsup[-maxkeep:] if m - s[1] < maxage]; fres = [s for s in fres[-maxkeep:] if m - s[1] < maxage]
        below = [s[0] for s in sup if s[0] <= cm]; above = [s[0] for s in res if s[0] >= cm]
        if below: out['ssw'][m] = max(below)
        if above: out['rsw'][m] = min(above)
        below = [s[0] for s in fsup if s[0] <= cm]; above = [s[0] for s in fres if s[0] >= cm]
        if below: out['sfl'][m] = max(below)
        if above: out['rfl'][m] = min(above)
        te = float(b.tend[m])
        if asc is not None:
            p1, t1, sl, off = asc; lv = p1 + sl * (te - t1)
            if cm < lv: asc = None
            else: ch['asc_lo'][m] = (p1, t1, sl); ch['asc_up'][m] = (p1 + off, t1, sl)
        if dsc is not None:
            p1, t1, sl, off = dsc; lv = p1 + sl * (te - t1)
            if cm > lv: dsc = None
            else: ch['dsc_up'][m] = (p1, t1, sl); ch['dsc_lo'][m] = (p1 + off, t1, sl)
    return out, ch


def per_minute(d, b, lv, ch):
    ci = b.ci; ok = ci >= 0; cc = np.clip(ci, 0, None); tt = d.t.astype(float); Z = {}
    w = np.where(ok, 0.25 * b.atr[cc], np.nan)
    for k in ('ssw', 'rsw', 'sfl', 'rfl'): Z[k] = np.where(ok, lv[k][cc], np.nan)
    for k in ('asc_lo', 'asc_up', 'dsc_up', 'dsc_lo'):
        a = ch[k][cc]; Z[k] = np.where(ok, a[:, 0] + a[:, 2] * (tt - a[:, 1]), np.nan); Z[k + '_id'] = np.where(ok, a[:, 1], np.nan)
    return Z, w


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def zone_sets(d, tfs):
    out = {}; b60 = zbars(d, 60); w60 = np.where(b60.ci >= 0, 0.25 * b60.atr[np.clip(b60.ci, 0, None)], np.nan)
    e50 = ema(b60.c, 50); e200 = ema(b60.c, 200); trend = np.where(b60.ci >= 0, np.sign(e50 - e200)[np.clip(b60.ci, 0, None)], 0.0)
    for tf in tfs:
        b = b60 if tf == 60 else zbars(d, tf); lv, ch = levels(b); Z, w = per_minute(d, b, lv, ch); nm = TFNAME[tf]
        out[f'Z1 swing {nm}'] = (Z['ssw'], Z['rsw'], None, None, w); out[f'Z2 flip {nm}'] = (Z['sfl'], Z['rfl'], None, None, w)
        out[f'Z3 channel with-trend {nm}'] = (Z['asc_lo'], Z['dsc_up'], Z['asc_lo_id'], Z['dsc_up_id'], w)
        out[f'Z4 channel counter {nm}'] = (Z['dsc_lo'], Z['asc_up'], Z['dsc_lo_id'], Z['asc_up_id'], w)
    return out, w60, trend


def extra_sets(d, w60):
    wk = pd.Series(d.tday).map(lambda x: x).values
    days = np.unique(d.tday); dt = pd.to_datetime(days * 86400, unit='s'); iso = dt.isocalendar(); wid = (iso.year.values * 100 + iso.week.values)
    dmap = dict(zip(days, wid)); wofm = np.array([dmap[x] for x in d.tday]) if len(days) < 0 else pd.Series(d.tday).map(dmap).values
    g = pd.DataFrame({'w': wofm, 'h': d.h, 'l': d.l}).groupby('w').agg(h=('h', 'max'), l=('l', 'min')); prev = g.shift(1)
    ws = pd.Series(wofm).map(prev.l).values; wr = pd.Series(wofm).map(prev.h).values
    pcr = np.r_[d.rc[0], d.rc[:-1]]; rs = np.floor(pcr / 100.0) * 100.0 + d.shift; rr = (np.floor(pcr / 100.0) + 1) * 100.0 + d.shift
    return {'Z5 weekly': (ws, wr, None, None, w60), 'Z6 round 100': (rs, rr, None, None, w60)}


def stacked(sets, w60, pc):
    cs = [k for k in sets if k.startswith(('Z1 swing 60m', 'Z1 swing 240m', 'Z1 swing daily', 'Z2 flip 60m', 'Z2 flip 240m', 'Z2 flip daily', 'Z5', 'Z6'))]
    def pick(side, below):
        A = np.column_stack([sets[k][side] for k in cs]); A = np.where((A <= pc[:, None]) if below else (A >= pc[:, None]), A, np.nan)
        A = np.sort(A, axis=1); out = np.full(len(pc), np.nan)
        for a in range(A.shape[1] - 1):
            pair = np.isfinite(A[:, a]) & np.isfinite(A[:, a + 1]) & ((A[:, a + 1] - A[:, a]) <= w60)
            lvl = A[:, a + 1] if below else A[:, a]
            if below: out = np.where(pair, lvl, out)
            else: out = np.where(pair & np.isnan(out), lvl, out)
        return out
    return {'Z7 stacked': (pick(0, True), pick(1, False), None, None, w60)}


def exec_bars(d, e):
    if e == 1: return np.arange(d.n), np.arange(d.n), d.h, d.l, d.c, d.o
    key = d.t // e; st = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]; en = np.r_[st[1:] - 1, d.n - 1]
    return st, en, np.maximum.reduceat(d.h, st), np.minimum.reduceat(d.l, st), d.c[en], d.o[st]


def events(d, zs, e, trend, win):
    sup, res, sid, rid, w = zs; st, en, H, L, C, O = exec_bars(d, e); nb = len(st); out = []
    md0 = d.md[st]; okw = d.rth[st] & (md0 >= 570) & (md0 <= 945 - (e - 1)) & (d.tday[st] >= win[0]) & (d.tday[st] <= win[1])
    pcl = np.r_[np.nan, C[:-1]]
    for side, lvlA, idA in ((1, sup, sid), (-1, res, rid)):
        s = lvlA[st]; ww = w[st]; good = okw & np.isfinite(s) & np.isfinite(ww) & (ww > 0) & np.isfinite(pcl)
        if side == 1:
            outside = pcl > s + ww
            t1 = good & outside & (L < s); t2 = good & outside & (L <= s + ww) & (C > s + ww); t3 = good & outside & (L < s - ww) & (C > s)
        else:
            outside = pcl < s - ww
            t1 = good & outside & (H > s); t2 = good & outside & (H >= s - ww) & (C < s - ww); t3 = good & outside & (H > s + ww) & (C < s)
        ident = idA[st] if idA is not None else np.round(s / TICK)
        for ti, m in enumerate((t1, t2, t3)):
            bi = np.flatnonzero(m)
            if not len(bi): continue
            if ti == 0:
                k = st[bi].copy()
                if e > 1:
                    for q, b_ in enumerate(bi):
                        seg = d.l[st[b_]:en[b_] + 1] if side == 1 else d.h[st[b_]:en[b_] + 1]
                        k[q] = st[b_] + int(np.argmax(seg < s[b_] if side == 1 else seg > s[b_]))
                entry = np.where(side * (d.o[k] - s[bi]) < 0, d.o[k], s[bi]); stop = s[bi] - side * 2 * ww[bi]
            else:
                k = en[bi] + 1; k = np.clip(k, 0, d.n - 1); entry = d.o[k]
                if ti == 1: stop = (np.minimum(L[bi], s[bi] - ww[bi]) - TICK) if side == 1 else (np.maximum(H[bi], s[bi] + ww[bi]) + TICK)
                else: stop = (L[bi] - TICK) if side == 1 else (H[bi] + TICK)
            okk = (k < d.n) & (d.tday[k] == d.tday[st[bi]]) & d.rth[k] & (d.send[k] >= k) & (side * (entry - stop) >= 2 * TICK)
            tr = trend[k]
            out.append(pd.DataFrame({'trig': ti, 'side': side, 'k': k[okk], 'entry': entry[okk], 'stop': stop[okk], 'day': d.tday[k[okk]], 'id': ident[bi][okk],
                                     'ctx': (tr[okk] * side)}))
    if not out: return pd.DataFrame(columns=['trig', 'side', 'k', 'entry', 'stop', 'day', 'id', 'ctx'])
    ev = pd.concat(out, ignore_index=True).drop_duplicates(['trig', 'side', 'day', 'id'], keep='first')
    return ev.sort_values('k', kind='stable').reset_index(drop=True)


def outcomes(d, ev):
    n = len(ev); res = np.zeros((n, 3)); xi = np.zeros((n, 3), np.int64)
    if n == 0: return res, xi
    k = ev.k.values; side = ev.side.values.astype(float); en = ev.entry.values; sp = ev.stop.values; R = side * (en - sp); se = d.send[k]; lim0 = (ev.trig.values == 0)
    span = int((se - k).max()) + 1; CH = 4000
    for a in range(0, n, CH):
        sl = slice(a, min(n, a + CH)); kk = k[sl]; m = len(kk); idx = kk[:, None] + np.arange(span)[None, :]; lim = se[sl][:, None]
        valid = idx <= lim; idx = np.minimum(idx, d.n - 1)
        hh, ll, oo = d.h[idx], d.l[idx], d.o[idx]; sd = side[sl][:, None]
        adv = np.where(sd > 0, ll, -hh); fav = np.where(sd > 0, hh, -ll); so = np.where(sd > 0, oo, -oo)
        e_ = (sd[:, 0] * en[sl]); s_ = (sd[:, 0] * sp[sl]); r_ = R[sl]
        hit_s = valid & (adv <= s_[:, None]); fs = np.where(hit_s.any(1), hit_s.argmax(1), span)
        last = (se[sl] - kk); cl = sd[:, 0] * d.c[se[sl]]
        for x, mult in enumerate((1.0, 2.0, None)):
            if mult is None:
                ft = np.full(m, span)
            else:
                tg = e_ + mult * r_; hit_t = valid & (fav >= tg[:, None]); hit_t[:, 0] &= ~lim0[sl]; ft = np.where(hit_t.any(1), hit_t.argmax(1), span)
            stop_first = fs <= ft; none = (fs >= span) & (ft >= span)
            j = np.where(none, last, np.where(stop_first, fs, ft)); jj = np.clip(j, 0, span - 1); rows = np.arange(m)
            o_at = so[rows, jj]
            px = np.where(none, cl, np.where(stop_first, np.minimum(o_at, s_) if True else s_, np.maximum(o_at, e_ + (mult or 0) * r_)))
            px = np.where(stop_first & ~none & (jj == 0), s_, px)
            res[sl, x] = px - e_; xi[sl, x] = kk + j
    return res, xi


def grade_cells(d, ev, res, xi, days, raw_entry):
    out = []; usd = res / raw_entry[:, None] * NUSD - COST; usd0 = res / raw_entry[:, None] * NUSD
    dayix = np.searchsorted(days, ev.day.values); kk = ev.k.values
    for ti in range(3):
        for ci, cm in enumerate((None, 1.0, -1.0)):
            for si, sm in enumerate((1, -1, 0)):
                m = (ev.trig.values == ti) & ((ev.side.values == sm) if sm else True) & ((ev.ctx.values == cm) if cm is not None else True)
                ix = np.flatnonzero(m)
                for x in range(3):
                    take = []; free = -1
                    for i in ix:
                        if kk[i] > free: take.append(i); free = xi[i, x]
                    take = np.array(take, np.int64)
                    dl = np.bincount(dayix[take], weights=usd[take, x], minlength=len(days)) if len(take) else np.zeros(len(days))
                    d0 = np.bincount(dayix[take], weights=usd0[take, x], minlength=len(days)) if len(take) else np.zeros(len(days))
                    out.append((ti, ci, si, x, len(take), dl.astype(np.float32), d0.astype(np.float32)))
    return out


def tstat(x):
    s = x.std(); return float(x.mean() / s * math.sqrt(len(x))) if s > 0 else 0.0


def worker(args):
    name, group = args[0], args[1]; execs = args[2] if len(args) > 2 else (1, 5); d = load(name); win = [int((pd.Timestamp(x).value // 10 ** 9 + 0) // 86400) for x in WINDOWS[name]]
    days = np.unique(d.tday[d.rth & (d.tday >= win[0]) & (d.tday <= win[1])]); t0 = time.time()
    if group == 'extra':
        sets, w60, trend = zone_sets(d, (60, 240, 1440)); ex = extra_sets(d, w60); pc = np.r_[d.c[0], d.c[:-1]]
        allz = dict(sets); allz.update(ex); st = stacked(allz, w60, pc); sets = dict(ex); sets.update(st)
    else:
        sets, w60, trend = zone_sets(d, (group,))
    rows = []; series = {}
    for zname, zs in sets.items():
        for e in execs:
            ev = events(d, zs, e, trend, win); res, xi = outcomes(d, ev); raw = ev.entry.values - d.shift[ev.k.values] if len(ev) else np.zeros(0)
            for (ti, ci, si, x, ntr, dl, d0) in grade_cells(d, ev, res, xi, days, raw):
                cid = f"{zname} | {e}m {TRIG[ti]} | {CTX[ci]} | {SIDES[si]} | {EXITS[x]}"
                rows.append(dict(cell=cid, zone=zname, exe=e, trig=ti, ctx=ci, side=si, exit=x, trades_yr=ntr / (len(days) / 252), usd_yr=float(dl.mean() * 252),
                                 t=tstat(dl), t0=tstat(d0), wr=np.nan))
                series[cid] = dl
    return rows, series, days, time.time() - t0


def discover(small=False):
    t0 = time.time(); groups = [('nq', 1, (1, 2)), ('nq', 2, (1, 2))] if small else [('nq', tf) for tf in TFS] + [('nq', 'extra')]
    with Pool(len(groups)) as p: outs = p.map(worker, groups)
    rows = [r for o in outs for r in o[0]]; series = {}; [series.update(o[1]) for o in outs]; days = outs[0][2]
    T = pd.DataFrame(rows); K = len(T); from scipy.stats import norm; bar = float(norm.ppf(1 - 0.025 / (K + (4374 if small else 0))))
    dts = pd.to_datetime(days * 86400, unit='s'); yrs = dts.year.values
    def extra(cid):
        x = series[cid]; yy = pd.Series(x).groupby(yrs).sum(); xs = np.sort(x)[::-1]; d5 = x.copy(); d5[np.argsort(x)[::-1][:5]] = 0
        return int((yy < 0).sum()), len(yy), tstat(d5)
    tag = 'nq_1m2m' if small else 'nq'; T.to_csv(os.path.join(PATHS.CACHE, f'bz_cells_{tag}.csv'), index=False)
    np.savez_compressed(os.path.join(PATHS.CACHE, f'bz_series_{tag}.npz'), days=days, **{f's{i}': series[c] for i, c in enumerate(T.cell)})
    say(); say("=" * 170)
    say(f"BZ{' AMENDMENT 1: ZONES ON 1- AND 2-MINUTE BARS, TRIGGERS ON 1 / 2 MINUTES' if small else ''}  BOUNCE ZONES, EVERY TIMEFRAME AND COMBINATION  bz_zones.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (BZ_PLAN.txt)  DISCOVERY NQ 2010-07 .. 2019-05, {len(days):,} sessions; "
        f"1 micro, $ at MNQ 27,875, live cost ${COST:.2f} a round trip")
    say("=" * 170)
    say(f"{K:,} cells{' (bar counted over 5,670 BZ cells)' if small else ''}; Bonferroni bar |t| >= {bar:.2f}. Cells beyond |t| 3: {int((T.t.abs() >= 3).sum())} (chance alone about {K * 0.0027:.0f} if independent); beyond |t| 2: "
        f"{int((T.t.abs() >= 2).sum())} (chance {K * 0.0455:.0f}); positive at live cost: {(T.t > 0).mean():.0%}; positive at ZERO cost: {(T.t0 > 0).mean():.0%}")
    say(f"median trades a year {T.trades_yr.median():.0f}; median $ a year {T.usd_yr.median():+,.0f}; median zero-cost t {T.t0.median():+.2f}")
    say(); say("BEST CELL OF EACH ZONE TYPE AND TIMEFRAME (by t at live cost):")
    for z, g in T.groupby('zone', sort=False):
        r = g.loc[g.t.idxmax()]; ly, ny, d5 = extra(r.cell)
        say(f"   {z:28s} best: {r.cell.split(' | ', 1)[1]:62s} t {r.t:+5.2f} (zero cost {r.t0:+5.2f}) ${r.usd_yr:+8,.0f}/yr {r.trades_yr:5.0f} trades/yr, losing yrs {ly}/{ny}, drop-5 t {d5:+.2f}"
            f" | zone median t {g.t.median():+.2f}, best zero-cost t {g.t0.max():+.2f}")
    say(); say("BY TRIGGER / EXIT / CONTEXT / SIDE (median t at live cost | median zero-cost t | share of cells positive at zero cost):")
    for col, names in (('trig', TRIG), ('exit', EXITS), ('ctx', CTX), ('side', SIDES), ('exe', None)):
        for v, g in T.groupby(col):
            lab = names[v] if names else f"{v}-minute triggers"
            say(f"   {lab:22s} {g.t.median():+.2f} | {g.t0.median():+.2f} | {(g.t0 > 0).mean():.0%}  ({len(g)} cells)")
    say(); say("TOP 15 CELLS (any zone):")
    for _, r in T.sort_values('t', ascending=False).head(15).iterrows():
        ly, ny, d5 = extra(r.cell); say(f"   t {r.t:+5.2f} (zero cost {r.t0:+5.2f}) ${r.usd_yr:+8,.0f}/yr {r.trades_yr:5.0f}/yr losing yrs {ly}/{ny} drop-5 {d5:+.2f}  {r.cell}")
    say("BOTTOM 5 CELLS (most negative - the zone breaks instead of holding):")
    for _, r in T.sort_values('t').head(5).iterrows(): say(f"   t {r.t:+5.2f} (zero cost {r.t0:+5.2f}) ${r.usd_yr:+8,.0f}/yr {r.trades_yr:5.0f}/yr  {r.cell}")
    surv = T[T.t >= bar].sort_values('t', ascending=False).drop_duplicates('zone').head(10)
    say(); say(f"CLEARING THE BAR (t >= {bar:.2f}): {len(T[T.t >= bar])} cells" + ("" if len(surv) else " -> NOTHING to take to the holdout; MNQ 2019-26 stays unopened."))
    for _, r in surv.iterrows(): say(f"   {r.cell}  t {r.t:+.2f}")
    surv[['cell', 'zone', 'exe', 'trig', 'ctx', 'side', 'exit']].to_csv(os.path.join(PATHS.CACHE, f'bz_survivors_{tag}.csv'), index=False)
    say(f"run {time.time() - t0:.0f} s (workers: {', '.join(f'{o[3]:.0f}s' for o in outs)})")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


def holdout():
    cells = {'C1': ('Z3 channel with-trend 240m | 5m T2 reject | all | short | no target', 240, (5,)),
             'C2': ('Z7 stacked | 5m T3 sweep | against 60m trend | long | 1R', 'extra', (5,)),
             'C3': ('Z4 channel counter 1m | 1m T2 reject | all | long | no target', 1, (1,))}
    jobs = [(m, g, e) for m in ('mnqx', 'es', 'mes') for (_, g, e) in cells.values()]
    with Pool(len(jobs)) as p: outs = p.map(worker, jobs)
    say(); say("=" * 170); say(f"BZ AMENDMENT 2  THE 3 BEST CELLS ON DATA NEVER USED FOR THIS QUESTION  bz_zones.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  live cost, daily P&L"); say("=" * 170)
    res = {}
    for (m, g, e), o in zip(jobs, outs):
        T = pd.DataFrame(o[0]).set_index('cell')
        for code, (cid, gg, ee) in cells.items():
            if gg == g and cid in T.index:
                r = T.loc[cid]; res[(code, m)] = r; yrs = pd.to_datetime(o[2] * 86400, unit='s').year.values; x = o[1][cid]
                yy = pd.Series(x).groupby(yrs).sum(); say(f"   {code} {m:5s} t {r.t:+5.2f} (zero cost {r.t0:+5.2f}) ${r.usd_yr:+8,.0f}/yr {r.trades_yr:5.0f} trades/yr, losing years {int((yy < 0).sum())}/{len(yy)}   {cid}")
    for code, (cid, _, _) in cells.items():
        a, b, c = res[(code, 'mnqx')], res[(code, 'es')], res[(code, 'mes')]
        ok = a.t >= 2.5 and b.t > 0 and c.t > 0
        say(f"   -> {code}: {'CANDIDATE (forward shadow next, nothing traded)' if ok else 'dead'}  (MNQ t {a.t:+.2f} vs 2.5; ES {b.t:+.2f}; MES {c.t:+.2f})")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


def test():
    d = D(); n = 60; base = 27000.0 * 0 + 1440 * 20000
    d.t = np.arange(n, dtype=np.int64) + base + 570; d.shift = np.zeros(n)
    c = np.array([110, 108, 106, 104, 100, 103, 106, 108, 110, 112, 114, 116, 115, 113, 111, 108, 106, 104, 103, 102, 101.5, 103, 105, 108, 110] + [111] * 35, float)
    d.rc = c.copy(); d.ro = np.r_[c[0], c[:-1]]; d.rh = np.maximum(d.ro, d.rc) + 0.5; d.rl = np.minimum(d.ro, d.rc) - 0.5
    d.o, d.h, d.l, d.c = d.ro, d.rh, d.rl, d.rc; d.md = d.t % 1440; d.tday = (d.t + 360) // 1440; d.rth = (d.md >= 570) & (d.md < 960); d.n = n; d.send = np.full(n, n - 1)
    b = D(); b.h, b.l, b.c = d.h, d.l, d.c; b.tstart = d.t; b.tend = d.t + 1; b.n = n; b.atr = np.full(n, 2.0); b.ci = np.arange(n) - 1
    lv, ch = levels(b)
    ok1 = lv['ssw'][7] == 99.5 and np.isnan(lv['ssw'][6])
    ok2 = lv['rsw'][14] == 116.5
    ev = pd.DataFrame({'trig': [1], 'side': [1], 'k': [22], 'entry': [103.0], 'stop': [100.0], 'day': [d.tday[22]], 'id': [0], 'ctx': [0]})
    r, xi = outcomes(d, ev); ok3 = abs(r[0, 0] - 3.0) < 1e-9 and xi[0, 0] == 23
    ev2 = pd.DataFrame({'trig': [1], 'side': [-1], 'k': [22], 'entry': [103.0], 'stop': [104.0], 'day': [d.tday[22]], 'id': [0], 'ctx': [0]})
    r2, xi2 = outcomes(d, ev2); ok4 = abs(r2[0, 0] - (-1.0)) < 1e-9 and abs(r2[0, 2] - (-1.0)) < 1e-9
    d2 = D(); d2.__dict__.update(d.__dict__); d2.o = d.o.copy(); d2.h = d.h.copy(); d2.l = d.l.copy(); d2.l[23] = 95.0; d2.h[23] = 112.0
    r3, _ = outcomes(d2, ev); ok5 = abs(r3[0, 1] - (-3.0)) < 1e-9 or abs(r3[0, 1] - (min(d2.o[23], 100.0) - 103.0)) < 1e-9
    for nm, ok in (('swing low confirmed 3 bars later, not before', ok1), ('swing high level', ok2), ('long 1R target reached on the next minute', ok3),
                   ('short stopped when price rises through the stop', ok4), ('stop and target in one minute -> stop', ok5)):
        print(f"{'PASS' if ok else 'FAIL'}  {nm}")
    return all((ok1, ok2, ok3, ok4, ok5))


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'test': sys.exit(0 if test() else 1)
    elif a == 'discover': discover()
    elif a == 'discover12': discover(small=True)
    elif a == 'holdout': holdout()
    else: print(__doc__)
