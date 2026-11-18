r"""NT (NT_PLAN.txt): E5n's band leg at night (Asia / Europe / whole night) and the trend-filtered overnight hold.
    python nt_night.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_calendar as CAL
import red_rules as R

NUSD = 55750.0; SIDE = 4.5 * 0.25 * 2 + 0.61; JOIN = pd.Timestamp('2019-05-06'); OUT = os.path.join(HERE, 'NT_RESULTS.txt'); LINES = []
WIN = {'NB1 Asia 18:00-00:30': (18 * 60, 30, 390), 'NB2 Europe 02:00-08:30': (2 * 60, 0, 390), 'NB3 whole night 18:00-09:25': (18 * 60, 30, 925)}


def say(s=''):
    print(s, flush=True); LINES.append(s)


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def frame(name):
    d = R.load(name); t = pd.to_datetime(d['t'] * 60, unit='s')
    df = pd.DataFrame({'t': t, 'o': d['o'], 'h': d['h'], 'l': d['l'], 'c': d['c'], 'v': d['v'], 'tday': d['tday']}).set_index('t')
    f5 = df.resample('5min', label='right', closed='left').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'}).dropna()
    tr = np.maximum.reduce([f5.h - f5.l, (f5.h - f5.c.shift()).abs(), (f5.l - f5.c.shift()).abs()])
    f5['atr'] = pd.Series(np.asarray(tr), index=f5.index).ewm(alpha=1 / 14, adjust=False).mean(); f5['trend'] = np.sign(ema(f5.c.values, 50) - ema(f5.c.values, 200))
    return df, f5


def band_night(df, f5, start, offset, length):
    T1 = df.index.values.astype('datetime64[m]').astype(np.int64); O1 = df.o.values; C1 = df.c.values
    T5 = f5.index.values.astype('datetime64[m]').astype(np.int64); C5 = f5.c.values; H5 = f5.h.values; L5 = f5.l.values; V5 = f5.v.values
    A5 = f5.atr.values; TR5 = f5.trend.values
    rmask = (T1 % 1440) == 959; RT = T1[rmask]; RC = C1[rmask]
    res = {}; hist = []
    d0 = int(T1[0] // 1440); d1 = int(T1[-1] // 1440)
    for day in range(d0, d1 + 1):
        s = day * 1440 + start; e = s + length; i0, i1 = np.searchsorted(T1, s), np.searchsorted(T1, e)
        if i1 - i0 < length * 0.6: continue
        o0 = O1[i0]; mv = np.full(length, np.nan); mv[(T1[i0:i1] - s).astype(int)] = np.abs(C1[i0:i1] / o0 - 1)
        wid = np.nanmean(np.array(hist[-14:]), axis=0) if len(hist) >= 7 else None
        hist.append(mv)
        if wid is None: continue
        if start >= 600:
            k = np.searchsorted(RT, s) - 1
            if k < 0: continue
            pc = RC[k]
        else:
            k = np.searchsorted(T1, day * 1440 + 30) - 1; pc = C1[k] if k >= 0 else o0
        hi, lo = max(o0, pc), min(o0, pc); j0, j1 = np.searchsorted(T5, s + 1), np.searchsorted(T5, e + 1)
        if j1 - j0 < 10: continue
        tp = (H5[j0:j1] + L5[j0:j1] + C5[j0:j1]) / 3; cv = np.cumsum(tp * V5[j0:j1]) / np.maximum(np.cumsum(V5[j0:j1]), 1)
        e20 = ema(C5[j0:j1], 20); usd = 0.0; pos = 0; ent = best = 0.0; inside = 0
        for q, k in enumerate(range(j0, j1)):
            tm = int(T5[k] - s); wk = wid[min(max(tm - 1, 0), length - 1)]
            if not np.isfinite(wk): continue
            ub, lb = hi * (1 + wk), lo * (1 - wk); c = C5[k]; fi = np.searchsorted(T1, T5[k])
            if fi >= len(O1): break
            fill = O1[fi]
            if pos:
                best = max(best, c) if pos > 0 else min(best, c)
                inside = inside + 1 if (pos > 0 and c < max(ub, cv[q])) or (pos < 0 and c > min(lb, cv[q])) else 0
                if inside >= 4 or abs(c - best) >= 3 * A5[k] or tm >= length - 5:
                    usd += pos * (fill - ent) / ent * NUSD - SIDE; pos = 0; inside = 0
                continue
            if tm % 30 == 0 and 30 <= tm <= length - 30:
                if c > ub and c > e20[q] and TR5[k] > 0: pos, ent, best, inside = 1, fill, c, 0; usd -= SIDE
                elif c < lb and c < e20[q] and TR5[k] < 0: pos, ent, best, inside = -1, fill, c, 0; usd -= SIDE
        if pos: usd += pos * (C1[i1 - 1] - ent) / ent * NUSD - SIDE
        res[pd.Timestamp((s + 360) // 1440 * 86400, unit='s')] = usd
    return pd.Series(res, dtype=float)


def hold(df, f5, both):
    T1 = df.index.values.astype('datetime64[m]').astype(np.int64); O1 = df.o.values; C1 = df.c.values
    T5 = f5.index.values.astype('datetime64[m]').astype(np.int64); TR5 = f5.trend.values; res = {}
    for day in range(int(T1[0] // 1440), int(T1[-1] // 1440) + 1):
        s = day * 1440 + 18 * 60 + 5; e = (day + 1) * 1440 + 9 * 60 + 29
        i = np.searchsorted(T1, s); j = np.searchsorted(T1, e, side='right') - 1; k = np.searchsorted(T5, day * 1440 + 18 * 60, side='right') - 1
        if i >= len(T1) or T1[i] - s > 10 or j < 0 or e - T1[j] > 10 or k < 0: continue
        sg = TR5[k] if both else max(TR5[k], 0); key = pd.Timestamp((day + 1) * 86400, unit='s')
        res[key] = 0.0 if sg == 0 else sg * (C1[j] - O1[i]) / O1[i] * NUSD - 2 * SIDE
    return pd.Series(res, dtype=float)


def grade(x, lab, lo, hi):
    x = x[(x.index >= lo) & (x.index < hi)]; x = x[[CAL.full_session(d.date()) for d in x.index]]
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan; yrs = x.groupby(x.index.year).sum()
    say(f"   {lab:44s} days {len(x):5d} traded {int((x != 0).sum()):5d} | ${x.mean() * 252:+8,.0f}/yr  t {t:+5.2f}  losing years {int((yrs < 0).sum())}/{len(yrs)}  "
        f"drop-5 ${x.sum() - x.nlargest(5).sum():+,.0f}  max DD ${float((x.cumsum().cummax() - x.cumsum()).max()):,.0f}")
    return t, int((yrs < 0).sum()), len(yrs), x.sum() - x.nlargest(5).sum()


def main():
    t0 = time.time(); global SIDE
    say(); say("=" * 150); say(f"NT  E5n AT NIGHT  nt_night.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  $ at MNQ 27,875, $5.72 a round trip (4.5 ticks + $0.61 a side)"); say("=" * 150)
    nq = frame('nq'); alive = []
    for nm, (st, off, ln) in WIN.items():
        x = band_night(*nq, st, off, ln); t, ly, ny, d5 = grade(x, nm, '2010-07-01', '2019-05-06')
        keep = SIDE; SIDE = 0.0; grade(band_night(*nq, st, off, ln), nm + ' (zero cost)', '2010-07-01', '2019-05-06'); SIDE = keep
        ok = t >= 2.6 and ly <= ny / 2 and d5 > 0; say(f"      -> {'SURVIVES discovery' if ok else 'DEAD (kill rule)'}")
        if ok: alive.append(('band', nm, st, off, ln))
    for both, nm in ((False, 'NH1 hold long 18:05-09:29 with trend'), (True, 'NH2 hold long/short 18:05-09:29 by trend')):
        x = hold(*nq, both); t, ly, ny, d5 = grade(x, nm, '2010-07-01', '2019-05-06')
        keep = SIDE; SIDE = 0.0; grade(hold(*nq, both), nm + ' (zero cost)', '2010-07-01', '2019-05-06'); SIDE = keep
        ok = t >= 2.6 and ly <= ny / 2 and d5 > 0; say(f"      -> {'SURVIVES discovery' if ok else 'DEAD (kill rule)'}")
        if ok: alive.append(('hold', nm, both))
    if alive:
        mnq = frame('mnqx'); es = frame('es')
        for a in alive:
            if a[0] == 'band':
                grade(band_night(*mnq, *a[2:]), a[1] + ' HOLDOUT MNQ', '2019-05-06', '2026-11-01'); grade(band_night(*es, *a[2:]), a[1] + ' ES discovery', '2010-07-01', '2019-05-06')
            else:
                grade(hold(*mnq, a[2]), a[1] + ' HOLDOUT MNQ', '2019-05-06', '2026-11-01'); grade(hold(*es, a[2]), a[1] + ' ES discovery', '2010-07-01', '2019-05-06')
    else: say("   no survivor - holdout not opened")
    say(f"   trial count +5; run {time.time() - t0:.0f} s")
    with open(OUT, 'a', encoding='utf-8') as f: f.write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
