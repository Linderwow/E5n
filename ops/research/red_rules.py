r"""RED (RED_PLAN.txt): mechanical rules from the Reddit scout - VIX Fix pullbacks, Bollinger squeeze + CCI, momentum-decay reversal.
    python red_rules.py
"""
import os, sys, math, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_paths as P

RES = os.path.join(HERE, 'RED_RESULTS.txt'); LINES = []; TICK = 0.25; NUSD = 2 * 27875.0; COST = 2 * (2.35 * 0.5 + 0.61)
WIN = {'nq': ('2010-07-01', '2019-05-03'), 'mnqx': ('2019-05-06', '2026-10-15'), 'es': ('2010-07-01', '2019-05-03'), 'mes': ('2019-05-06', '2026-10-15')}
NAMES = {'V1': 'VIX Fix pullback, long only, 3-min, 09:30-11:00', 'V2': 'VIX Fix pullback, both sides', 'S1': 'Bollinger squeeze + CCI + daily trend, 5-min, 25-pt target, no stop',
         'S2': 'S1 with a stop at the opposite band', 'M1': 'momentum-decay reversal, 1-min, 08:00-16:00'}


def say(s=''):
    print(s, flush=True); LINES.append(s)


def load(name):
    z = np.load(P.bars_1m(name)); d = {}
    t = z['t'].astype('datetime64[m]').astype(np.int64); sh = z['shift'].astype(float)
    d.update(t=t, sh=sh, o=z['o'] + sh, h=z['h'] + sh, l=z['l'] + sh, c=z['c'] + sh, v=z['v'].astype(float), md=t % 1440, tday=(t + 360) // 1440)
    return d


def bars(d, e):
    if e == 1:
        return dict(o=d['o'], h=d['h'], l=d['l'], c=d['c'], v=d['v'], md=d['md'], tday=d['tday'], sh=d['sh'])
    key = d['t'] // e; st = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]; en = np.r_[st[1:] - 1, len(key) - 1]
    return dict(o=d['o'][st], h=np.maximum.reduceat(d['h'], st), l=np.minimum.reduceat(d['l'], st), c=d['c'][en], v=np.add.reduceat(d['v'], st),
                md=d['md'][st], tday=d['tday'][st], sh=d['sh'][st])


def wilder(b, n):
    pc = np.r_[b['c'][0], b['c'][:-1]]; tr = np.maximum.reduce([b['h'] - b['l'], np.abs(b['h'] - pc), np.abs(b['l'] - pc)])
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def vwap(b):
    rth = (b['md'] >= 570) & (b['md'] < 960); tp = (b['h'] + b['l'] + b['c']) / 3
    df = pd.DataFrame({'d': b['tday'], 'pv': np.where(rth, tp * b['v'], 0.0), 'v': np.where(rth, b['v'], 0.0)})
    cpv = df.groupby('d').pv.cumsum().values; cv = df.groupby('d').v.cumsum().values
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(rth & (cv > 0), cpv / cv, np.nan)


def last_rth(b):
    rth = (b['md'] >= 570) & (b['md'] < 960); idx = np.flatnonzero(rth); dd = b['tday'][idx]
    ends = idx[np.r_[np.flatnonzero(dd[1:] != dd[:-1]), len(idx) - 1]]
    return dict(zip(b['tday'][ends], ends))


def run_trades(b, signals, stop_of, target_of, exit_sig=None, buy_stop=False, day_rule=False, first_md=570, last_entry_md=955):
    n = len(b['c']); lr = last_rth(b); trades = []; pos = 0; i = 1
    while i < n:
        side = signals[i - 1]
        if pos == 0 and side != 0 and b['tday'][i] == b['tday'][i - 1] and first_md <= b['md'][i] <= last_entry_md and b['tday'][i] in lr and i <= lr[b['tday'][i]]:
            day = b['tday'][i]
            if day_rule and (done.get(day, False) or b['md'][i] >= 660): i += 1; continue
            if buy_stop:
                trig = b['h'][i - 1] + TICK if side > 0 else b['l'][i - 1] - TICK
                if (side > 0 and b['h'][i] < trig) or (side < 0 and b['l'][i] > trig): i += 1; continue
                entry = max(b['o'][i], trig) if side > 0 else min(b['o'][i], trig); fill_bar = True
            else:
                entry = b['o'][i]; fill_bar = False
            stop = stop_of(i - 1, side, entry); tgt = target_of(i - 1, side, entry, stop); end = lr[day]; px = None; j = i
            while j <= end:
                if stop is not None and ((side > 0 and b['l'][j] <= stop) or (side < 0 and b['h'][j] >= stop)):
                    px = stop if j == i else (min(b['o'][j], stop) if side > 0 else max(b['o'][j], stop)); break
                if tgt is not None and not (fill_bar and j == i) and ((side > 0 and b['h'][j] >= tgt) or (side < 0 and b['l'][j] <= tgt)):
                    px = tgt if j == i else (max(b['o'][j], tgt) if side > 0 else min(b['o'][j], tgt)); break
                if exit_sig is not None and j < end and exit_sig(j, side): px = b['o'][j + 1]; j += 1; break
                j += 1
            if px is None: j = end; px = b['c'][end]
            pts = side * (px - entry); raw = entry - b['sh'][i]; trades.append((day, pts / raw * NUSD))
            if day_rule and pts >= 10: done[day] = True
            i = j + 1; continue
        i += 1
    return trades


done = {}


def trials(name):
    global done
    d = load(name); out = {}
    b3 = bars(d, 3); a3 = wilder(b3, 14); vw = vwap(b3); c = b3['c']; s = pd.Series(c)
    hh = s.rolling(22).max().values; ll = s.rolling(22).min().values
    wv = (hh - b3['l']) / hh * 100; wvi = (b3['h'] - ll) / ll * 100
    def spike(w):
        ws = pd.Series(w); up = ws.rolling(20).mean() + 2 * ws.rolling(20).std(); rh = ws.rolling(50).max() * 0.85
        return ((ws >= up) | (ws >= rh)).values
    sp, spi = spike(wv), spike(wvi); vw5 = np.r_[np.full(5, np.nan), vw[:-5]]
    win = (b3['md'] >= 570) & (b3['md'] <= 657)
    upt = win & (c > vw) & (vw > vw5); dnt = win & (c < vw) & (vw < vw5)
    lo5 = pd.Series(b3['l']).rolling(5).min().values; hi5 = pd.Series(b3['h']).rolling(5).max().values
    def stopV(k, side, entry):
        a = math.ceil(a3[k] / 5) * 5
        return min(entry - a, lo5[k] - TICK) if side > 0 else max(entry + a, hi5[k] + TICK)
    tgt2 = lambda k, side, entry, stop: entry + side * 2 * abs(entry - stop)
    for code, sig in (('V1', np.where(upt & sp, 1, 0)), ('V2', np.where(upt & sp, 1, np.where(dnt & spi, -1, 0)))):
        done = {}; out[code] = run_trades(b3, sig, stopV, tgt2, buy_stop=True, day_rule=True, last_entry_md=659)
    b5 = bars(d, 5); c5 = pd.Series(b5['c']); m20 = c5.rolling(20).mean(); sd20 = c5.rolling(20).std(ddof=0)
    bu, bl = (m20 + 2 * sd20).values, (m20 - 2 * sd20).values; e20 = c5.ewm(span=20, adjust=False).mean().values; a20 = wilder(b5, 20)
    on = (bu < e20 + 1.5 * a20) & (bl > e20 - 1.5 * a20); rel = np.r_[False, on[:-1] & ~on[1:]]
    tp = pd.Series((b5['h'] + b5['l'] + b5['c']) / 3); md_ = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = ((tp - tp.rolling(20).mean()) / (0.015 * md_)).values
    lr5 = last_rth(b5); dc = pd.Series({k: b5['c'][v] for k, v in lr5.items()}).sort_index(); prevc = dc.shift(1); sma = dc.rolling(20).mean().shift(1)
    trend = pd.Series(np.sign(prevc - sma)).reindex(b5['tday']).values if False else pd.Series(b5['tday']).map(np.sign(prevc - sma)).values
    rthsig = (b5['md'] >= 570) & (b5['md'] <= 950)
    sigS = np.where(rthsig & rel & (cci > 100) & (trend > 0), 1, np.where(rthsig & rel & (cci < -100) & (trend < 0), -1, 0))
    out['S1'] = run_trades(b5, sigS, lambda k, side, e: None, lambda k, side, e, s_: e + side * 25.0)
    out['S2'] = run_trades(b5, sigS, lambda k, side, e: (bl[k] if side > 0 else bu[k]), lambda k, side, e, s_: e + side * 25.0)
    b1 = bars(d, 1); h1, l1, c1 = pd.Series(b1['h']), pd.Series(b1['l']), b1['c']
    mid = ((h1.rolling(30).max() + l1.rolling(30).min()) / 2).values; dd = c1 - mid; a30 = wilder(b1, 30)
    x = np.arange(10) - 4.5; wts = x / (x * x).sum(); slope = np.r_[np.full(9, np.nan), np.convolve(dd, wts[::-1], mode='valid')]
    sp_ = np.r_[np.nan, slope[:-1]]
    fomc = set()
    fp = os.path.join(P.WEB, 'fomc_dates.txt')
    if os.path.exists(fp):
        for s_ in open(fp):
            s_ = s_.strip()
            if s_ and not s_.startswith('#'): fomc.add(int(pd.Timestamp(s_).value // 86_400_000_000_000))
    w1 = (b1['md'] >= 480) & (b1['md'] < 959) & ~np.isin(b1['tday'], list(fomc))
    sigM = np.where(w1 & (dd >= a30) & (sp_ > 0) & (slope <= 0), -1, np.where(w1 & (dd <= -a30) & (sp_ < 0) & (slope >= 0), 1, 0))
    exitM = lambda j, side: (side < 0 and sp_[j] < 0 and slope[j] >= 0) or (side > 0 and sp_[j] > 0 and slope[j] <= 0)
    out['M1'] = run_trades(b1, sigM, lambda k, side, e: None, lambda k, side, e, s_: None, exit_sig=exitM, first_md=480, last_entry_md=958)
    days = np.unique(d['tday'][(d['md'] >= 570) & (d['md'] < 960)])
    return out, days


def grade(trades, days, a, b, cost=True):
    a = int(pd.Timestamp(a).value // 86_400_000_000_000); b = int(pd.Timestamp(b).value // 86_400_000_000_000); dd = days[(days >= a) & (days <= b)]
    tr = [(dy, u) for dy, u in trades if a <= dy <= b]; ix = {v: i for i, v in enumerate(dd)}; x = np.zeros(len(dd)); x0 = np.zeros(len(dd))
    for dy, u in tr:
        if dy in ix: x[ix[dy]] += u - COST; x0[ix[dy]] += u
    t = lambda y: float(y.mean() / y.std() * math.sqrt(len(y))) if y.std() > 0 else 0.0
    yrs = pd.to_datetime(dd * 86400, unit='s').year; yy = pd.Series(x).groupby(yrs).sum()
    w = np.array([u for _, u in tr]) - COST
    return dict(n=len(tr), yr=len(tr) / (len(dd) / 252), t=t(x), t0=t(x0), usd=x.mean() * 252, lose=int((yy < 0).sum()), ny=len(yy), win=float((w > 0).mean()) if len(w) else np.nan)


def main():
    t0 = time.time(); say(); say("=" * 160); say(f"RED  RULES FROM THE REDDIT SCOUT  red_rules.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (RED_PLAN.txt) $ at MNQ 27,875, live cost ${COST:.2f} a round trip"); say("=" * 160)
    out, days = trials('nq'); surv = []
    for k in NAMES:
        g = grade(out[k], days, *WIN['nq']); ok = g['t'] >= 2.6 and g['lose'] <= g['ny'] / 2
        say(f"   {k} {NAMES[k]:62s} | NQ 2010-19: {g['n']:6,d} trades ({g['yr']:5.0f}/yr) winners {g['win']:.0%} t {g['t']:+6.2f} (zero cost {g['t0']:+6.2f}) ${g['usd']:+8,.0f}/yr losing yrs {g['lose']}/{g['ny']} -> "
            f"{'PASSES DISCOVERY' if ok else 'dead'}")
        if ok: surv.append(k)
    if not surv: say("HOLDOUT: not opened - nothing passed discovery.")
    else:
        for m in ('mnqx', 'es', 'mes'):
            o2, dy2 = trials(m)
            for k in surv:
                g = grade(o2[k], dy2, *WIN[m]); say(f"   {k} {m:5s}: t {g['t']:+.2f} (zero cost {g['t0']:+.2f}) ${g['usd']:+,.0f}/yr, losing yrs {g['lose']}/{g['ny']}")
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    main()
