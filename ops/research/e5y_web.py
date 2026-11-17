r"""E5Y (E5Y_PLAN.txt): day-level and opening rules from the 2026-11-15 web scout, on NQ / MNQ with ES / MES replication.
    python e5y_web.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import say
X = L.X; U = L.U; WEB = X.PATHS.WEB; COST = 2 * X.SIDE(2.35) * U
MEGA = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX']


def sessions(L2, name):
    M = L2.Mins(name); rows = []
    for d in M.dates:
        i0, i1 = M.days[d]; j = i0 + int(np.searchsorted(M.m[i0:i1 + 1], 600)); j2 = i0 + int(np.searchsorted(M.m[i0:i1 + 1], 720))
        rows.append((d, M.o[i0], M.c[i1], M.o[j] if j <= i1 and M.m[j] == 600 else np.nan, M.ao[i0], M.ac[i1], M.o[j2] if j2 <= i1 and M.m[j2] == 720 else np.nan))
    S = pd.DataFrame(rows, columns=['date', 'o', 'c', 'p10', 'ao', 'ac', 'p12']).set_index('date'); S['noon'] = S.p12 / S.o - 1
    S['day'] = S.c / S.o - 1; S['first'] = S.p10 / S.o - 1; S['night'] = (S.ao - S.ac.shift(1)) / S.c.shift(1); S['cc'] = (S.ac - S.ac.shift(1)) / S.c.shift(1)
    return S


def signals(idx, cc):
    out = pd.DataFrame(index=idx)
    v = pd.read_csv(os.path.join(WEB, 'VIX_History.csv')); v['d'] = pd.to_datetime(v['DATE']); v = v.set_index('d')['CLOSE']; dv = v.diff()
    prev = dv.reindex(idx.union(dv.index)).shift(1).reindex(idx)
    out['Y1'] = np.where(prev >= 2.5, 1.0, 0.0); out['Y2'] = np.where(prev >= 3.0, 1.0, 0.0)
    y = pd.read_csv(os.path.join(WEB, 'DGS10.csv')); y.columns = ['d', 'y']; y['d'] = pd.to_datetime(y.d); y['y'] = pd.to_numeric(y.y, errors='coerce'); y = y.set_index('d').y.ffill()
    yy = y.reindex(idx.union(y.index)).ffill().reindex(idx); rb = (-8.5 * yy.diff() / 100 + yy.shift(1) / 100 / 252).fillna(0.0); rs = cc.fillna(0.0)
    month = idx.to_period('M'); ws = np.zeros(len(idx)); wt = np.zeros(len(idx)); a, b = 0.6, 0.4; a2, b2 = 0.6, 0.4
    for i in range(len(idx)):
        if i and month[i] != month[i - 1]: a, b = 0.6, 0.4
        a *= 1 + rs.iloc[i]; b *= 1 + rb.iloc[i]; ws[i] = a / (a + b) - 0.6
        a2 *= 1 + rs.iloc[i]; b2 *= 1 + rb.iloc[i]; dev = a2 / (a2 + b2) - 0.6; wt[i] = dev
        if abs(dev) > 0.015: a2, b2 = 0.6, 0.4
    ws = pd.Series(ws, index=idx).shift(1); wt = pd.Series(wt, index=idx).shift(1)
    left = pd.Series(1, index=idx).groupby(month).transform(lambda s: np.arange(len(s))[::-1]).values
    out['Y3'] = np.where(left < 5, -np.sign(ws.fillna(0.0)), 0.0); out['Y4'] = np.where(wt > 0.0075, -1.0, np.where(wt < -0.0075, 1.0, 0.0))
    f = os.path.join(WEB, 'earn_calendar.csv'); out['Y5'] = np.nan; out['Y6'] = np.nan
    if os.path.exists(f) and os.path.exists(os.path.join(WEB, 'earn_calendar_done.txt')):
        e = pd.read_csv(f); e['symbol'] = e.symbol.replace({'GOOG': 'GOOGL', 'FB': 'META'}); e = e[e.symbol.isin(MEGA)].drop_duplicates(['date', 'symbol'])
        n = e.groupby('date').symbol.nunique(); n.index = pd.to_datetime(n.index); n = n.reindex(idx).fillna(0)
        out['Y5'] = np.where(n >= 2, 1.0, 0.0); out['Y6'] = np.where(n >= 1, 1.0, 0.0)
    out['Y9'] = np.where(idx.dayofweek == 0, 1.0, 0.0)
    return out


def card(pos, ret, sel, cost=COST):
    p = pos[sel]; r = ret[sel]; x = np.where(p != 0, p * r * U - cost, 0.0); x = np.nan_to_num(x); n = (p != 0).sum(); yrs = sel.sum() / 252
    s = pd.Series(x, index=ret.index[sel]); yr = s.groupby(s.index.year).sum(); tt = lambda z: z.mean() / z.std() * np.sqrt(len(z)) if z.std() > 0 else 0.0
    g = np.nan_to_num(np.where(p != 0, p * r * U, 0.0)); eq = np.cumsum(x)
    return dict(n=n / yrs, bp=float(np.nanmean((p * r)[p != 0]) * 1e4) if n else 0.0, usd=x.mean() * 252, t=tt(x), t0=tt(g), neg=int((yr < 0).sum()), ny=len(yr), t5=tt(s.drop(s.nlargest(5).index).values),
                dd=float((np.maximum.accumulate(eq) - eq).max()), win=float((x[p != 0] > 0).mean()) if n else 0.0)


def main():
    t0 = time.time(); sys.path.insert(0, L.TR.RESEARCH)
    import src2_lib as L2
    D = {k: sessions(L2, k) for k in ('nq', 'mnqx', 'es', 'mes')}
    DESC = {'Y1': 'long the session after a VIX jump of 2.5+', 'Y2': 'long the session after a VIX jump of 3.0+', 'Y3': 'rebalancing, last 5 days of the month', 'Y4': 'rebalancing, drift over 0.75 points',
            'Y5': 'long when 2+ megacaps report after the close', 'Y6': 'long when 1+ megacap reports after the close', 'Y7': 'against the overnight move, 09:30-10:00',
            'Y8': 'against an overnight move over 0.5%, 09:30-10:00', 'Y9': 'long every Monday', 'Y14': 'same-weekday momentum (11 months, skip 1)'}
    say(); say("=" * 160)
    say(f"WEB SCOUT BATCH 1  e5y_web.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5Y_PLAN.txt) rules from outside information, all inside 09:30-16:00; 1 micro, $ at today's price, live cost; discovery NQ 2010-19, holdout MNQ 2019-26")
    say("=" * 160)
    P = {}
    for k, S in D.items():
        sg = signals(S.index, S.cc); pos = {c: sg[c].values for c in sg.columns}
        pos['Y7'] = -np.sign(S.night.fillna(0.0).values); pos['Y8'] = np.where(S.night.abs().values > 0.005, pos['Y7'], 0.0)
        wd = S.index.dayofweek.values; y14 = np.zeros(len(S)); dv = S.day.values; ix = S.index
        for i in range(len(S)):
            m = (wd[:i] == wd[i]) & (ix[:i] >= ix[i] - pd.DateOffset(months=12)) & (ix[:i] < ix[i] - pd.DateOffset(months=1))
            if m.sum() >= 20: y14[i] = np.sign(np.nanmean(dv[:i][m]))
        pos['Y14'] = y14
        tf = (ix.dayofweek == 4) & (ix.day >= 15) & (ix.day <= 21); pos['Y12'] = np.where(tf & (S.night.values > 0), -1.0, 0.0); pos['Y13'] = np.where(tf & np.isin(ix.month, (3, 6, 9, 12)), -1.0, 0.0)
        P[k] = (S, pos)
    S, pos = P['nq']; sel = np.ones(len(S), bool)
    say(f"   {'rule':52s} | trades/yr | wins | bp a trade | $/yr live | t live | t zero cost | losing yrs | drop-5 t | max DD | overnight share (bp, same signal)")
    surv = []
    for c in DESC:
        if np.isnan(pos[c]).all(): say(f"   {c:3s} {DESC[c]:48s} | waiting for the earnings calendar download"); continue
        ret = S['first'] if c in ('Y7', 'Y8') else S['day']; k = card(pos[c], ret, sel)
        ov = float(np.nanmean((pos[c] * S.night.values)[pos[c] != 0]) * 1e4) if c not in ('Y7', 'Y8') and (pos[c] != 0).any() else float('nan')
        ok = k['t'] >= 2.8 and k['neg'] <= 1 and k['t5'] >= 2.0
        say(f"   {c:3s} {DESC[c]:48s} | {k['n']:9.0f} | {k['win'] * 100:3.0f}% | {k['bp']:+10.1f} | ${k['usd']:+8,.0f} | {k['t']:+6.2f} | {k['t0']:+11.2f} | {k['neg']:3d} of {k['ny']:2d} | {k['t5']:+8.2f} | ${k['dd']:6,.0f} | {ov:+.1f}  {'PASS' if ok else 'dead'}")
        if ok: surv.append(c)
    say(); say("   holdout MNQ 2019-26 and S&P replication: " + ("for " + ", ".join(surv) if surv else "not opened - nothing passed discovery"))
    for c in surv:
        for k_, lab in (('mnqx', 'MNQ 2019-26'), ('es', 'ES 2010-19'), ('mes', 'MES 2019-26')):
            S2, p2 = P[k_]; ret = S2['first'] if c in ('Y7', 'Y8') else S2['day']; k = card(p2[c], ret, np.ones(len(S2), bool))
            say(f"      {c} {lab}: {k['n']:.0f} trades/yr, {k['bp']:+.1f} bp, ${k['usd']:+,.0f}/yr, t {k['t']:+.2f}, losing years {k['neg']} of {k['ny']}")
    say(); say('   EXPIRATION FRIDAY short 09:30-12:00, pooled NQ 2010-19 + MNQ 2019-26, every third Friday incl. the roll days rebuilt from back-adjusted minutes (needs t >= 2.8, drop-5 t >= 2.0, both periods positive, S&P same sign):')
    EX = {}
    for k_ in ('nq', 'mnqx', 'es', 'mes'):
        M = L2.Mins(k_); rth = (M.m >= 570) & (M.m < 960); dts = pd.DatetimeIndex(M.date); rows = []
        for d in pd.DatetimeIndex(np.unique(M.date[rth])):
            if not (d.dayofweek == 4 and 15 <= d.day <= 21): continue
            ix = np.flatnonzero(rth & (M.date == np.datetime64(d, 'D')))
            if len(ix) < 150 or M.m[ix[0]] != 570: continue
            q = np.searchsorted(M.m[ix], 720)
            if q >= len(ix): continue
            pv = np.flatnonzero(rth[:ix[0]])
            if not len(pv): continue
            rows.append((d, (M.ao[ix[q]] - M.ao[ix[0]]) / M.o[ix[0]], (M.ao[ix[0]] - M.ac[pv[-1]]) / M.c[pv[-1]]))
        EX[k_] = pd.DataFrame(rows, columns=['date', 'noon', 'night']).set_index('date')
    for c, lab in (('Y12', 'only after a positive overnight move'), ('Y13', 'quarterly expirations only'), ('base', 'for reference, not a trial: every expiration Friday')):
        parts = []
        for k_ in ('nq', 'mnqx', 'es', 'mes'):
            E_ = EX[k_]; on = (E_.night > 0) if c == 'Y12' else E_.index.month.isin((3, 6, 9, 12)) if c == 'Y13' else np.ones(len(E_), bool)
            ev_ = pd.Series(np.where(on, -E_.noon.values * U - COST, 0.0), index=E_.index); allx = pd.Series(0.0, index=P[k_][0].index.union(E_.index)); allx[ev_.index] = ev_.values; parts.append(allx)
        x = pd.concat(parts[:2]); xs = pd.concat(parts[2:]); tt = lambda z: z.mean() / z.std() * np.sqrt(len(z)) if z.std() > 0 else 0.0; ev = x[x != 0]
        ok = c != 'base' and tt(x.values) >= 2.8 and tt(x.drop(x.nlargest(5).index).values) >= 2.0 and parts[0].sum() > 0 and parts[1].sum() > 0 and xs.sum() > 0
        say(f"      {c} {lab}: {len(ev)} events, average ${ev.mean():+,.0f} an event, wins {np.mean(ev > 0) * 100:.0f}%, ${x.mean() * 252:+,.0f}/yr, t {tt(x.values):+.2f}, without the 5 best days t {tt(x.drop(x.nlargest(5).index).values):+.2f}; "
            f"NQ 2010-19 ${parts[0].sum():+,.0f}, MNQ 2019-26 ${parts[1].sum():+,.0f}; S&P (ES + MES) t {tt(xs.values):+.2f}  {'PASS' if ok else '' if c == 'base' else 'dead'}")
        if ok: surv.append(c)
    say(f"VERDICT (batch 1, day-level rules): {', '.join(surv) if surv else 'none passes discovery'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
