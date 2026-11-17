import os, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import e5n_calendar as CAL

LOOK = 14; GATE = 4; CH = 3.0; NS = 78; A20 = 2 / 21
CHECKS = frozenset(range(600, 931, 30)); LUNCH = frozenset((720, 750, 780))


def five(m1):
    end = pd.DatetimeIndex(m1.end).values.astype('datetime64[m]').astype('int64'); b = ((end + 4) // 5) * 5
    g = pd.DataFrame({'b': b, 'o': m1.o.values, 'h': m1.h.values, 'l': m1.l.values, 'c': m1.c.values, 'v': m1.v.values}).groupby('b', sort=True)
    out = g.agg(open=('o', 'first'), high=('h', 'max'), low=('l', 'min'), close=('c', 'last'), volume=('v', 'sum')).reset_index()
    out['time'] = out.b.values.astype('datetime64[m]').astype('datetime64[ns]')
    return out[['time', 'open', 'high', 'low', 'close', 'volume']]


def is_short(d):
    d = pd.Timestamp(d).date(); return d in CAL.short_days(d.year)


def run(df5, live=False, warmup=LOOK, bars=None, gate=None, force=None, extra=None, pyramid=None, look=LOOK, checks=CHECKS, lunch=LUNCH):
    df = df5.sort_values('time').reset_index(drop=True); t = pd.DatetimeIndex(df.time); n = len(df)
    etmin = (t.hour * 60 + t.minute).values.astype(int)
    o, h, l, c, v = (df[x].values.astype(float) for x in ('open', 'high', 'low', 'close', 'volume'))
    tr = np.empty(n); tr[0] = h[0] - l[0]; tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = np.empty(n); atr[0] = tr[0]
    for i in range(1, n): atr[i] = (13 * atr[i - 1] + tr[i]) / 14 if i >= 14 else tr[:i + 1].mean()
    s_ = pd.Series(c); q = np.sign(s_.ewm(span=50, adjust=False).mean() - s_.ewm(span=200, adjust=False).mean()).to_numpy()
    rth = (etmin > 570) & (etmin <= 960); idx = np.flatnonzero(rth); dd = t.normalize().values[idx]
    brk = np.r_[0, np.flatnonzero(dd[1:] != dd[:-1]) + 1, len(idx)]
    hist = []; pc = np.nan; e20 = None; k = 0; rows = []
    for a, b in zip(brk[:-1], brk[1:]):
        ii = idx[a:b]; M = etmin[ii]; nb = len(ii); date = pd.Timestamp(dd[a])
        if M[0] != 575 or (not live and nb < 30): continue
        O, H, L, C, V, A = o[ii], h[ii], l[ii], c[ii], v[ii], atr[ii]; slot = (M - 575) // 5
        E = np.empty(nb); e = e20
        for i in range(nb):
            e = C[i] if e is None else e + A20 * (C[i] - e); E[i] = e
        trade = k >= max(look, warmup) and pc == pc and not (live and is_short(date))
        if trade:
            blk = np.array(hist[-look:]); cnt = (~np.isnan(blk)).sum(axis=0)
            with np.errstate(invalid='ignore'), np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning); mu = np.nanmean(blk, axis=0)
            mu[cnt < max(1, (look + 1) // 2)] = np.nan; sg = mu[slot]
            vw = np.cumsum((H + L + C) / 3 * V) / np.maximum(np.cumsum(V), 1)
            hi, lo = max(O[0], pc), min(O[0], pc)
            st = dict(s=0.0, ref=0.0, p=0.0, vol=0.0, d=0, e1=0.0, ext=0.0, w=0.0, added=True); viol = 0

            def fill(target, px):
                nd = int(np.sign(target))
                st['p'] += st['s'] * (px - st['ref']); st['vol'] += abs(target - st['s']); st['s'] = target; st['ref'] = px
                if nd != 0 and nd != st['d']: st['e1'] = px; st['ext'] = px
                st['d'] = nd

            flat_i = nb if live else (nb - 1 if M[-1] >= 960 else nb - 2)
            tg = np.zeros(nb)
            for i in range(flat_i):
                if live and M[i] >= 960:
                    if st['s'] != 0: fill(0.0, C[i])
                    break
                nxt = o[ii[i] + 1] if ii[i] + 1 < n else C[i]
                cc = C[i]; d = st['d']
                ub = hi * (1 + sg[i]) if sg[i] == sg[i] else np.nan; lb = lo * (1 - sg[i]) if sg[i] == sg[i] else np.nan
                if d != 0:
                    st['ext'] = max(st['ext'], cc) if d > 0 else min(st['ext'], cc)
                    if (d > 0 and cc < st['ext'] - CH * A[i]) or (d < 0 and cc > st['ext'] + CH * A[i]):
                        fill(0.0, nxt); viol = 0; continue
                    if force is not None and force(date, int(M[i]), d):
                        fill(0.0, nxt); viol = 0; continue
                if d != 0 and sg[i] == sg[i]:
                    bad = (cc < max(ub, vw[i])) if d > 0 else (cc > min(lb, vw[i]))
                    viol = viol + 1 if bad else 0
                    if viol >= GATE: fill(0.0, nxt); viol = 0; continue
                if pyramid is not None and d != 0 and not st['added']:
                    if pyramid[0] == 'profit' and d * (cc - st['e1']) >= pyramid[1] * st['w']:
                        fill(2.0 * d, nxt); st['added'] = True
                    elif pyramid[0] == 'check' and M[i] in checks and sg[i] == sg[i] and d * (cc - st['e1']) > 0 and ((d > 0 and cc > ub) or (d < 0 and cc < lb)):
                        fill(2.0 * d, nxt); st['added'] = True
                regular = M[i] in checks
                if not (sg[i] == sg[i] and (regular or (extra is not None and d == 0 and 600 < M[i] < 930 and not 715 < M[i] < 810 and extra(date, int(M[i]))))):
                    tg[i] = st['s']; continue
                if d == 0:
                    new = 1 if cc > ub else (-1 if cc < lb else 0)
                    if M[i] in lunch: new = 0
                else:
                    new = d
                if new != 0 and new != d:
                    if (new == 1 and not cc > E[i]) or (new == -1 and not cc < E[i]): new = 0
                if new != 0 and new != d and q[ii[i]] != new: new = 0
                if new != 0 and new != d and gate is not None and not gate(date, int(M[i]), new, regular): new = 0
                if new != d:
                    fill(float(new), nxt)
                    if new != 0: st['w'] = O[0] * sg[i]; st['added'] = False
                tg[i] = st['s']
            else:
                if st['s'] != 0:
                    if live: fill(0.0, o[ii[-1] + 1] if ii[-1] + 1 < n else C[-1])
                    else: fill(0.0, C[flat_i])
            rows.append((date, st['p'], st['vol'], O[0]))
            if bars is not None:
                bars.append(pd.DataFrame({'time': t[ii], 'target': tg, 'q': q[ii], 'close': C, 'ub': hi * (1 + sg), 'lb': lo * (1 - sg), 'vw': vw, 'ema': E, 'atr': A}))
        if nb >= 30:
            mv = np.full(NS, np.nan); mv[slot] = np.abs(C / O[0] - 1); hist.append(mv); pc = C[-1]; e20 = e; k += 1
    return pd.DataFrame(rows, columns=['date', 'points', 'sides', 'open930']).set_index('date')
