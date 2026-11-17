r"""RED2 (RED2_PLAN.txt): 17 rules from the full Reddit read.
    python red2_rules.py            discovery for every trial; holdout / replication for survivors
"""
import os, sys, math, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_paths as P
import red_rules as R

RES = os.path.join(HERE, 'RED2_RESULTS.txt'); LINES = []; TICK = 0.25; NUSD = R.NUSD; COST = R.COST
PAIR_COST = COST + 2 * (2.35 * 1.25 + 0.61) * NUSD / (5 * 6800.0)
NAMES = {'N1': 'lower-high streak, buy stop at the last high', 'N2': 'large opening gap (> 0.5%) continuation', 'N3': '30-minute single print continuation',
         'N4': 'Sunday reopen against Friday', 'N5': 'ETH continuation after a strong close (18:00 -> 09:30)', 'N6': 'crude down before the open -> long 09:30-09:45',
         'N7': 'ADX(14) cross above 25, 1-hour, long', 'N9': 'NQ / ES spread reversion (z 2)', 'N10': 'SMT divergence NQ vs ES, 15-minute',
         'N11': 'first extreme in by 11:00', 'N12': 'confirmed overnight move continues 09:35-11:00', 'N13': 'unconfirmed overnight move reverses 09:35-11:00',
         'N14': 'VIX term structure back to normal -> long next session', 'N15': '2-minute pullback-bar breakout 09:30-10:10', 'N16': 'Heikin-Ashi 2HA, 5-minute',
         'N17': '20-bar breakout + SMA 20/100, 15-minute'}


def say(s=''):
    print(s, flush=True); LINES.append(s)


class Mk:
    def __init__(self, name):
        d = R.load(name); self.__dict__.update(d); self.name = name; self.n = len(self.t)
        self.ro = self.o - self.sh; self.pos = {int(v): i for i, v in enumerate(self.t)}
        rth = (self.md >= 570) & (self.md < 960); idx = np.flatnonzero(rth); dd = self.tday[idx]
        st = idx[np.r_[0, np.flatnonzero(dd[1:] != dd[:-1]) + 1]]; en = idx[np.r_[np.flatnonzero(dd[1:] != dd[:-1]), len(idx) - 1]]
        self.days = self.tday[st]; self.r0 = dict(zip(self.days, st)); self.r1 = dict(zip(self.days, en))
        ok = (self.md[st] == 570) & (self.md[en] == 959); self.full = set(self.days[ok])
        self.ts = np.r_[0, np.flatnonzero(self.tday[1:] != self.tday[:-1]) + 1]; self.first = dict(zip(self.tday[self.ts], self.ts))

    def at(self, day, hhmm):
        cal = day if hhmm >= 18 * 60 or False else day
        return self.pos.get(int(day * 1440 + hhmm))

    def usd(self, i_entry, side, entry, exit_):
        return side * (exit_ - entry) / (entry - self.sh[i_entry]) * NUSD

    def path_exit(self, i0, i1, side, stop, tgt=None):
        for j in range(i0, i1 + 1):
            if stop is not None and ((side > 0 and self.l[j] <= stop) or (side < 0 and self.h[j] >= stop)):
                return (min(self.o[j], stop) if side > 0 else max(self.o[j], stop)) if j > i0 else stop, j
            if tgt is not None and ((side > 0 and self.h[j] >= tgt) or (side < 0 and self.l[j] <= tgt)):
                return (max(self.o[j], tgt) if side > 0 else min(self.o[j], tgt)) if j > i0 else tgt, j
        return self.c[i1], i1


def daily(m):
    rows = []
    for d in sorted(m.full):
        a, b = m.r0[d], m.r1[d]; rows.append((d, m.o[a], m.h[a:b + 1].max(), m.l[a:b + 1].min(), m.c[b]))
    return pd.DataFrame(rows, columns=['d', 'o', 'h', 'l', 'c']).set_index('d')


def n1(m, D):
    out = []; H = D.h.values; days = D.index.values; streak = 0; stop = None
    for k in range(1, len(days)):
        d = days[k]
        if stop is not None:
            a, b = m.r0[d], m.r1[d]
            if m.o[a] >= stop: e, ie = m.o[a], a
            else:
                hit = np.flatnonzero(m.h[a:b + 1] >= stop); e, ie = (max(m.o[a + hit[0]], stop), a + hit[0]) if len(hit) else (None, None)
            if e is not None: out.append((d, m.usd(ie, 1, e, m.c[b])))
        streak = streak + 1 if H[k] < H[k - 1] else 0
        stop = H[k] if streak >= 4 else None
    return out


def n2(m, D):
    out = []; prevc = D.c.shift(1)
    for d in D.index[1:]:
        a, b = m.r0[d], m.r1[d]; pc = prevc[d]; g = m.o[a] / pc - 1
        if abs(g) <= 0.005 or a + 1 > b: continue
        s = 1 if g > 0 else -1; e = m.o[a + 1]; px, _ = m.path_exit(a + 1, b, s, pc); out.append((d, m.usd(a + 1, s, e, px)))
    return out


def n3(m, D):
    out = []
    for d in D.index:
        a, b = m.r0[d], m.r1[d]; hi = [m.h[a + 30 * p:a + 30 * p + 30].max() for p in range(13)]; lo = [m.l[a + 30 * p:a + 30 * p + 30].min() for p in range(13)]
        for p in range(1, 11):
            H, Lw = max(hi[:p]), min(lo[:p])
            if hi[p] > H and lo[p + 1] > H: s, st = 1, H
            elif lo[p] < Lw and hi[p + 1] < Lw: s, st = -1, Lw
            else: continue
            i = a + 30 * (p + 2); px, _ = m.path_exit(i, b, s, st); out.append((d, m.usd(i, s, m.o[i], px))); break
    return out


def n4n5(m, D, which):
    out = []; days = list(D.index)
    for k in range(len(days) - 1):
        d, nx = days[k], days[k + 1]; a, b = m.r0[d], m.r1[d]; o, c, h, l = D.loc[d, ['o', 'c', 'h', 'l']]
        ro = m.first.get(nx); x = m.r0.get(nx)
        if ro is None or x is None or m.md[ro] != 1080: continue
        if which == 'N4':
            if pd.Timestamp(d * 86400, unit='s').weekday() != 4: continue
            fc = m.pos.get(int(d * 1440 + 16 * 60 + 59)); fc = m.c[fc] if fc is not None else c
            s = 1 if (c > o and m.o[ro] < fc) else (-1 if (c < o and m.o[ro] > fc) else 0)
        else:
            loc = (c - l) / (h - l) if h > l else 0.5; s = 1 if (c > o and loc >= 0.8) else (-1 if (c < o and loc <= 0.2) else 0)
        if s: out.append((nx, m.usd(ro, s, m.o[ro], m.o[x])))
    return out


def n6():
    cl = Mk('mcl'); mq = Mk('mnqx'); out = []
    for d in sorted(mq.full):
        ro = cl.first.get(d); e = cl.pos.get(int(d * 1440 + 569))
        if ro is None or e is None or cl.md[ro] != 1080: continue
        if cl.c[e] < cl.o[ro]:
            a = mq.r0[d]; x = mq.pos.get(int(d * 1440 + 585))
            if x is not None: out.append((d, mq.usd(a, 1, mq.o[a], mq.o[x])))
    return out, mq


def n7(m):
    b = R.bars({k: getattr(m, k) for k in ('t', 'o', 'h', 'l', 'c', 'v', 'md', 'tday', 'sh')}, 60); n = len(b['c'])
    up = b['h'] - np.r_[np.nan, b['h'][:-1]]; dn = np.r_[np.nan, b['l'][:-1]] - b['l']
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0); atr = R.wilder(b, 14)
    sm = lambda x: pd.Series(x).ewm(alpha=1 / 14, adjust=False).mean().values
    pdi = 100 * sm(pdm) / atr; mdi = 100 * sm(mdm) / atr; dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9); adx = sm(dx)
    out = []; i = 15
    while i < n - 1:
        if adx[i - 1] <= 25 < adx[i] and b['md'][i + 1] != 16 * 60 and b['md'][i + 1] != 17 * 60:
            j0 = i + 1; e = b['o'][j0]; st = e - 1.5 * atr[i]; tg = e + 3.5 * 1.5 * atr[i]; day = b['tday'][j0]; px = None; j = j0
            while j < n and b['tday'][j] == day and b['md'][j] != 17 * 60:
                if b['l'][j] <= st: px = min(b['o'][j], st) if j > j0 else st; break
                if b['h'][j] >= tg: px = max(b['o'][j], tg) if j > j0 else tg; break
                if b['md'][j] == 16 * 60: px = b['c'][j]; break
                j += 1
            if px is None: px = b['c'][min(j, n - 1) - 1]
            out.append((day, (px - e) / (e - b['sh'][j0]) * NUSD)); i = j + 1; continue
        i += 1
    return out


def n9(nq, es):
    t = np.intersect1d(nq.t, es.t); ia = np.searchsorted(nq.t, t); ib = np.searchsorted(es.t, t)
    s = np.log(nq.c[ia]) - np.log(es.c[ib]); ss = pd.Series(s); z = ((ss - ss.rolling(390).mean()) / ss.rolling(390).std()).values
    md = t % 1440; day = (t + 360) // 1440; out = []; k = 390; n = len(t)
    while k < n - 1:
        if 600 <= md[k] <= 930 and abs(z[k]) >= 2 and day[k + 1] == day[k]:
            side = -1 if z[k] > 0 else 1; j0 = k + 1; e1, e2 = nq.o[ia[j0]], es.o[ib[j0]]; j = j0
            while j < n - 1 and day[j] == day[j0] and md[j] < 959 and np.sign(z[j]) == np.sign(z[k]): j += 1
            x1, x2 = (nq.o[ia[j]], es.o[ib[j]]) if md[j] < 959 and day[j] == day[j0] else (nq.c[ia[j]], es.c[ib[j]])
            r = side * ((x1 / e1 - 1) - (x2 / e2 - 1)); out.append((day[j0], r * NUSD + COST - PAIR_COST)); k = j + 1; continue
        k += 1
    return out


def n10(nq, es):
    key = lambda m: m.t // 15
    bn = R.bars({k: getattr(nq, k) for k in ('t', 'o', 'h', 'l', 'c', 'v', 'md', 'tday', 'sh')}, 15); be = R.bars({k: getattr(es, k) for k in ('t', 'o', 'h', 'l', 'c', 'v', 'md', 'tday', 'sh')}, 15)
    kn = np.unique(nq.t // 15); ke = np.unique(es.t // 15); common = np.intersect1d(kn, ke); ia = np.searchsorted(kn, common); ib = np.searchsorted(ke, common)
    lnq = pd.Series(bn['l'][ia]); hnq = pd.Series(bn['h'][ia]); les = pd.Series(be['l'][ib]); hes = pd.Series(be['h'][ib])
    pl_n = lnq.shift(1).rolling(8).min().values; ph_n = hnq.shift(1).rolling(8).max().values; pl_e = les.shift(1).rolling(8).min().values; ph_e = hes.shift(1).rolling(8).max().values
    sig_c = np.where((lnq.values < pl_n) & (les.values >= pl_e), 1, np.where((hnq.values > ph_n) & (hes.values <= ph_e), -1, 0))
    sig = np.zeros(len(bn['c']), int); sig[ia] = sig_c; rth = (bn['md'] >= 570) & (bn['md'] <= 930); sig = np.where(rth, sig, 0)
    stop = lambda k, s, e: (bn['l'][k] - TICK) if s > 0 else (bn['h'][k] + TICK)
    return R.run_trades(bn, sig, stop, lambda k, s, e, st: e + s * 2 * abs(e - st))


def n11(m, D):
    out = []
    for d in D.index:
        a, b = m.r0[d], m.r1[d]; x = a + 90
        if x > b: continue
        lo_i = a + int(np.argmin(m.l[a:x])); hi_i = a + int(np.argmax(m.h[a:x]))
        s, st = (1, m.l[a:x].min() - TICK) if lo_i < hi_i else (-1, m.h[a:x].max() + TICK)
        px, _ = m.path_exit(x, b, s, st); out.append((d, m.usd(x, s, m.o[x], px)))
    return out


def n12n13(m, D):
    A, B = [], []; days = list(D.index); vols = {}
    for d in days: a = m.r0[d]; vols[d] = m.v[a:a + 5].sum()
    vs = pd.Series(vols); avg = vs.rolling(20).mean().shift(1)
    for k in range(1, len(days)):
        d, pd_ = days[k], days[k - 1]; a = m.r0[d]; pre = m.pos.get(int(d * 1440 + 569)); x = m.pos.get(int(d * 1440 + 660))
        if pre is None or x is None or not np.isfinite(avg.get(d, np.nan)): continue
        ov = np.sign(m.c[pre] - m.c[m.r1[pd_]]); f = np.sign(m.c[a + 4] - m.o[a])
        if ov == 0 or f == 0: continue
        if f == ov and vols[d] > avg[d]: A.append((d, m.usd(a + 5, ov, m.o[a + 5], m.o[x])))
        if f == -ov: B.append((d, m.usd(a + 5, -ov, m.o[a + 5], m.o[x])))
    return A, B


def n14(m, D):
    def rd(f):
        x = pd.read_csv(os.path.join(P.WEB, f)); x.columns = [c.strip().upper() for c in x.columns]; dc = [c for c in x.columns if 'DATE' in c][0]
        return pd.Series(x['CLOSE'].values, index=pd.to_datetime(x[dc]).values.astype('datetime64[D]').astype(np.int64))
    v9 = rd('VIX9D_History.csv'); v = rd('VIX_History.csv'); j = pd.concat([v9, v], axis=1, keys=['a', 'b']).dropna()
    inv = j.a > j.b; flip = (~inv) & inv.shift(1, fill_value=False); out = []; days = np.array(sorted(m.full))
    for dd in j.index[flip.values]:
        k = np.searchsorted(days, dd, side='right')
        if k < len(days): d = days[k]; a, b = m.r0[d], m.r1[d]; out.append((d, m.usd(a, 1, m.o[a], m.c[b])))
    return out


def n15(m):
    b = R.bars({k: getattr(m, k) for k in ('t', 'o', 'h', 'l', 'c', 'v', 'md', 'tday', 'sh')}, 2); n = len(b['c']); e9 = pd.Series(b['c']).ewm(span=9, adjust=False).mean().values
    lr = R.last_rth(b); out = []; k = 1
    while k < n - 2:
        if 570 <= b['md'][k] <= 608 and b['tday'][k + 2] == b['tday'][k] and b['tday'][k] in lr:
            s = 0
            if b['c'][k] > b['c'][k - 1] and b['c'][k] > e9[k] and b['c'][k + 1] < b['o'][k + 1]: s = 1
            elif b['c'][k] < b['c'][k - 1] and b['c'][k] < e9[k] and b['c'][k + 1] > b['o'][k + 1]: s = -1
            if s:
                j0 = k + 2; trig = b['h'][k + 1] + TICK if s > 0 else b['l'][k + 1] - TICK
                if (s > 0 and b['h'][j0] >= trig) or (s < 0 and b['l'][j0] <= trig):
                    e = max(b['o'][j0], trig) if s > 0 else min(b['o'][j0], trig); st = b['l'][k + 1] - TICK if s > 0 else b['h'][k + 1] + TICK
                    r = abs(e - st); tg = e + s * 2 * r; end = lr[b['tday'][k]]; px = None; j = j0; be = False
                    while j <= end:
                        if (s > 0 and b['l'][j] <= st) or (s < 0 and b['h'][j] >= st): px = st if j == j0 else (min(b['o'][j], st) if s > 0 else max(b['o'][j], st)); break
                        if j > j0 and ((s > 0 and b['h'][j] >= tg) or (s < 0 and b['l'][j] <= tg)): px = max(b['o'][j], tg) if s > 0 else min(b['o'][j], tg); break
                        if not be and s * ((b['h'][j] if s > 0 else b['l'][j]) - e) >= 0.85 * r: st = e; be = True
                        j += 1
                    if px is None: j = end; px = b['c'][end]
                    out.append((b['tday'][j0], s * (px - e) / (e - b['sh'][j0]) * NUSD)); k = j + 1; continue
        k += 1
    return out


def n16(m):
    b = R.bars({k: getattr(m, k) for k in ('t', 'o', 'h', 'l', 'c', 'v', 'md', 'tday', 'sh')}, 5); n = len(b['c'])
    hc = (b['o'] + b['h'] + b['l'] + b['c']) / 4; ho = np.zeros(n); ho[0] = (b['o'][0] + b['c'][0]) / 2
    for i in range(1, n): ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    hh = np.maximum.reduce([b['h'], ho, hc]); hl = np.minimum.reduce([b['l'], ho, hc]); g = hc > ho; r = hc < ho
    nolow = np.isclose(hl, ho); nohigh = np.isclose(hh, ho); ng = b['c'] > b['o']; nr = b['c'] < b['o']
    rth = (b['md'] >= 575) & (b['md'] <= 950)
    sl = np.r_[False, g[1:] & g[:-1] & nolow[1:] & nolow[:-1] & ng[1:] & ng[:-1]] & rth
    ss = np.r_[False, r[1:] & r[:-1] & nohigh[1:] & nohigh[:-1] & nr[1:] & nr[:-1]] & rth
    sig = np.where(sl, 1, np.where(ss, -1, 0))
    ex = lambda j, s: (s > 0 and r[j]) or (s < 0 and g[j])
    return R.run_trades(b, sig, lambda k, s, e: None, lambda k, s, e, st: None, exit_sig=ex)


def n17(m):
    b = R.bars({k: getattr(m, k) for k in ('t', 'o', 'h', 'l', 'c', 'v', 'md', 'tday', 'sh')}, 15); n = len(b['c']); c = pd.Series(b['c'])
    s20 = c.rolling(20).mean().values; s100 = c.rolling(100).mean().values; hh = pd.Series(b['h']).shift(1).rolling(20).max().values; ll = pd.Series(b['l']).shift(1).rolling(20).min().values
    atr = R.wilder(b, 14); out = []; i = 101
    while i < n - 1:
        s = 1 if (b['c'][i] > hh[i] and b['c'][i] > s20[i] and s20[i] > s100[i]) else (-1 if (b['c'][i] < ll[i] and b['c'][i] < s20[i] and s20[i] < s100[i]) else 0)
        j0 = i + 1
        if s and b['tday'][j0] == b['tday'][i] and not (16 * 60 + 45 <= b['md'][j0] or b['md'][j0] < 0):
            if 17 * 60 <= b['md'][j0] < 18 * 60: i += 1; continue
            e = b['o'][j0]; st = b['c'][i] - s * 2 * atr[i]; day = b['tday'][j0]; j = j0; px = None
            while j < n and b['tday'][j] == day:
                if (s > 0 and b['l'][j] <= st) or (s < 0 and b['h'][j] >= st): px = st if j == j0 else (min(b['o'][j], st) if s > 0 else max(b['o'][j], st)); break
                if b['md'][j] >= 16 * 60 + 45: px = b['c'][j]; break
                if (s > 0 and s20[j] < s100[j]) or (s < 0 and s20[j] > s100[j]):
                    px = b['o'][j + 1] if j + 1 < n else b['c'][j]; break
                j += 1
            if px is None: px = b['c'][j - 1]
            out.append((day, s * (px - e) / (e - b['sh'][j0]) * NUSD)); i = j + 1; continue
        i += 1
    return out


def grade(trades, days, a, b, cost=COST):
    a = int(pd.Timestamp(a).value // 86_400_000_000_000); b = int(pd.Timestamp(b).value // 86_400_000_000_000); dd = np.array(sorted(x for x in days if a <= x <= b))
    ix = {v: i for i, v in enumerate(dd)}; x = np.zeros(len(dd)); x0 = np.zeros(len(dd)); n = 0; w = []
    for dy, u in trades:
        if dy in ix: x[ix[dy]] += u - cost; x0[ix[dy]] += u; n += 1; w.append(u - cost)
    t = lambda y: float(y.mean() / y.std() * math.sqrt(len(y))) if y.std() > 0 else 0.0
    yy = pd.Series(x).groupby(pd.to_datetime(dd * 86400, unit='s').year).sum()
    return dict(n=n, yr=n / (len(dd) / 252), t=t(x), t0=t(x0), usd=x.mean() * 252, lose=int((yy < 0).sum()), ny=len(yy), win=float(np.mean(np.array(w) > 0)) if w else np.nan)


def run_all(name, partner=None):
    m = Mk(name); D = daily(m); out = {}
    out['N1'] = n1(m, D); out['N2'] = n2(m, D); out['N3'] = n3(m, D); out['N4'] = n4n5(m, D, 'N4'); out['N5'] = n4n5(m, D, 'N5'); out['N7'] = n7(m)
    if partner: es = Mk(partner); out['N9'] = n9(m, es); out['N10'] = n10(m, es)
    out['N11'] = n11(m, D); out['N12'], out['N13'] = n12n13(m, D); out['N14'] = n14(m, D); out['N15'] = n15(m); out['N16'] = n16(m); out['N17'] = n17(m)
    return out, set(m.full)


def main():
    t0 = time.time(); say(); say("=" * 165)
    say(f"RED2  RULES FROM THE FULL REDDIT READ  red2_rules.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (RED2_PLAN.txt) $ at MNQ 27,875, live cost ${COST:.2f} (pairs ${PAIR_COST:.2f}) a round trip"); say("=" * 165)
    out, days = run_all('nq', 'es'); surv = []
    for k in [x for x in NAMES if x != 'N6']:
        g = grade(out[k], days, *R.WIN['nq'], cost=PAIR_COST if k == 'N9' else COST); ok = g['t'] >= 3.0 and g['lose'] <= g['ny'] / 2
        say(f"   {k:3s} {NAMES[k]:55s} | NQ 2010-19: {g['n']:6,d} trades ({g['yr']:5.0f}/yr) winners {g['win']:.0%} t {g['t']:+6.2f} (zero cost {g['t0']:+6.2f}) ${g['usd']:+8,.0f}/yr "
            f"losing yrs {g['lose']}/{g['ny']} -> {'PASSES DISCOVERY' if ok else 'dead'}")
        if ok: surv.append(k)
    o6, mq = n6(); g = grade(o6, set(mq.full), '2021-07-12', '2023-12-31'); ok6 = g['t'] >= 3.0 and g['lose'] <= g['ny'] / 2
    say(f"   N6  {NAMES['N6']:55s} | MNQ 2021-23: {g['n']:5,d} trades ({g['yr']:5.0f}/yr) winners {g['win']:.0%} t {g['t']:+6.2f} (zero cost {g['t0']:+6.2f}) ${g['usd']:+8,.0f}/yr "
        f"losing yrs {g['lose']}/{g['ny']} -> {'PASSES DISCOVERY' if ok6 else 'dead'}")
    if ok6:
        g = grade(o6, set(mq.full), '2024-01-01', '2026-10-15'); say(f"       N6 holdout 2024-26: t {g['t']:+.2f} ${g['usd']:+,.0f}/yr")
    if not surv: say("HOLDOUT: not opened for N1-N17 - nothing passed discovery.")
    else:
        for mkt, partner in (('mnqx', 'mes'), ('es', None)):
            o2, d2 = run_all(mkt, partner)
            for k in surv:
                if k in o2:
                    g = grade(o2[k], d2, *R.WIN[mkt], cost=PAIR_COST if k == 'N9' else COST); say(f"   {k} {mkt}: t {g['t']:+.2f} (zero cost {g['t0']:+.2f}) ${g['usd']:+,.0f}/yr, losing yrs {g['lose']}/{g['ny']}")
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    main()
