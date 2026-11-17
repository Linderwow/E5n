import numpy as np, pandas as pd
from types import SimpleNamespace as NS

TICK = 0.25; FVG_MIN = 0.25; LB = {1: 1440, 5: 2880, 15: 4320, 60: 10080, 240: 28800}; SWING_LB = 2880; WIN = 10080


def agg(E, tf):
    if tf == 1: key, o, h, l, c = E.t, E.o, E.h, E.l, E.c
    else:
        k = E.t // tf; st = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
        o = E.o[st]; h = np.maximum.reduceat(E.h, st); l = np.minimum.reduceat(E.l, st); c = E.c[np.r_[st[1:] - 1, len(k) - 1]]; key = k[st]
    pc = np.r_[c[0], c[:-1]]; tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    unit = 1 if tf == 1 else tf
    return NS(key=key if tf == 1 else key, tend=(key + 1) if tf == 1 else (key + 1) * tf, step=1 if tf == 1 else 1, unit=unit, o=o, h=h, l=l, c=c, atr=atr)


def fvgs(B, tf):
    n = len(B.h); k = np.arange(2, n); gap = 2 if tf != 1 else 2
    cons = (B.key[k] - B.key[k - 2]) == gap
    up = cons & (B.l[k] > B.h[k - 2]) & ((B.l[k] - B.h[k - 2]) >= FVG_MIN * B.atr[k])
    dn = cons & (B.h[k] < B.l[k - 2]) & ((B.l[k - 2] - B.h[k]) >= FVG_MIN * B.atr[k])
    ku, kd = k[up], k[dn]
    t = np.r_[B.tend[ku], B.tend[kd]]; typ = np.r_[np.ones(len(ku)), -np.ones(len(kd))]
    lo = np.r_[B.h[ku - 2], B.h[kd]]; hi = np.r_[B.l[ku], B.l[kd - 2]]
    s = np.argsort(t, kind='stable')
    return NS(t=t[s], typ=typ[s], lo=lo[s], hi=hi[s])


def swings(B, n=5):
    h, l = pd.Series(B.h), pd.Series(B.l)
    lh = h.shift(1).rolling(n).max().values; rh = pd.Series(B.h[::-1]).shift(1).rolling(n).max().values[::-1]
    ll = l.shift(1).rolling(n).min().values; rl = pd.Series(B.l[::-1]).shift(1).rolling(n).min().values[::-1]
    ih = np.flatnonzero((B.h > lh) & (B.h >= rh)); il = np.flatnonzero((B.l < ll) & (B.l <= rl))
    ih = ih[ih + n < len(B.h)]; il = il[il + n < len(B.l)]
    return NS(ht=B.tend[ih + n], hp=B.h[ih], hb=B.tend[ih], lt=B.tend[il + n], lp=B.l[il], lb=B.tend[il])


def key_levels(E):
    md, dd = E.md, E.dd; rth = (md >= 570) & (md < 960)
    g = pd.DataFrame({'d': dd[rth], 'h': E.h[rth], 'l': E.l[rth], 'c': E.c[rth]}).groupby('d').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    prev = g.shift(1)
    on = (md >= 1080) | (md < 570); sess = dd + (md >= 1080)
    o = pd.DataFrame({'d': sess[on], 'h': E.h[on], 'l': E.l[on]}).groupby('d').agg(h=('h', 'max'), l=('l', 'min'))
    orr = (md >= 570) & (md < 585); r = pd.DataFrame({'d': dd[orr], 'h': E.h[orr], 'l': E.l[orr]}).groupby('d').agg(h=('h', 'max'), l=('l', 'min'))
    out = {}
    for d in g.index:
        v = [prev.h.get(d, np.nan), prev.l.get(d, np.nan), prev.c.get(d, np.nan), o.h.get(d, np.nan), o.l.get(d, np.nan), r.h.get(d, np.nan), r.l.get(d, np.nan)]
        out[int(d)] = np.array(v, float)
    return out


def load(L2, name):
    z = np.load(L2.CACHE(name)); E = NS(); E.t = z['t'].astype('datetime64[m]').astype(np.int64); E.shift = z['shift'].astype(float)
    E.o, E.h, E.l, E.c = (z[k].astype(float) + E.shift for k in ('o', 'h', 'l', 'c')); E.md = E.t % 1440; E.dd = E.t // 1440
    E.bars = {tf: agg(E, tf) for tf in (1, 5, 15, 60, 240)}
    for tf in (5, 15, 60, 240): E.bars[tf].key = E.bars[tf].tend // tf - 1
    E.fvg = {tf: fvgs(E.bars[tf], tf) for tf in (1, 5, 15, 60, 240)}; E.sw = swings(E.bars[5]); E.kl = key_levels(E)
    return E


def atr5(E, tau):
    B = E.bars[5]; i = np.searchsorted(B.tend, tau, 'right') - 1
    return B.atr[i] if i >= 0 else np.nan


def window(E, tau):
    a = np.searchsorted(E.t, tau - WIN); b = np.searchsorted(E.t, tau)
    if b - a < 60: return None
    return NS(t=E.t[a:b], smin=np.minimum.accumulate(E.l[a:b][::-1])[::-1], smax=np.maximum.accumulate(E.h[a:b][::-1])[::-1])


def after(W, ft, which):
    pos = np.searchsorted(W.t, ft); none = pos >= len(W.t); p2 = np.minimum(pos, len(W.t) - 1)
    v = W.smin[p2] if which == 'min' else W.smax[p2]
    return np.where(none, np.inf if which == 'min' else -np.inf, v)


def nearest(vals, d, ahead):
    vals = vals[np.isfinite(vals)]
    if not len(vals): return np.nan
    return (vals.min() if d > 0 else vals.max()) if ahead else (vals.max() if d > 0 else vals.min())


def levels(E, tau, d, price, A, tm=1.0, sm=0.5):
    W = window(E, tau)
    if W is None or not A == A: return None
    out = {}
    for tf in (1, 5, 15, 60):
        F = E.fvg[tf]; i0 = np.searchsorted(F.t, tau - LB[tf], 'right'); i1 = np.searchsorted(F.t, tau, 'right')
        typ, lo, hi, ft = F.typ[i0:i1], F.lo[i0:i1], F.hi[i0:i1], F.t[i0:i1]
        un = np.where(typ > 0, after(W, ft, 'min') > lo, after(W, ft, 'max') < hi)
        if d > 0:
            out[('T', 'fvg', tf)] = nearest(lo[un & (typ < 0) & (lo >= price + tm * A)], d, True)
            out[('S', 'fvg', tf)] = nearest(lo[un & (typ > 0) & (hi <= price) & (lo <= price - sm * A)], d, False)
        else:
            out[('T', 'fvg', tf)] = nearest(hi[un & (typ > 0) & (hi <= price - tm * A)], d, True)
            out[('S', 'fvg', tf)] = nearest(hi[un & (typ < 0) & (lo >= price) & (hi >= price + sm * A)], d, False)
    S = E.sw
    i0 = np.searchsorted(S.ht, tau - SWING_LB, 'right'); i1 = np.searchsorted(S.ht, tau, 'right'); hp = S.hp[i0:i1]; hp = hp[after(W, S.hb[i0:i1], 'max') <= hp]
    i0 = np.searchsorted(S.lt, tau - SWING_LB, 'right'); i1 = np.searchsorted(S.lt, tau, 'right'); lp = S.lp[i0:i1]; lp = lp[after(W, S.lb[i0:i1], 'min') >= lp]
    kl = E.kl.get(int(tau // 1440), np.full(7, np.nan))
    if d > 0:
        out[('T', 'sw')] = nearest(hp[hp >= price + tm * A], d, True); out[('S', 'sw')] = nearest(lp[lp <= price - sm * A], d, False)
        out[('T', 'kl')] = nearest(kl[kl >= price + tm * A], d, True); out[('S', 'kl')] = nearest(kl[kl <= price - sm * A], d, False)
    else:
        out[('T', 'sw')] = nearest(lp[lp <= price - tm * A], d, True); out[('S', 'sw')] = nearest(hp[hp >= price + sm * A], d, False)
        out[('T', 'kl')] = nearest(kl[kl <= price - tm * A], d, True); out[('S', 'kl')] = nearest(kl[kl >= price + sm * A], d, False)
    return out


def trail_events(E, tau, tau1, d, src):
    if src == 'sw':
        S = E.sw; t, p = (S.lt, S.lp) if d > 0 else (S.ht, S.hp)
        i0 = np.searchsorted(t, tau, 'right'); i1 = np.searchsorted(t, tau1, 'right'); return t[i0:i1], p[i0:i1]
    F = E.fvg[src]; i0 = np.searchsorted(F.t, tau, 'right'); i1 = np.searchsorted(F.t, tau1, 'right')
    m = F.typ[i0:i1] == d
    return F.t[i0:i1][m], (F.lo[i0:i1][m] if d > 0 else F.hi[i0:i1][m])


def sim_exit(E, tau, tau1, d, T, S, ev=None, market=False, entry_bar_target=True):
    ia = np.searchsorted(E.t, tau); ib = np.searchsorted(E.t, tau1)
    if ib <= ia: return None
    t, o, h, l, c = E.t[ia:ib], E.o[ia:ib], E.h[ia:ib], E.l[ia:ib], E.c[ia:ib]; n = ib - ia
    stop = np.full(n, S if S == S else (-np.inf if d > 0 else np.inf))
    if ev is not None:
        for ft, lv in zip(*ev):
            p = np.searchsorted(t, ft)
            if p < n: stop[p:] = np.maximum(stop[p:], lv) if d > 0 else np.minimum(stop[p:], lv)
    hasT = T == T
    if market:
        hs = (c <= stop) if d > 0 else (c >= stop); ht = ((c >= T) if d > 0 else (c <= T)) if hasT else np.zeros(n, bool)
        hit = hs | ht
        if not hit.any(): return None
        i = int(hit.argmax())
        if i + 1 >= n: return None
        return int(t[i + 1]) - 1, float(o[i + 1]), ('stop' if hs[i] else 'target')
    hs = (l <= stop) if d > 0 else (h >= stop)
    ht = ((h >= T + TICK) if d > 0 else (l <= T - TICK)) if hasT else np.zeros(n, bool)
    if not entry_bar_target: ht[0] = False
    hit = hs | ht
    if not hit.any(): return None
    i = int(hit.argmax())
    if hs[i]: return int(t[i]), float(min(stop[i], o[i]) if d > 0 else max(stop[i], o[i])), 'stop'
    return int(t[i]), float(max(T, o[i]) if d > 0 else min(T, o[i])), 'target'
