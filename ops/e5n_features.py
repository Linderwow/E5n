import numpy as np, pandas as pd

TFS = (5, 15, 30)
BASE = ['r5', 'r15', 'r30', 'r60', 'ropen', 'gap', 'prevday', 'vol30', 'vol120', 'dvol20', 'vwapd', 'rngpos', 'vratio', 'tod', 'dow', 'r15z', 'ropenz']
IND = ['rsi', 'macd', 'bbb', 'stk', 'std', 'wr', 'cci', 'adx', 'did', 'atrp', 'roc', 'e20', 'e50', 'don', 'kel', 'vz', 'obv', 'mfi', 'upv']
DAYF = ['prng', 'pdh', 'pdl', 'pdc', 'orpos', 'orbrk', 'vwz', 'opex', 'mend']
FEATS = BASE + [f'{i}{tf}' for tf in TFS for i in IND] + DAYF
STEP = 5
HORIZON = 5
FIRST_DEC = 15
P_REF = 27875.38793103448
N_USD = 2 * P_REF
SIDE_USD = 0.5 * 2.35 + 0.61
THRESHOLD = 4 * SIDE_USD / N_USD


def third_friday(y, m):
    d = pd.Timestamp(y, m, 1); return d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)


def tf_bars(S, tf):
    n = S.C.shape[0]; nb = 390 // tf
    r = lambda a: a[:, :nb * tf].reshape(n, nb, tf)
    return nb, r(S.O)[:, :, 0].ravel(), r(S.H).max(2).ravel(), r(S.L).min(2).ravel(), r(S.C)[:, :, -1].ravel(), r(S.V).sum(2).ravel()


def indicators(o, h, l, c, v):
    s = pd.Series(c); pc = s.shift(1).values
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))); tr[0] = h[0] - l[0]
    w = lambda x, n: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values
    atr = w(tr, 14); d = np.diff(c, prepend=c[0])
    up = w(np.clip(d, 0, None), 14); dn = w(np.clip(-d, 0, None), 14)
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        out['rsi'] = np.where(dn > 0, 100 - 100 / (1 + up / dn), 100.0)
        e12, e26 = s.ewm(span=12, adjust=False).mean().values, s.ewm(span=26, adjust=False).mean().values
        m = e12 - e26; out['macd'] = (m - pd.Series(m).ewm(span=9, adjust=False).mean().values) / c
        ma = s.rolling(20).mean().values; sd = s.rolling(20).std(ddof=0).values; out['bbb'] = (c - (ma - 2 * sd)) / (4 * sd)
        hh14 = pd.Series(h).rolling(14).max().values; ll14 = pd.Series(l).rolling(14).min().values
        k = (c - ll14) / (hh14 - ll14); out['stk'] = k; out['std'] = pd.Series(k).rolling(3).mean().values; out['wr'] = (hh14 - c) / (hh14 - ll14) * -100
        tp = (h + l + c) / 3; tps = pd.Series(tp); out['cci'] = (tp - tps.rolling(20).mean().values) / (0.015 * 0.7979 * tps.rolling(20).std(ddof=0).values)
        upm = h - np.r_[h[0], h[:-1]]; dnm = np.r_[l[0], l[:-1]] - l
        pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0); ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
        pdi = 100 * w(pdm, 14) / atr; ndi = 100 * w(ndm, 14) / atr; dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
        out['adx'] = w(np.nan_to_num(dx), 14); out['did'] = pdi - ndi
        out['atrp'] = atr / c; out['roc'] = c / s.shift(10).values - 1
        e20 = s.ewm(span=20, adjust=False).mean().values; e50 = s.ewm(span=50, adjust=False).mean().values
        out['e20'] = (c - e20) / atr; out['e50'] = (c - e50) / atr
        hh20 = pd.Series(h).rolling(20).max().values; ll20 = pd.Series(l).rolling(20).min().values; out['don'] = (c - ll20) / (hh20 - ll20)
        out['kel'] = (c - e20) / (2 * atr)
        vs = pd.Series(v); out['vz'] = (v - vs.rolling(20).mean().values) / vs.rolling(20).std(ddof=0).values
        obv = np.cumsum(np.sign(d) * v); out['obv'] = (obv - pd.Series(obv).shift(10).values) / (vs.rolling(20).mean().values * 10)
        mf = tp * v; dtp = np.diff(tp, prepend=tp[0]); pos = pd.Series(np.where(dtp > 0, mf, 0.0)).rolling(14).sum().values
        neg = pd.Series(np.where(dtp < 0, mf, 0.0)).rolling(14).sum().values; out['mfi'] = 100 - 100 / (1 + pos / neg)
        out['upv'] = pd.Series(np.where(c > o, v, 0.0)).rolling(20).sum().values / vs.rolling(20).sum().values
    return out


def features(S):
    D = S.D; n = len(D); lc = np.log(S.C); lo = np.log(S.O)
    r1 = np.diff(lc, axis=1, prepend=lc[:, :1])
    tp = (S.H + S.L + S.C) / 3; vw = np.cumsum(tp * S.V, 1) / np.maximum(np.cumsum(S.V, 1), 1)
    hi = np.maximum.accumulate(S.H, 1); low = np.minimum.accumulate(S.L, 1)
    cv = np.cumsum(S.V, 1); v15 = cv - np.c_[np.zeros((n, 15)), cv[:, :-15]]
    v15avg = pd.DataFrame(v15).shift(1).rolling(20, min_periods=10).mean().values
    prevC = np.r_[np.nan, S.C[:-1, -1]]; prevO = np.r_[np.nan, S.O[:-1, 0]]
    same_id = np.r_[False, D.iid.values[1:] == D.iid.values[:-1]]
    same = same_id & np.r_[False, (D.date.diff().dt.days.values[1:] <= 5)]
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = np.where(same, np.log(S.O[:, 0] / prevC), np.nan); prevday = np.where(same, np.log(prevC / prevO), np.nan)
    dret = np.log(S.C[:, -1] / S.O[:, 0]); dvol = pd.Series(dret).shift(1).rolling(20, min_periods=10).std().values
    rows = []
    for p_ in range(STEP, 390 - HORIZON + 1, STEP):
        p = p_ - 1; f = {}
        for w in (5, 15, 30, 60): f[f'r{w}'] = lc[:, p] - lc[:, p - w] if p - w >= 0 else np.full(n, np.nan)
        f['ropen'] = lc[:, p] - lo[:, 0]; f['gap'] = gap; f['prevday'] = prevday
        f['vol30'] = r1[:, max(1, p - 29):p + 1].std(1) if p >= 10 else np.full(n, np.nan)
        f['vol120'] = r1[:, max(1, p - 119):p + 1].std(1) if p >= 30 else np.full(n, np.nan)
        f['dvol20'] = dvol; f['vwapd'] = lc[:, p] - np.log(vw[:, p])
        with np.errstate(invalid='ignore', divide='ignore'):
            f['rngpos'] = (S.C[:, p] - low[:, p]) / np.where(hi[:, p] > low[:, p], hi[:, p] - low[:, p], np.nan)
            f['vratio'] = v15[:, p] / v15avg[:, p] if p >= 14 else np.full(n, np.nan)
            f['r15z'] = f['r15'] / (f['vol30'] * np.sqrt(15)); f['ropenz'] = f['ropen'] / (f['vol30'] * np.sqrt(p + 1))
        f['tod'] = np.full(n, 570 + p_); f['dow'] = D.date.dt.dayofweek.values.astype(float)
        e = p + 1; x = min(p + HORIZON, 389)
        f['y'] = lc[:, x] - lo[:, e]; f['entry'] = S.O[:, e]; f['exitp'] = S.C[:, x]; f['dec'] = np.full(n, p_)
        f['k'] = np.arange(n); f['date'] = D.date.values
        rows.append(pd.DataFrame(f))
    T = pd.concat(rows, ignore_index=True).sort_values(['k', 'dec']).reset_index(drop=True)
    T['open930'] = S.O[T.k.values, 0]
    k = T.k.values; p_ = T.dec.values; p = p_ - 1
    for tf in TFS:
        nb, o, h, l, c, v = tf_bars(S, tf); ind = indicators(o, h, l, c, v)
        j = k * nb + (p_ // tf) - 1; ok = j >= 0; j = np.clip(j, 0, None)
        for nm in IND: T[f'{nm}{tf}'] = np.where(ok, ind[nm][j], np.nan)
    dh, dl, dc = S.H.max(1), S.L.min(1), S.C[:, -1]
    pc_ = np.r_[np.nan, dc[:-1]]; dtr = np.maximum(dh - dl, np.maximum(np.abs(dh - pc_), np.abs(dl - pc_)))
    datr = pd.Series(dtr).shift(1).rolling(14, min_periods=10).mean().values
    pH, pL, pC = np.where(same_id, np.r_[np.nan, dh[:-1]], np.nan), np.where(same_id, np.r_[np.nan, dl[:-1]], np.nan), np.where(same_id, pc_, np.nan)
    px = S.C[k, p]
    with np.errstate(invalid='ignore', divide='ignore'):
        T['prng'] = ((pH - pL) / datr)[k]; T['pdh'] = (px - pH[k]) / datr[k]; T['pdl'] = (px - pL[k]) / datr[k]; T['pdc'] = (px - pC[k]) / datr[k]
        orh, orl = S.H[:, :30].max(1), S.L[:, :30].min(1)
        T['orpos'] = np.where(p_ >= 30, (px - orl[k]) / (orh[k] - orl[k]), np.nan)
        T['orbrk'] = np.where(p_ >= 30, np.where(px > orh[k], 1.0, np.where(px < orl[k], -1.0, 0.0)), np.nan)
        sd30 = pd.DataFrame(S.C).T.rolling(30, min_periods=10).std().T.values
        T['vwz'] = (px - vw[k, p]) / sd30[k, p]
    dates = pd.DatetimeIndex(D.date.values)
    tf_ = np.array([(third_friday(d.year, d.month) - d).days for d in dates], float)
    nxt = np.array([(third_friday(d.year + (d.month == 12), d.month % 12 + 1) - d).days for d in dates], float)
    T['opex'] = np.where(tf_ >= 0, tf_, nxt)[k]
    T['mend'] = mend_flags(dates).astype(float)[k]
    return T[T.dec >= FIRST_DEC].reset_index(drop=True)


def mend_flags(dates):
    me = dates.to_period('M')
    return pd.Series(me).shift(-2).values != me.values
