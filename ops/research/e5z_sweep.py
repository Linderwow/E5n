r"""E5Z - the wide sweep (E5X_PLAN amendment F): every rule and model lever, selected on 2013-19, confirmed on 2019-26.
    python e5z_sweep.py models     fit and cache every model variant (NQ/MNQ and ES/MES)
    python e5z_sweep.py run        score every variant, select, confirm -> appended to E5X_RESULTS.txt + cache\e5z_table.csv
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import e5x_lab as X
import e5x_dd as DD
import e5x_tf as TF
import e5n_train as TR
import e5n_features as F

ON = ['on_ret', 'on_rng', 'pm_ret', 'on_pos', 'on_vol']
R5 = 5 ** 0.5
T0 = X.T0
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def more(L2, nm):
    R = TR.Raw(L2, nm); mk = L2.Mins(nm); T = pd.read_parquet(os.path.join(X.CACHE, f'{nm}.parquet'), columns=['k', 'dec', 'date'])
    k = T.k.values; p_ = T.dec.values; out = pd.DataFrame({'k': k, 'dec': p_, 'date': T.date.values})
    out['lo5'] = sliding_window_view(R.L, 5, axis=1).min(2)[k, p_]; out['hi5'] = sliding_window_view(R.H, 5, axis=1).max(2)[k, p_]
    out['c2'] = R.C[k, np.minimum(p_ + 2, 389)]
    n = len(R.D); dc = R.C[:, -1]; dh = R.H.max(1); dl = R.L.min(1); pc = np.r_[np.nan, dc[:-1]]
    tr = np.maximum(dh - dl, np.maximum(np.abs(dh - pc), np.abs(dl - pc))); datr = pd.Series(tr).shift(1).rolling(14, min_periods=10).mean().values
    f = {c: np.full(n, np.nan) for c in ON}; vol = np.full(n, np.nan); tt = mk.t
    for s, d in enumerate(R.D.date):
        i0 = mk.days[d][0]; j0 = int(np.searchsorted(tt, tt[i0] - np.timedelta64(930, 'm')))
        if i0 - j0 < 30: continue
        same = mk.iid[j0:i0] == mk.iid[i0]
        if same.sum() < 30: continue
        o = mk.o[j0:i0][same]; h = mk.h[j0:i0][same]; l = mk.l[j0:i0][same]; v = mk.v[j0:i0][same]; m = mk.m[j0:i0][same]; O0 = mk.o[i0]
        hi, lo = h.max(), l.min(); f['on_ret'][s] = np.log(O0 / o[0]); f['on_rng'][s] = (hi - lo) / datr[s] if datr[s] == datr[s] else np.nan
        f['on_pos'][s] = (O0 - lo) / (hi - lo) if hi > lo else np.nan; vol[s] = v.sum()
        pm = np.where((m >= 480) & (m < 570))[0]
        if len(pm): f['pm_ret'][s] = np.log(O0 / o[pm[0]])
    with np.errstate(invalid='ignore', divide='ignore'): f['on_vol'] = vol / pd.Series(vol).shift(1).rolling(20, min_periods=10).mean().values
    for c in ON: out[c] = f[c][k]
    return out


def joined(fn, L2, a, b):
    A = fn(L2, a); A = A[A.date <= X.JOIN]; B = fn(L2, b); B = B[B.date > X.JOIN].assign(k=lambda d: d.k + 10 ** 6)
    return pd.concat([A, B], ignore_index=True).drop(columns='date')


def extras(tag):
    p = os.path.join(X.CACHE, f'e5z_extra_{tag}.parquet')
    if os.path.exists(p): return pd.read_parquet(p)
    L2 = TR.research()[0]; pair = ('nq', 'mnqx') if tag == 'nq' else ('es', 'mes'); bk = X.NQ if tag == 'nq' else X.ES
    E = bk.T[['k', 'dec']].merge(joined(TF.extras, L2, *pair), on=['k', 'dec'], how='left').merge(joined(more, L2, *pair), on=['k', 'dec'], how='left')
    assert len(E) == len(bk.T) and (E.k.values == bk.T.k.values).all()
    E.to_parquet(p); return E


def hgb(**kw):
    base = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
    base.update(kw); return lambda: HistGradientBoostingRegressor(**base)


class Forest:
    def __init__(self): self.m = RandomForestRegressor(n_estimators=100, min_samples_leaf=300, max_features=0.33, max_samples=0.3, n_jobs=-1, random_state=0)
    def fit(self, X_, y_, sample_weight=None): self.m.fit(np.where(np.isfinite(X_), X_, np.nan), y_); return self
    def predict(self, X_): return self.m.predict(np.where(np.isfinite(X_), X_, np.nan))


class Linear:
    def fit(self, X_, y_, sample_weight=None):
        Z = np.where(np.isfinite(X_), X_, np.nan); self.med = np.nanmedian(Z, 0); Z = np.where(np.isnan(Z), self.med, Z)
        self.mu = Z.mean(0); self.sd = Z.std(0); self.sd[self.sd == 0] = 1.0
        self.m = Ridge(alpha=100.0).fit(np.clip((Z - self.mu) / self.sd, -5, 5), y_); return self
    def predict(self, X_):
        Z = np.where(np.isfinite(X_), X_, np.nan); Z = np.where(np.isnan(Z), self.med, Z)
        return self.m.predict(np.clip((Z - self.mu) / self.sd, -5, 5))


def walk2(T, y, Xm, mk=None, window=None, half=None, clip=False, extra=None, quarterly=False):
    mk = mk or hgb(); dates = T.date.values.astype('datetime64[D]'); pred = np.full(len(T), np.nan); fin = np.isfinite(y)
    y0, y1 = 2013, int(pd.Timestamp(dates.max()).year)
    starts = [np.datetime64(f'{yy}-{mm:02d}-01') for yy in range(y0, y1 + 1) for mm in ((1, 4, 7, 10) if quarterly else (1,))]
    for a, b in zip(starts, starts[1:] + [np.datetime64('2100-01-01')]):
        te = (dates >= a) & (dates < b)
        if not te.any(): continue
        tr = (dates < a) & fin
        if window: tr &= dates >= a - np.timedelta64(int(365.25 * window), 'D')
        if tr.sum() < 5000: continue
        Xt, yt, dt_ = Xm[tr], y[tr], dates[tr]
        if extra is not None:
            Xe, ye, de = extra; me = (de < a) & np.isfinite(ye); Xt = np.r_[Xt, Xe[me]]; yt = np.r_[yt, ye[me]]; dt_ = np.r_[dt_, de[me]]
        if clip: lo, hi = np.percentile(yt, [1, 99]); yt = np.clip(yt, lo, hi)
        w = 0.5 ** ((a - dt_).astype(float) / 365.25 / half) if half else None
        m = mk(); m.fit(Xt, yt, sample_weight=w); pred[te] = m.predict(Xm[te])
    return pred


MODELS = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19']
MDESC = {'M01': '+ 65-minute and daily indicators', 'M02': '+ overnight features', 'M03': '+ 65-minute, daily and overnight', 'M04': 'absolute-error loss',
         'M05': 'target clipped at 1% / 99%', 'M06': 'learning rate 0.03, 500 trees', 'M07': '31 leaves', 'M08': 'minimum leaf 1,000', 'M09': '150 trees',
         'M10': 'L2 = 10', 'M11': '5-year rolling window', 'M12': '3-year rolling window', 'M13': 'recency weights (half-life 3 years)',
         'M14': 'pooled training on both indexes', 'M15': 'mean of deployed, M06, M07, M08', 'M16': 'refit every quarter', 'M17': 'random forest', 'M18': 'mean of deployed and random forest', 'M19': 'ridge regression (linear)'}


def fc(mid, tag):
    return np.load(os.path.join(X.CACHE, f'e5z_{mid}_{tag}.npy'))


def models():
    t0 = time.time(); X.setup(); E = {t: extras(t) for t in ('nq', 'es')}; B = {'nq': X.NQ, 'es': X.ES}
    base = {t: B[t].T[F.FEATS].values for t in B}; y = {t: B[t].T.y.values for t in B}; dts = {t: B[t].T.date.values.astype('datetime64[D]') for t in B}
    XM = {'M01': lambda t: np.c_[base[t], E[t][TF.HTF].values], 'M02': lambda t: np.c_[base[t], E[t][ON].values], 'M03': lambda t: np.c_[base[t], E[t][TF.HTF].values, E[t][ON].values]}
    KW = {'M04': dict(mk=hgb(loss='absolute_error')), 'M05': dict(clip=True), 'M06': dict(mk=hgb(learning_rate=0.03, max_iter=500)), 'M07': dict(mk=hgb(max_leaf_nodes=31)),
          'M08': dict(mk=hgb(min_samples_leaf=1000)), 'M09': dict(mk=hgb(max_iter=150)), 'M10': dict(mk=hgb(l2_regularization=10.0)), 'M11': dict(window=5), 'M12': dict(window=3),
          'M13': dict(half=3.0), 'M16': dict(quarterly=True), 'M17': dict(mk=Forest), 'M19': dict(mk=Linear)}
    for mid in MODELS:
        for t in ('nq', 'es'):
            p = os.path.join(X.CACHE, f'e5z_{mid}_{t}.npy')
            if os.path.exists(p): continue
            o = 'es' if t == 'nq' else 'nq'
            if mid == 'M01' and os.path.exists(os.path.join(X.CACHE, f'm1_{t}.npy')): v = np.load(os.path.join(X.CACHE, f'm1_{t}.npy'))
            elif mid in XM: v = walk2(B[t].T, y[t], XM[mid](t))
            elif mid == 'M14': v = walk2(B[t].T, y[t], base[t], extra=(base[o], y[o], dts[o]))
            elif mid == 'M15': v = np.mean([X.F_NQ if t == 'nq' else X.F_ES, fc('M06', t), fc('M07', t), fc('M08', t)], axis=0)
            elif mid == 'M18': v = np.mean([X.F_NQ if t == 'nq' else X.F_ES, fc('M17', t)], axis=0)
            else: v = walk2(B[t].T, y[t], base[t], **KW[mid])
            np.save(p, v); print(f"{mid} {t} done {time.time() - t0:.0f} s", flush=True)
    print(f"models done {time.time() - t0:.0f} s")


class Arr:
    def __init__(self, bk, tag):
        T = bk.T; E = extras(tag); n0 = np.nan_to_num; self.bk = bk
        self.entry = bk.entry; self.exitp = bk.exitp; self.o930 = bk.o930; self.m = 570 + T.dec.values; self.vol = T.vol30.values
        self.vw = n0(T.vwapd.values); self.e50 = n0(T['e5030'].values); self.day = n0(E.d_s20.values); self.vr = bk.volratio()
        self.f15 = n0(np.load(os.path.join(X.CACHE, f'x5_{tag}.npy'))); self.f30 = n0(np.load(os.path.join(X.CACHE, f'c5_{tag}.npy'))); self.f60 = n0(np.load(os.path.join(X.CACHE, f'c4_{tag}.npy')))
        self.lo5 = E.lo5.values; self.hi5 = E.hi5.values; self.c2 = E.c2.values; self.dec = T.dec.values; self.f = X.F_NQ if tag == 'nq' else X.F_ES


def sim(A, f, o, ticks=2.35):
    bk = A.bk; c1 = X.SIDE(ticks); n = len(bk.st); net = np.zeros(n); trd = np.zeros(n)
    entry, exitp, o930, vol, lo5, hi5 = A.entry, A.exitp, A.o930, A.vol, A.lo5, A.hi5
    thrL = o.get('thrL', T0); thrS = o.get('thrS', T0); zc = o.get('z'); allow = o.get('allow'); conf = o.get('conf'); anti = o.get('anti'); gate = o.get('gate')
    minhold = o.get('minhold', 0); cool = o.get('cool', 0); stop = o.get('stop'); tgt = o.get('tgt'); turn = o.get('turn', False)
    long_ok = o.get('long', True); short_ok = o.get('short', True); risk = stop is not None or tgt is not None; skip = o.get('skip'); tx = o.get('tx'); c2 = A.c2
    for d in range(n):
        a, b = bk.st[d], bk.en[d]; pos = 0; pnl = 0.0; sides = 0; last = 0.0; held = 0; cl = 0; epx = 0.0; S = 0.0; cost2 = 2 * c1 * o930[a]
        for i in range(a, b):
            fi = f[i]
            if fi != fi: continue
            if skip is not None and skip[i]: continue
            if pos != 0: pnl += pos * (entry[i] - last); last = entry[i]; held += 1
            tl = thrL; ts = thrS
            if zc is not None:
                mv = zc * vol[i] * R5
                if mv == mv:
                    if mv > tl: tl = mv
                    if mv > ts: ts = mv
            t = 1 if fi > tl else (-1 if fi < -ts else 0)
            if (t == 1 and not long_ok) or (t == -1 and not short_ok): t = 0
            if t != 0 and conf is not None:
                for c in conf:
                    if c[i] * t <= 0: t = 0; break
            if t != 0 and anti is not None and anti[i] * t >= 0: t = 0
            if t != 0 and gate is not None and not gate[i]: t = 0
            if turn and t == 0 and pos != 0 and fi * pos >= 0: t = pos
            new = t
            if new != 0 and new != pos and ((allow is not None and not allow[i]) or cl > 0): new = 0
            if pos != 0 and new != pos and held < minhold: new = pos
            if cl > 0: cl -= 1
            if new != pos:
                if pos != 0 and cool and pos * (entry[i] - epx) - cost2 < 0: cl = cool
                sides += abs(new - pos); pos = new; last = entry[i]
                if new != 0: epx = entry[i]; held = 0; S = vol[i] * R5 * entry[i]
            if tx and pos != 0:
                pnl += pos * (c2[i] - last); sides += 1; pos = 0; continue
            if risk and pos != 0 and S == S and S > 0:
                if stop is not None:
                    sp = epx - pos * stop * S
                    if (lo5[i] <= sp) if pos > 0 else (hi5[i] >= sp):
                        pnl += pos * (sp - last); sides += 1; pos = 0
                        if cool: cl = cool
                        continue
                if tgt is not None:
                    tp = epx + pos * tgt * S
                    if (hi5[i] >= tp) if pos > 0 else (lo5[i] <= tp): pnl += pos * (tp - last); sides += 1; pos = 0
        if pos != 0: pnl += pos * (exitp[b - 1] - last); sides += 1
        net[d] = pnl / o930[a] - sides * c1; trd[d] = sides / 2
    return pd.Series(net[bk.keep] * X.U, index=bk.dates[bk.keep]), float(trd[bk.keep].sum())


RULES = [
    ('R01', 'threshold 0.75 x T', 'thr', lambda A: dict(thrL=0.75 * T0, thrS=0.75 * T0)), ('R02', 'threshold 1.25 x T', 'thr', lambda A: dict(thrL=1.25 * T0, thrS=1.25 * T0)),
    ('R03', 'threshold 1.5 x T', 'thr', lambda A: dict(thrL=1.5 * T0, thrS=1.5 * T0)), ('R04', 'threshold 2 x T', 'thr', lambda A: dict(thrL=2 * T0, thrS=2 * T0)),
    ('R05', 'long 1.5 T, short T', 'thr', lambda A: dict(thrL=1.5 * T0)), ('R06', 'long T, short 1.5 T', 'thr', lambda A: dict(thrS=1.5 * T0)),
    ('R07', 'long only', 'thr', lambda A: dict(short=False)), ('R08', 'short only', 'thr', lambda A: dict(long=False)),
    ('R09', 'threshold max(T, 0.10 x move)', 'thr', lambda A: dict(z=0.10)), ('R10', 'threshold max(T, 0.15 x move)', 'thr', lambda A: dict(z=0.15)),
    ('R11', 'threshold max(T, 0.20 x move)', 'thr', lambda A: dict(z=0.20)),
    ('R12', 'no entry before 10:00', 'time', lambda A: dict(allow=A.m >= 600)), ('R13', 'no entry 12:00-13:30', 'time', lambda A: dict(allow=~((A.m >= 720) & (A.m < 810)))),
    ('R14', 'no entry after 15:30', 'time', lambda A: dict(allow=A.m <= 930)), ('R15', 'entries only 09:45-11:30 and 14:00-15:55', 'time', lambda A: dict(allow=(A.m <= 690) | (A.m >= 840))),
    ('R16', 'minimum hold 10 minutes', 'exit', lambda A: dict(minhold=2)), ('R17', 'after a losing trade skip 2 decisions', 'exit', lambda A: dict(cool=2)),
    ('R18', 'stop at 1 move', 'exit', lambda A: dict(stop=1.0)), ('R19', 'stop at 1.5 moves', 'exit', lambda A: dict(stop=1.5)), ('R20', 'stop at 2.5 moves', 'exit', lambda A: dict(stop=2.5)),
    ('R21', 'target at 3 moves', 'exit', lambda A: dict(tgt=3.0)), ('R22', 'stop 1.5 + target 3 moves', 'exit', lambda A: dict(stop=1.5, tgt=3.0)),
    ('R23', '30-minute EMA(50) side', 'conf', lambda A: dict(conf=[A.e50])), ('R24', 'daily trend side', 'conf', lambda A: dict(conf=[A.day])),
    ('R25', 'VWAP side', 'conf', lambda A: dict(conf=[A.vw])), ('R26', 'against the VWAP side', 'conf', lambda A: dict(anti=A.vw)),
    ('R27', 'EMA(50) and VWAP sides', 'conf', lambda A: dict(conf=[A.e50, A.vw])),
    ('R28', '15-minute model agrees', 'conf', lambda A: dict(conf=[A.f15])), ('R29', '30-minute model agrees', 'conf', lambda A: dict(conf=[A.f30])),
    ('R30', '60-minute model agrees', 'conf', lambda A: dict(conf=[A.f60])), ('R31', '15- and 60-minute models agree', 'conf', lambda A: dict(conf=[A.f15, A.f60])),
    ('R32', '15-, 30- and 60-minute models agree', 'conf', lambda A: dict(conf=[A.f15, A.f30, A.f60])),
    ('R33', 'only above-median volatility', 'time', lambda A: dict(gate=A.vr > 1.0)), ('R34', 'only below-median volatility', 'time', lambda A: dict(gate=A.vr <= 1.0)),
    ('R35', 'hold until the forecast turns', 'exit', lambda A: dict(turn=True)),
    ('R36', 'decide every 10 minutes', 'time', lambda A: dict(skip=(A.dec - 15) % 10 != 0)), ('R37', 'decide every 15 minutes', 'time', lambda A: dict(skip=(A.dec - 15) % 15 != 0)),
    ('R38', 'exit 3 minutes after the entry', 'exit', lambda A: dict(tx=True))]


def run():
    t0 = time.time(); X.setup(); AN, AE = Arr(X.NQ, 'nq'), Arr(X.ES, 'es')
    sa = lambda s: X.sharpe(s[s.index <= X.JOIN]); sb = lambda s: X.sharpe(s[s.index > X.JOIN]); pa = lambda s: s[s.index <= X.JOIN]; pb = lambda s: s[s.index > X.JOIN]
    bn, nbn = sim(AN, AN.f, {}); be, _ = sim(AE, AE.f, {}); chk, _ = X.BASE[2.35]; assert abs(bn.sum() - chk.sum()) < 1e-6
    yrs = len(bn) / 252
    say(); say("=" * 124); say(f"THE WIDE SWEEP  e5z_sweep.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5X_PLAN amendment F)  1 micro, today's price, 2.35 ticks + $0.61 a side")
    say("select on 2013-19 (NQ and ES both better than the base), confirm on 2019-26 (MNQ and MES both better)"); say("=" * 124)
    rows = []; series = {}
    def add(vid, desc, grp, xn, tn, xe):
        med, bad, p, _ = DD.boot_dd(xn)
        r = dict(id=vid, desc=desc, grp=grp, a_nq=sa(xn), a_es=sa(xe), b_mnq=sb(xn), b_mes=sb(xe), usd=xn.sum() / yrs, trades=tn / yrs, dda=DD.mdd(pa(xn)), ddb=DD.mdd(pb(xn)),
                 dd_typ=med, dd_bad=bad, dd_p=p, worst=xn.min())
        rows.append(r); series[vid] = (xn, xe); return r
    base = add('BASE', 'E5n as deployed', '-', bn, nbn, be)
    for vid, desc, grp, mk in RULES:
        xn, tn = sim(AN, AN.f, mk(AN)); xe, _ = sim(AE, AE.f, mk(AE)); add(vid, desc, grp, xn, tn, xe)
    print(f"rules done {time.time() - t0:.0f} s", flush=True)
    for mid in MODELS:
        if not os.path.exists(os.path.join(X.CACHE, f'e5z_{mid}_es.npy')): say(f"{mid} not fitted yet - left out"); continue
        xn, tn = sim(AN, fc(mid, 'nq'), {}); xe, _ = sim(AE, fc(mid, 'es'), {}); add(mid, MDESC[mid], 'model', xn, tn, xe)
    D = pd.DataFrame(rows).set_index('id'); b = D.loc['BASE']
    D['selected'] = (D.a_nq > b.a_nq) & (D.a_es > b.a_es); D.loc['BASE', 'selected'] = False
    sel = D.index[D.selected].tolist()
    say(f"{'id':5s}{'variant':44s} select: NQ   ES  | confirm: MNQ  MES |   $/yr  trades/yr | maxDD 13-19 / 19-26 | 1-yr DD typical / >$2,500")
    def line(i):
        r = D.loc[i]
        say(f"{i:5s}{r.desc[:43]:44s}        {r.a_nq:5.2f} {r.a_es:5.2f} |          {r.b_mnq:5.2f} {r.b_mes:5.2f} | {r.usd:6,.0f}  {r.trades:6,.0f}   | {r.dda:7,.0f} / {r.ddb:6,.0f}     | "
            f"{r.dd_typ:6,.0f} / {r.dd_p:4.0%}" + ('   SELECTED' if r.selected else ''))
    for i in D.index: line(i)
    combo = None
    if sel:
        ms = [i for i in sel if D.loc[i, 'grp'] == 'model']; bestm = max(ms, key=lambda i: D.loc[i, 'a_nq'] + D.loc[i, 'a_es']) if ms else None
        opts_n, opts_e, parts = {}, {}, []
        for g in ('thr', 'time', 'exit', 'conf'):
            c = [i for i in sel if D.loc[i, 'grp'] == g]
            if not c: continue
            bi = max(c, key=lambda i: D.loc[i, 'a_nq'] + D.loc[i, 'a_es']); mk = next(m for v, _, _, m in RULES if v == bi)
            for k_, v_ in mk(AN).items(): opts_n[k_] = opts_n.get(k_, []) + v_ if k_ == 'conf' and k_ in opts_n else v_
            for k_, v_ in mk(AE).items(): opts_e[k_] = opts_e.get(k_, []) + v_ if k_ == 'conf' and k_ in opts_e else v_
            parts.append(bi)
        if bestm or len(parts) > 1:
            fn_ = fc(bestm, 'nq') if bestm else AN.f; fe_ = fc(bestm, 'es') if bestm else AE.f
            xn, tn = sim(AN, fn_, opts_n); xe, _ = sim(AE, fe_, opts_e); combo = '+'.join(([bestm] if bestm else []) + parts)
            r = add('COMBO', combo, 'combo', xn, tn, xe); D = pd.DataFrame(rows).set_index('id'); b = D.loc['BASE']
            D['selected'] = (D.a_nq > b.a_nq) & (D.a_es > b.a_es); D.loc['BASE', 'selected'] = False; say(); line('COMBO')
    sel = D.index[D.selected].tolist()
    say(); say(f"SELECTED on 2013-19: {len(sel)} of {len(D) - 1}: " + (", ".join(sel) if sel else "none"))
    conf = []
    for i in sel:
        r = D.loc[i]; xn = series[i][0]; p = X.boot_p(pb(xn), pb(bn)); ok = r.b_mnq > b.b_mnq and r.b_mes > b.b_mes
        strong = ok and p < 0.05 / len(sel)
        say(f"   {i:5s} {r.desc[:50]:50s} 2019-26 Sharpe MNQ {r.b_mnq:.2f} vs {b.b_mnq:.2f}, MES {r.b_mes:.2f} vs {b.b_mes:.2f}; P {p:.3f} -> "
            f"{'CONFIRMED (strongly)' if strong else 'confirmed in sign' if ok else 'not confirmed'}")
        if ok: conf.append(i)
    calm = [i for i in D.index if i != 'BASE' and D.loc[i, 'dda'] < b.dda and D.loc[i, 'ddb'] < b.ddb and D.loc[i, 'usd'] >= 0.9 * b.usd]
    say(); say("CALMER (lower max drawdown in both halves, >= 90% of the profit): " + (", ".join(f"{i} (1-yr DD typical ${D.loc[i, 'dd_typ']:,.0f}, >$2,500 in {D.loc[i, 'dd_p']:.0%})" for i in calm) if calm else "none"))
    say(f"CONFIRMED: " + (", ".join(conf) if conf else "none")); say(f"run {time.time() - t0:.0f} s")
    D.to_csv(os.path.join(X.CACHE, 'e5z_table.csv'))
    with open(X.OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'models': models()
    elif cmd == 'run': run()
    else: print(__doc__)
