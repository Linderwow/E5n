r"""E5T family NL - a model leg on the Globex night session (E5T_PLAN.txt).
    python e5t_night.py build     night sessions, features, walk-forward forecasts -> cache\e5t_*  (about 30 minutes)
    python e5t_night.py run       trials NL1-NL4 -> E5T_RESULTS.txt
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_features as F
import e5n_train as TR

N = 930; FIRST = 30; LAST = 920; OUT = os.path.join(HERE, 'E5T_RESULTS.txt'); C = os.path.join(X.CACHE, 'e5t_')
NFEATS = F.FEATS
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


class Night:
    def __init__(self, L2, name):
        mk = L2.Mins(name); m = mk.m; cal = mk.date.astype('datetime64[D]')
        night = (m >= 1080) | (m < 570)
        nd = np.where(m >= 1080, cal + np.timedelta64(1, 'D'), cal); pos = np.where(m >= 1080, m - 1080, m + 360)
        rth = {pd.Timestamp(d): mk.days[d] for d in mk.dates}; dates = list(rth); prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
        idx = np.flatnonzero(night); keys = nd[idx]; brk = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(idx)]
        rows = []; O = []; Hh = []; Ll = []; Cc = []; V = []; P = []
        for a, b in zip(brk[:-1], brk[1:]):
            ii = idx[a:b]; d = pd.Timestamp(keys[a])
            if d not in rth or d not in prev: continue
            pp = pos[ii]
            if len(ii) < 600 or pp.min() > 10 or pp.max() < 920 or len(set(mk.iid[ii])) != 1: continue
            arr = {k: np.full(N, np.nan) for k in 'ohlcv'}
            for k, src in (('o', mk.o), ('h', mk.h), ('l', mk.l), ('c', mk.c), ('v', mk.v)): arr[k][pp] = src[ii]
            c = pd.Series(arr['c']).ffill().bfill().values; o = np.where(np.isnan(arr['o']), c, arr['o'])
            h = np.where(np.isnan(arr['h']), c, arr['h']); l = np.where(np.isnan(arr['l']), c, arr['l']); v = np.nan_to_num(arr['v'])
            r0, r1 = rth[prev[d]]; same = mk.iid[r0] == mk.iid[ii[0]]
            rows.append(dict(date=d, iid=int(mk.iid[ii[0]]), r_o=mk.o[r0], r_c=mk.c[r1], r_h=mk.h[r0:r1 + 1].max(), r_l=mk.l[r0:r1 + 1].min(), same=bool(same)))
            O.append(o); Hh.append(h); Ll.append(l); Cc.append(c); V.append(v); P.append(~np.isnan(arr['c']))
        self.D = pd.DataFrame(rows); self.O, self.H, self.L, self.C, self.V = (np.array(x) for x in (O, Hh, Ll, Cc, V)); self.present = np.array(P)


def features(S):
    D = S.D; n = len(D); lc = np.log(S.C); lo = np.log(S.O)
    r1 = np.diff(lc, axis=1, prepend=lc[:, :1])
    tp = (S.H + S.L + S.C) / 3; vw = np.cumsum(tp * S.V, 1) / np.maximum(np.cumsum(S.V, 1), 1)
    hi = np.maximum.accumulate(S.H, 1); low = np.minimum.accumulate(S.L, 1)
    cv = np.cumsum(S.V, 1); v15 = cv - np.c_[np.zeros((n, 15)), cv[:, :-15]]
    v15avg = pd.DataFrame(v15).shift(1).rolling(20, min_periods=10).mean().values
    same = D.same.values; rC, rO, rH, rL = D.r_c.values, D.r_o.values, D.r_h.values, D.r_l.values
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = np.where(same, np.log(S.O[:, 0] / rC), np.nan); prevday = np.log(rC / rO)
    dvol = pd.Series(prevday).rolling(20, min_periods=10).std().values
    pc = np.r_[np.nan, rC[:-1]]; dtr = np.maximum(rH - rL, np.maximum(np.abs(rH - pc), np.abs(rL - pc))); datr = pd.Series(dtr).rolling(14, min_periods=10).mean().values
    rows = []
    for p_ in range(FIRST, LAST + 1, 5):
        p = p_ - 1; f = {}
        for w in (5, 15, 30, 60): f[f'r{w}'] = lc[:, p] - lc[:, p - w] if p - w >= 0 else np.full(n, np.nan)
        f['ropen'] = lc[:, p] - lo[:, 0]; f['gap'] = gap; f['prevday'] = prevday
        f['vol30'] = r1[:, max(1, p - 29):p + 1].std(1); f['vol120'] = r1[:, max(1, p - 119):p + 1].std(1)
        f['dvol20'] = dvol; f['vwapd'] = lc[:, p] - np.log(vw[:, p])
        with np.errstate(invalid='ignore', divide='ignore'):
            f['rngpos'] = (S.C[:, p] - low[:, p]) / np.where(hi[:, p] > low[:, p], hi[:, p] - low[:, p], np.nan)
            f['vratio'] = v15[:, p] / v15avg[:, p]
            f['r15z'] = f['r15'] / (f['vol30'] * np.sqrt(15)); f['ropenz'] = f['ropen'] / (f['vol30'] * np.sqrt(p + 1))
        f['tod'] = np.full(n, float(p_)); f['dow'] = D.date.dt.dayofweek.values.astype(float)
        e = p + 1; x = min(p + 5, N - 1); x15 = min(p + 15, N - 1)
        f['y'] = lc[:, x] - lo[:, e]; f['y15'] = lc[:, x15] - lo[:, e]; f['entry'] = S.O[:, e]; f['exitp'] = S.C[:, x]; f['dec'] = np.full(n, p_)
        f['has_bar'] = S.present[:, p]; f['k'] = np.arange(n); f['date'] = D.date.values
        rows.append(pd.DataFrame(f))
    T = pd.concat(rows, ignore_index=True).sort_values(['k', 'dec']).reset_index(drop=True)
    T['open930'] = S.O[T.k.values, 0]
    k = T.k.values; p_ = T.dec.values; p = p_ - 1
    for tf in F.TFS:
        nb = N // tf; r = lambda a: a[:, :nb * tf].reshape(n, nb, tf)
        o, h, l, c, v = r(S.O)[:, :, 0].ravel(), r(S.H).max(2).ravel(), r(S.L).min(2).ravel(), r(S.C)[:, :, -1].ravel(), r(S.V).sum(2).ravel()
        ind = F.indicators(o, h, l, c, v); j = k * nb + (p_ // tf) - 1; ok = j >= 0; j = np.clip(j, 0, None)
        for nm in F.IND: T[f'{nm}{tf}'] = np.where(ok, ind[nm][j], np.nan)
    px = S.C[k, p]
    with np.errstate(invalid='ignore', divide='ignore'):
        T['prng'] = ((rH - rL) / datr)[k]; T['pdh'] = (px - rH[k]) / datr[k]; T['pdl'] = (px - rL[k]) / datr[k]; T['pdc'] = (px - rC[k]) / datr[k]
        T.loc[~same[k], ['pdh', 'pdl', 'pdc']] = np.nan
        orh, orl = S.H[:, :30].max(1), S.L[:, :30].min(1)
        T['orpos'] = (px - orl[k]) / (orh[k] - orl[k]); T['orbrk'] = np.where(px > orh[k], 1.0, np.where(px < orl[k], -1.0, 0.0))
        sd30 = pd.DataFrame(S.C).T.rolling(30, min_periods=10).std().T.values
        T['vwz'] = (px - vw[k, p]) / sd30[k, p]
    dates = pd.DatetimeIndex(D.date.values)
    tf_ = np.array([(F.third_friday(d.year, d.month) - d).days for d in dates], float)
    nxt = np.array([(F.third_friday(d.year + (d.month == 12), d.month % 12 + 1) - d).days for d in dates], float)
    T['opex'] = np.where(tf_ >= 0, tf_, nxt)[k]; T['mend'] = F.mend_flags(dates).astype(float)[k]
    return T


def build():
    t0 = time.time(); L2, ML, MR, IL = TR.research()
    for a, b, tag in (('nq', 'mnqx', 'nq'), ('es', 'mes', 'es')):
        parts = []
        for nm, sel, off in ((a, lambda d: d <= X.JOIN, 0), (b, lambda d: d > X.JOIN, 10 ** 6)):
            S = Night(L2, nm); T = features(S); T = T[sel(T.date)].assign(k=lambda d: d.k + off); parts.append(T)
            print(f"{nm}: {S.D.shape[0]:,} night sessions, {len(T):,} decision rows, {time.time() - t0:.0f} s", flush=True)
        T = pd.concat(parts, ignore_index=True).sort_values(['k', 'dec']).reset_index(drop=True); T.to_parquet(C + tag + '.parquet')
        Xm = T[NFEATS].values
        for nm_, y in (('f5', T.y.values), ('f15', T.y15.values)):
            pred = X.walk(T, y, MR.model, X=Xm); np.save(C + f'{tag}_{nm_}.npy', pred); print(f"{tag} {nm_}: done, {time.time() - t0:.0f} s", flush=True)
    print('done', flush=True)


def tstat(s):
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std() > 0 else 0.0


def run():
    t0 = time.time(); import e5b_lab as L
    L.setup(); day = L.sim(L.G.model, L.G.band)
    res = {}
    for tag in ('nq', 'es'):
        T = pd.read_parquet(C + tag + '.parquet'); bk = X.Book(T); f5 = np.load(C + f'{tag}_f5.npy'); f15 = np.load(C + f'{tag}_f15.npy')
        f5 = np.where(T.has_bar.values, f5, np.nan); dec = T.dec.values
        win = {'NL1': (30, 920), 'NL2': (540, 920), 'NL3': (30, 535), 'NL4': (30, 920)}
        for nm, (a, b) in win.items():
            inw = (dec >= a) & (dec <= b)
            if nm == 'NL4':
                rule = (lambda inw, g: lambda pos, f, i: (1 if f > X.T0 else (-1 if f < -X.T0 else 0)) * (inw[i] and g[i] == g[i] and g[i] * f > 0))(inw, f15)
            else:
                rule = (lambda inw: lambda pos, f, i: (1 if f > X.T0 else (-1 if f < -X.T0 else 0)) * inw[i])(inw)
            res[(tag, nm)] = {t: bk.run(f5, rule, t) for t in (0.0, 2.35, 3.5, 5.0)}
    say(); say("=" * 160); say(f"E5T FAMILY NL  e5t_night.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  model leg on the Globex night session; $ per micro at today's price; costs on every side"); say("=" * 160)
    T = pd.read_parquet(C + 'nq.parquet', columns=['date', 'k']); say(f"night sessions: {T.k.nunique():,} ({pd.Timestamp(T.date.min()):%Y-%m-%d} .. {pd.Timestamp(T.date.max()):%Y-%m-%d}); out of sample from 2013")
    for nm, lab in (('NL1', 'whole night 18:30-09:20'), ('NL2', 'Europe + pre-market 03:00-09:20'), ('NL3', 'Asia 18:30-02:55'), ('NL4', 'whole night with its 15-minute check')):
        s, n = res[('nq', nm)][2.35]; z, _ = res[('nq', nm)][0.0]; s35, _ = res[('nq', nm)][3.5]; s5, _ = res[('nq', nm)][5.0]; e, _ = res[('es', nm)][2.35]
        A = s[s.index <= X.JOIN]; B = s[s.index > X.JOIN]; yrs = s.groupby(s.index.year).sum(); top = s.sort_values(ascending=False)
        dA = A.drop(A.sort_values(ascending=False).index[:5]); eA, eB = e[e.index <= X.JOIN], e[e.index > X.JOIN]
        j = pd.concat([day.rename('day'), s.rename('night')], axis=1, join='inner'); jA, jB = j[j.index <= X.JOIN], j[j.index > X.JOIN]
        ok = {'NQ 2013-19 t >= 2.6': tstat(A) >= 2.6, 'MNQ 2019-26 t >= 2.0 same sign': tstat(B) >= 2.0, 'at most 1 losing year': int((yrs < 0).sum()) <= 1,
              'positive without its 5 best days (NQ 2013-19)': dA.sum() > 0, 'positive at 3.5 ticks': s35.sum() > 0, 'ES and MES same sign': eA.sum() > 0 and eB.sum() > 0,
              'book Sharpe up in both halves': X.sharpe(jA.day + jA.night) > X.sharpe(jA.day) and X.sharpe(jB.day + jB.night) > X.sharpe(jB.day)}
        say(); say(f"{nm} {lab}: {n / (len(s) / 252):,.0f} round trips a year")
        say(f"   NQ 2013-19: ${A.mean() * 252:+,.0f}/yr, t {tstat(A):+.2f} | MNQ 2019-26: ${B.mean() * 252:+,.0f}/yr, t {tstat(B):+.2f} | all: Sharpe {X.sharpe(s):.2f}, max DD ${L.mdd(s.values):,.0f}")
        say(f"   commission only (no spread): ${z.mean() * 252:+,.0f}/yr t {tstat(z):+.2f} | 3.5 ticks: ${s35.mean() * 252:+,.0f}/yr | 5 ticks: ${s5.mean() * 252:+,.0f}/yr | "
            f"ES 2013-19 ${eA.mean() * 252:+,.0f}/yr t {tstat(eA):+.2f}, MES 2019-26 ${eB.mean() * 252:+,.0f}/yr t {tstat(eB):+.2f}")
        say("   by year: " + ", ".join(f"{y} {v:+,.0f}" for y, v in yrs.items()) + f" | without its 5 best days (NQ 2013-19) ${dA.sum():+,.0f}")
        say(f"   book + night: Sharpe {X.sharpe(jA.day + jA.night):.2f} / {X.sharpe(jB.day + jB.night):.2f} vs day book {X.sharpe(jA.day):.2f} / {X.sharpe(jB.day):.2f}; correlation with the day book {j.day.corr(j.night):+.2f}")
        fails = [k for k, v in ok.items() if not v]
        say(f"   VERDICT {nm}: " + ('FORWARD-TEST CANDIDATE' if not fails else 'dead - fails: ' + '; '.join(fails)))
    say(f"run {time.time() - t0:.0f} s")
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build': build()
    elif a == 'run': run()
    else: print(__doc__)
