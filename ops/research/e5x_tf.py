import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X
import e5x_dd as DD
import e5n_train as TR
import e5n_features as F

DAILY = ['d_rsi', 'd_s20', 'd_s50', 'd_s200', 'd_r5', 'd_r20']
HTF = [f'{i}65' for i in F.IND] + DAILY


def extras(L2, nm):
    R = TR.Raw(L2, nm); T = pd.read_parquet(os.path.join(X.CACHE, f'{nm}.parquet'), columns=['k', 'dec', 'date'])
    k = T.k.values; p_ = T.dec.values; out = pd.DataFrame({'k': k, 'dec': p_, 'date': T.date.values})
    for h in (30, 60): out[f'y{h}'] = np.log(R.C[k, np.minimum(p_ + h - 1, 389)] / R.O[k, p_])
    nb, o, h_, l, c, v = F.tf_bars(R, 65); ind = F.indicators(o, h_, l, c, v); j = k * nb + (p_ // 65) - 1; ok = j >= 0; j = np.clip(j, 0, None)
    for i in F.IND: out[f'{i}65'] = np.where(ok, ind[i][j], np.nan)
    dc = pd.Series(R.C[:, -1]); dh = R.H.max(1); dl = R.L.min(1); pc = dc.shift(1).values
    tr = np.maximum(dh - dl, np.maximum(np.abs(dh - pc), np.abs(dl - pc))); datr = pd.Series(tr).shift(1).rolling(14, min_periods=10).mean().values
    prev = dc.shift(1); d = dc.diff(); up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).shift(1).values
    with np.errstate(invalid='ignore', divide='ignore'):
        day = {'d_rsi': rsi, 'd_s20': (prev - dc.shift(1).rolling(20).mean()).values / datr, 'd_s50': (prev - dc.shift(1).rolling(50).mean()).values / datr,
               'd_s200': (prev - dc.shift(1).rolling(200).mean()).values / datr, 'd_r5': np.log(prev / dc.shift(6)).values, 'd_r20': np.log(prev / dc.shift(21)).values}
    for nm_, a in day.items(): out[nm_] = a[k]
    return out


def joined(L2, a, b):
    A = extras(L2, a); A = A[A.date <= X.JOIN]; B = extras(L2, b); B = B[B.date > X.JOIN].assign(k=lambda d: d.k + 10 ** 6)
    return pd.concat([A, B], ignore_index=True).drop(columns='date')


def r_conf(conf, thr=X.T0):
    def rule(pos, f, i):
        t = 1 if f > thr else (-1 if f < -thr else 0)
        return t if t != 0 and conf[i] * t > 0 else 0
    return rule


def r_two(f2, f3, thr=X.T0):
    def rule(pos, f, i):
        t = 1 if f > thr else (-1 if f < -thr else 0)
        return t if t != 0 and f2[i] * t > 0 and f3[i] * t > 0 else 0
    return rule


def cached(name, fn):
    p = os.path.join(X.CACHE, name + '.npy')
    if os.path.exists(p): return np.load(p)
    v = fn(); np.save(p, v); return v


def main():
    t0 = time.time(); X.setup(); X.P_BAR = 0.05 / 16; L2, ML, MR, IL = TR.research()
    X.say(); X.say("=" * 120); X.say(f"HIGHER TIMEFRAMES AND CONFIRMATIONS  e5x_tf.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5X_PLAN amendment E; bar P < {X.P_BAR:.4f})"); X.say("=" * 120)
    S = {}
    for tag, bk, pair in (('nq', X.NQ, ('nq', 'mnqx')), ('es', X.ES, ('es', 'mes'))):
        E = bk.T[['k', 'dec']].merge(joined(L2, *pair), on=['k', 'dec'], how='left'); T = bk.T
        assert len(E) == len(T) and (E.k.values == T.k.values).all()
        f15 = np.load(os.path.join(X.CACHE, f'x5_{tag}.npy'))
        f30 = cached(f'c5_{tag}', lambda: X.walk(T, E.y30.values, MR.model)); f60 = cached(f'c4_{tag}', lambda: X.walk(T, E.y60.values, MR.model))
        Xh = np.c_[T[F.FEATS].values, E[HTF].values]
        fm = cached(f'm1_{tag}', lambda: X.walk(T, T.y.values, MR.model, X=Xh))
        S[tag] = dict(e50=np.nan_to_num(T['e5030'].values), day=np.nan_to_num(E.d_s20.values), vw=np.nan_to_num(T.vwapd.values),
                      f15=np.nan_to_num(f15), f30=np.nan_to_num(f30), f60=np.nan_to_num(f60), fm=fm)
        print(f"{tag} prepared {time.time() - t0:.0f} s", flush=True)
    N, E_ = S['nq'], S['es']; base, _ = X.BASE[2.35]; med, bad, p, _ = DD.boot_dd(base)
    X.say(f"BASE 1-year drawdown: typical ${med:,.0f} / bad case ${bad:,.0f} / chance of more than $2,500 {p:.0%}")
    trials = [('C1', '30-minute trend: price on the signal side of the 30-minute EMA(50)', lambda d: r_conf(d['e50'])),
              ('C2', "daily trend: yesterday's close on the signal side of its 20-session average", lambda d: r_conf(d['day'])),
              ('C3', 'session trend: price on the signal side of VWAP', lambda d: r_conf(d['vw'])),
              ('C4', '60-minute model agrees', lambda d: r_conf(d['f60'])),
              ('C5', '30-minute model agrees', lambda d: r_conf(d['f30'])),
              ('C6', '15- and 60-minute models both agree', lambda d: r_two(d['f15'], d['f60']))]
    surv = []
    for name, desc, mk in trials:
        ok, x = X.judge(name, desc, lambda bk, t, mk=mk: X.NQ.run(X.F_NQ, mk(N), t), lambda bk, t, mk=mk: X.ES.run(X.F_ES, mk(E_), t))
        med, bad, p, _ = DD.boot_dd(x); X.say(f"   1-year drawdown: typical ${med:,.0f} / bad case ${bad:,.0f} / chance of more than $2,500 {p:.0%}")
        if ok: surv.append(name)
    ok, x = X.judge('M1', 'model retrained with 65-minute and daily indicators added (108 features)',
                    lambda bk, t: X.NQ.run(N['fm'], X.r_base(), t), lambda bk, t: X.ES.run(E_['fm'], X.r_base(), t))
    med, bad, p, _ = DD.boot_dd(x); X.say(f"   1-year drawdown: typical ${med:,.0f} / bad case ${bad:,.0f} / chance of more than $2,500 {p:.0%}")
    if ok: surv.append('M1')
    X.say(); X.say("HIGHER-TIMEFRAME TRIALS THAT IMPROVE: " + (", ".join(surv) if surv else "none")); X.say(f"run {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
    with open(X.OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(X.LINES) + "\n")
