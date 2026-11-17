import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_train as TR


def cached(name, fn):
    p = os.path.join(X.CACHE, name + '.npy')
    if os.path.exists(p): return np.load(p)
    v = fn(); np.save(p, v); return v


def main():
    t0 = time.time(); X.setup(); L2, ML, MR, IL = TR.research()
    X.say(); X.say("=" * 120); X.say(f"e5x_models.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  model trials X5-X7 (walk-forward, refit each January)"); X.say("=" * 120)
    surv = []
    for tag, bk, fb in (('nq', X.NQ, X.F_NQ), ('es', X.ES, X.F_ES)):
        T = bk.T; v = T.vol30.values
        f15 = cached(f'x5_{tag}', lambda: X.walk(T, T.y15.values, MR.model)); print(f"{tag} x5 {time.time() - t0:.0f} s", flush=True)
        bag = cached(f'x6_{tag}', lambda: np.nanmean(np.stack([X.walk(T, T.y.values, MR.model, sample=s) for s in (1, 2, 3, 4, 5)]), axis=0))
        print(f"{tag} x6 {time.time() - t0:.0f} s", flush=True)
        with np.errstate(invalid='ignore', divide='ignore'): yz = np.where(np.isfinite(v) & (v > 0), T.y.values / v, np.nan)
        fz = cached(f'x7_{tag}', lambda: X.walk(T, yz, MR.model) * v); print(f"{tag} x7 {time.time() - t0:.0f} s", flush=True)
        if tag == 'nq': N = dict(f15=f15, bag=bag, fz=fz)
        else: E = dict(f15=f15, bag=bag, fz=fz)
    ok = np.isfinite(X.F_NQ) & np.isfinite(N['bag']) & np.isfinite(N['fz']) & np.isfinite(N['f15']); y = X.NQ.T.y.values; y15 = X.NQ.T.y15.values
    ic = lambda a, b: float(np.corrcoef(a[ok], b[ok])[0, 1])
    X.say(f"out-of-sample forecast correlation with the outcome: base {ic(X.F_NQ, y):+.4f} | ensemble {ic(N['bag'], y):+.4f} | vol-scaled {ic(N['fz'], y):+.4f} | "
          f"15-min model vs its own target {ic(N['f15'], y15):+.4f}; base vs ensemble forecasts {ic(X.F_NQ, N['bag']):.3f}")
    if X.judge('X5', '15-minute agreement: the base signal needs the same sign from a 15-minute model',
               lambda bk, t: X.NQ.run(X.F_NQ, X.r_agree(N['f15']), t), lambda bk, t: X.ES.run(X.F_ES, X.r_agree(E['f15']), t))[0]: surv.append('X5')
    if X.judge('X6', 'ensemble: mean of 5 models trained on bootstrap samples of the training sessions',
               lambda bk, t: X.NQ.run(N['bag'], X.r_base(), t), lambda bk, t: X.ES.run(E['bag'], X.r_base(), t))[0]: surv.append('X6')
    if X.judge('X7', 'volatility-scaled target: the model learns return / 30-minute volatility',
               lambda bk, t: X.NQ.run(N['fz'], X.r_base(), t), lambda bk, t: X.ES.run(E['fz'], X.r_base(), t))[0]: surv.append('X7')
    X.say(); X.say("MODEL TRIALS THAT IMPROVE: " + (", ".join(surv) if surv else "none")); X.say(f"run {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
    with open(X.OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(X.LINES) + "\n")
