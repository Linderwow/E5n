r"""NN (NN_PLAN.txt) step 3: score the networks against the deployed forecast, alone and averaged, both halves.
    python nn_score.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5s_lab as ES
from e5b_lab import G, say
X = L.X


def main():
    t0 = time.time(); L.setup(); L.OUT = os.path.join(HERE, 'NN_RESULTS.txt'); z = np.load(os.path.join(X.PATHS.CACHE, 'nn_preds.npz'))
    y = G.T.y.values.astype(float); f5 = np.asarray(X.F_NQ, float)
    halves = ((pd.Timestamp('2013-01-01'), pd.Timestamp('2019-05-05')), (pd.Timestamp('2019-05-06'), pd.Timestamp('2026-10-15')))
    say(); say("=" * 160); say(f"NN  NEURAL NETWORKS FOR THE MODEL LEG  nn_score.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (NN_PLAN.txt) walk-forward by year, halves 2013-19 / 2019-26"); say("=" * 160)
    base = ES.score(f5, y, halves); res = {'deployed forecast': base}
    for nm in ('mlp', 'hybrid'):
        p = z[nm].astype(float); res[f"{nm} alone"] = ES.score(p, y, halves); res[f"{nm} averaged with deployed"] = ES.score(np.where(np.isfinite(p), (f5 + p) / 2, f5), y, halves)
        ok = np.isfinite(p) & np.isfinite(f5); say(f"   {nm}: correlation of its forecast with the deployed one {np.corrcoef(p[ok], f5[ok])[0, 1]:+.2f}")
    for k, r in res.items():
        say(f"   {k:34s} accuracy {r['h1']['ic']:+.4f} / {r['h2']['ic']:+.4f} | model-leg Sharpe {r['h1']['sh']:.2f} / {r['h2']['sh']:.2f} | model-leg $/yr {r['h1']['yr']:7,.0f} / {r['h2']['yr']:7,.0f} | "
            f"book Sharpe {r['h1']['bsh']:.2f} / {r['h2']['bsh']:.2f}")
    for k, r in res.items():
        if k == 'deployed forecast': continue
        ok = all(r[h]['ic'] > base[h]['ic'] and r[h]['bsh'] >= base[h]['bsh'] for h in ('h1', 'h2'))
        say(f"   -> {k}: {'FORWARD-TEST CANDIDATE' if ok else 'not better than the deployed model in both halves'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
