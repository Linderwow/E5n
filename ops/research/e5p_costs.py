r"""E5n account predictions (e5p_predict.py, fair model leg, Bulenox 50K as stored) re-run at higher live fill costs - 2026-11-17 after
the first live session's fills cost 7.9 ticks a side (2.35 assumed). Information for the user, not a trial.
    python e5p_costs.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5x_lab as X
import e5p_predict as PR
from e5b_lab import say

ORIG = X.SIDE


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR, masterscale as MS
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0)
    L.OUT = os.path.join(HERE, 'E5P_COSTS.txt')
    say(); say("=" * 150); say(f"ACCOUNT PREDICTIONS AT HIGHER FILL COSTS  e5p_costs.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  fair model leg, Bulenox 50K as stored, same paths as e5p_predict"); say("=" * 150)
    say("   ticks a side | eval 0 + 6: pass a try / 3 tries | eval 1 + 6: pass / 3 tries | funded 1 + 1 cushion max 4: any payout / 2-yr avg / median | fixed 1 + 1: any payout / 2-yr avg / median")
    for tk in (2.35, 3.5, 5.0, 8.0):
        X.SIDE = lambda t, tk=tk: ORIG(t - 2.35 + tk)
        mk = PR.strengths(M, fair, dep)[0][1]; ev = []
        for cm, cb in ((0, 6), (1, 6)):
            u, w = mk(cm, cb); nt = (u != 0).astype(float); X_, W_, _ = BL.paths(BL.N_WIN, u, w, nt, 1, BL.SEED_REPORT)
            ok, _, _ = PR.evaluation_t(BL, X_, W_, a['dd'], a['dll'], a['target']); p = ok.mean(); ev.append(f"{p:5.0%} / {1 - (1 - p) ** 3:4.0%}")
        fu = []
        for rule in (lambda cu: np.clip(np.floor(cu / 2500), 1, 4), lambda cu: np.full(cu.shape, 1.0)):
            u, w = mk(1, 1); nt = (u != 0).astype(float); X_, W_, NT = BL.paths(BL.N_FUNDED, u, w, nt, 1, BL.SEED_REPORT)
            pay, ou, mic = MS.master_dyn(X_, W_, NT, a, rule); tot = pay.sum(1); fu.append(f"{(tot > 0).mean():4.0%} / ${tot.mean():7,.0f} / ${np.median(tot):7,.0f}")
        say(f"   {tk:12.2f} | {ev[0]:32s} | {ev[1]:24s} | {fu[0]:52s} | {fu[1]}")
    X.SIDE = ORIG; say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
