r"""Optimum funded size for build 6 (band leg only until the lock, then 1 + 1 per unit) - information, 2026-11-17. Same simulator and paths
as e5r_factory.py, fill cost 4.5 ticks a side, fair / no-edge model leg, Bulenox 50K as stored.
    python e5r_size_sweep.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5k_fair as K
import e5x_lab as X
import e5r_factory as F

O = X.SIDE; X.SIDE = lambda t: O(t - 2.35 + 4.5)


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); mu = K.day_arrays(fair)[0].mean()
    def legs(none):
        out = {}
        for nm, cm, cb in (('book', 1, 1), ('band', 0, 1)):
            u, w = K.day_arrays(cm * fair + cb * M.band)
            if none and cm: u = u - cm * mu; w = np.minimum(np.minimum(w - cm * mu * (w < 0), u), 0.0)
            out[nm] = (u, w)
        return out
    SF = F.streams(BR, legs(False), BL.SEED_REPORT); SN = F.streams(BR, legs(True), BL.SEED_REPORT)
    L.OUT = os.path.join(HERE, 'E5R_SIZE_SWEEP.txt')
    L.say(); L.say("=" * 150); L.say(f"FUNDED SIZE SWEEP (build 6: band only until the lock)  e5r_size_sweep.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  4.5 ticks, 2 years, 10,000 paths"); L.say("=" * 150)
    L.say(f"   {'$ a unit':>8s} {'max':>4s} {'micros':>6s} | any payout | open 1 yr / 2 yrs | payouts avg / 25% / median / 75% | worst size used | NO EDGE: avg / median, open 1 yr")
    for per in (2000, 2500, 3000, 3500):
        for mx in (4, 6, 8, 10, 12, 16):
            rl = F.rule(per, mx); r = F.factory(BL, SF, a, 1, 0, 'lock', rl, with_eval=False); tot = r.pay.sum(1)
            n = F.factory(BL, SN, a, 1, 0, 'lock', rl, with_eval=False); tn = n.pay.sum(1)
            L.say(f"   {per:8,d} {mx:4d} {2 * mx:6d} | {np.mean(tot > 0):10.0%} | {np.mean(r.first_lost > 251):5.0%} / {np.mean(r.first_lost > 503):4.0%}  | ${tot.mean():7,.0f} / ${np.percentile(tot, 25):6,.0f} / ${np.median(tot):6,.0f} / ${np.percentile(tot, 75):7,.0f} | "
                  f"{int(r.units.max()):3d} units | ${tn.mean():7,.0f} / ${np.median(tn):6,.0f}, {np.mean(n.first_lost > 251):4.0%}")
    X.SIDE = O; L.say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
