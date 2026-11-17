r"""E5n on a Bulenox 50K (rules as stored in the research account model): evaluation pass chances, time and fees, then the funded
account's payouts and their timing, for three strengths of the model leg (fair, as backtested, no edge). Information for the user.
    python e5p_predict.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5k_fair as K
from e5b_lab import say


def evaluation_t(BL, X, W, dd, dll, target):
    n_p, n_d = X.shape; bal = np.zeros(n_p); hi = np.zeros(n_p); alive = np.ones(n_p, bool); ok = np.zeros(n_p, bool); nres = np.zeros(n_p); when = np.full(n_p, -1)
    for t in range(n_d):
        x, w = BL.day(X[:, t], W[:, t], dll)
        br = alive & (bal + w < hi - dd)
        bal = np.where(alive & ~br, bal + x, bal)
        rs = br & ((n_d - 1 - t) >= BL.RESET_MIN_LEFT); nres += rs; bal[rs] = 0.0; hi[rs] = 0.0; br &= ~rs
        alive &= ~br
        hi = np.where(alive, np.maximum(hi, bal), hi); hit = alive & (bal >= target); when[hit & ~ok] = t + 1; ok |= hit; alive &= ~hit
    return ok, nres, when


def strengths(M, fair, dep):
    mu = K.day_arrays(fair)[0].mean()
    def none(cm, cb):
        u, w = K.day_arrays(cm * fair + cb * M.band); u2 = u - cm * mu; w2 = np.minimum(np.minimum(w - cm * mu * (w < 0), u2), 0.0)
        return u2, w2
    return (('FAIR (expected)', lambda cm, cb: K.day_arrays(cm * fair + cb * M.band)),
            ('AS BACKTESTED (optimistic)', lambda cm, cb: K.day_arrays(cm * dep + cb * M.band)),
            ('MODEL LEG NO EDGE (pessimistic)', none))


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR, masterscale as MS
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15)
    fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0)
    say(); say("=" * 160)
    say(f"ACCOUNT PREDICTIONS  e5p_predict.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  Bulenox 50K as stored (evaluation: ${a['qual']} a month, target ${a['target']:,}, trailing drawdown "
        f"${a['dd']:,} on closes breached intraday, daily loss limit ${a['dll']:,}, resets ${BL.RESET_FEE:.0f} with >= {BL.RESET_MIN_LEFT} sessions left; activation ${a['act']}; funded: "
        f"threshold locks at +$100, payouts after 10 trading days with 40% consistency, ${a['reserve']:,} kept, first 3 capped at ${a['cap']:,} then half the balance, 90% to you, "
        f"$1,000 minimum, 3 sessions' wait); bootstrap of the 2019-26 sessions in 21-session blocks, {BL.N_PATHS:,} paths; $ at today's price, live cost on net orders")
    say("=" * 160)
    S = strengths(M, fair, dep)
    for lab, mk in S:
        say(); say(f"{lab}: EVALUATION (one try = one month of 21 sessions, resets included)")
        say("   micros (model + band) | pass a try | within 3 tries | sessions to pass (median, of passes) | passed within 10 sessions | resets a try | fees a pass (+ activation)")
        for cm, cb in ((0, 6), (1, 5), (1, 6), (2, 4), (3, 3), (1, 1)):
            u, w = mk(cm, cb); nt = (u != 0).astype(float)
            X, W, _ = BL.paths(BL.N_WIN, u, w, nt, 1, BL.SEED_REPORT)
            ok, nres, when = evaluation_t(BL, X, W, a['dd'], a['dll'], a['target'])
            ok0, nres0 = BL.evaluation(X, W, a['dd'], a['dll'], a['target'], resets=True); assert np.array_equal(ok, ok0) and np.array_equal(nres, nres0)
            p = ok.mean(); cost = a['qual'] + BL.RESET_FEE * nres.mean(); fees = cost / p if p > 0 else np.inf
            say(f"   {cm} + {cb}{'':16s} | {p * 100:9.0f}% | {(1 - (1 - p) ** 3) * 100:13.0f}% | {np.median(when[ok]):35.0f} | {np.mean(ok & (when <= 10)) * 100:24.0f}% | {nres.mean():12.2f} | "
                f"${fees:,.0f} (+ ${a['act']} = ${fees + a['act']:,.0f})")
        say(f"{lab}: FUNDED ACCOUNT over 2 years (504 sessions)")
        say("   micros and sizing                     | any payout | first payout (median sessions) | lost before 1st | payouts yr 1 / yr 2 / 2 yrs (average) | 2-yr payouts 25% / median / 75% | open 1 yr / 2 yrs")
        rules = (('band 1, fixed', 0, 1, lambda cu: np.full(cu.shape, 1.0)), ('model 1 + band 1, fixed', 1, 1, lambda cu: np.full(cu.shape, 1.0)),
                 ('model 1 + band 1, cushion max 2', 1, 1, lambda cu: np.clip(np.floor(cu / 2500), 1, 2)),
                 ('model 1 + band 1, cushion max 4', 1, 1, lambda cu: np.clip(np.floor(cu / 2500), 1, 4)))
        for nm, cm, cb, rule in rules:
            u, w = mk(cm, cb); nt = (u != 0).astype(float); X_, W_, NT = BL.paths(BL.N_FUNDED, u, w, nt, 1, BL.SEED_REPORT)
            pay, ou, mic = MS.master_dyn(X_, W_, NT, a, rule); tot = pay.sum(1); got = tot > 0
            fp = np.where(got, (pay > 0).argmax(1) + 1, -1); y1 = pay[:, :252].sum(1); y2 = pay[:, 252:].sum(1)
            q25, q50, q75 = np.percentile(tot, [25, 50, 75])
            say(f"   {nm:37s} | {got.mean() * 100:9.0f}% | {np.median(fp[got]):30.0f} | {np.mean(~got & (ou < BL.N_FUNDED)) * 100:14.0f}% | "
                f"${y1.mean():7,.0f} / ${y2.mean():7,.0f} / ${tot.mean():7,.0f}{'':5s} | ${q25:7,.0f} / ${q50:7,.0f} / ${q75:7,.0f}{'':4s} | {np.mean(ou > 251) * 100:5.0f}% / {np.mean(ou > 503) * 100:4.0f}%")
    say(); say("reading: 'a try' is one month's subscription; within 3 tries = 1 - (1 - p)^3; fees a pass = all monthly fees and resets spent per pass on average;")
    say("first payout counts sessions after funding; the funded rows assume the evaluation is passed; sizes are the user's decision.")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
