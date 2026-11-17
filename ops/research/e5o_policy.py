r"""Account policies for the two-leg book on a Bulenox 50K (information for the user's sizing decision): a strategy-side daily stop in
the evaluation, and size earned from the cushion once funded. Rules as stored in the research account model.
    python e5o_policy.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5k_fair as K
from e5b_lab import say


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR, masterscale as MS
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15)
    fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0)
    say(); say("=" * 160)
    say(f"ACCOUNT POLICIES  e5o_policy.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  Bulenox 50K; bootstrap of the 2019-26 sessions; worst points marked every minute; $ at today's price; "
        f"FAIR model leg = models retrained on every-minute rows, deployed clock")
    say("=" * 160)
    say("EVALUATION with the strategy's own daily loss stop (the day ends flat at the stop; $1,100 = the account's own limit):")
    for lab, mp in (('fair model leg', fair), ('model leg as backtested', dep)):
        for cm, cb in ((0, 6), (1, 5), (1, 6), (2, 4)):
            if cm == 0 and lab != 'fair model leg': continue
            u, w = K.day_arrays(cm * mp + cb * M.band); nt = (u != 0).astype(float); row = []
            for stop in (1100, 900, 700, 500):
                e = BL.eval_stats(u, w, nt, 1, BL.SEED_REPORT, a['dd'], stop, a['target'], a['qual'], resets=True); row.append(f"stop ${stop:,}: {e['p'] * 100:.0f}% (${e['fees']:,.0f} a pass)")
            say(f"   model {cm} + band {cb} ({lab if cm else 'band leg only'}): " + " | ".join(row))
    say(); say("FUNDED ACCOUNT, 2 years, size earned from the cushion (balance above the drawdown threshold); 1 unit = the micros shown:")
    rules = (('fixed 1 unit', lambda cu: np.full(cu.shape, 1.0)), ('1 unit per $2,500 of cushion, max 2', lambda cu: np.clip(np.floor(cu / 2500), 1, 2)),
             ('1 unit per $2,500 of cushion, max 3', lambda cu: np.clip(np.floor(cu / 2500), 1, 3)), ('1 unit per $2,500 of cushion, max 4', lambda cu: np.clip(np.floor(cu / 2500), 1, 4)))
    for lab, net in (('unit = band 1', 1.0 * M.band), ('unit = fair model 1 + band 1', fair + M.band), ('unit = model 1 as backtested + band 1', dep + M.band)):
        u, w = K.day_arrays(net); nt = (u != 0).astype(float); X_, W_, NT = BL.paths(BL.N_FUNDED, u, w, nt, 1, BL.SEED_REPORT)
        for nm, rule in rules:
            pay, ou, mic = MS.master_dyn(X_, W_, NT, a, rule); s = MS.stats(pay, ou, mic)
            say(f"   {lab:38s} {nm:36s}: payouts ${s['pay']:7,.0f} | any payout {s['any'] * 100:3.0f}% | open after 1 yr {s['open1'] * 100:3.0f}% / 2 yrs {s['open2'] * 100:3.0f}% | average size {s['avgq']:.2f} units")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
