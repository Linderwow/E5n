r"""Account sizing for the two-leg book on a Bulenox 50K (rules as stored in the research model; information for the user's sizing decision).
    python e5g_account.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import G, say
X = L.X; U = L.U


def days(cm, cb, net=None):
    net = cm * G.model + cb * G.band if net is None else net; c1 = X.SIDE(2.35) * U
    tr = np.abs(np.diff(np.c_[np.zeros(G.n), net], axis=1))
    cum = np.cumsum(net * G.dpx, axis=1) / G.o930[:, None] * U - c1 * np.cumsum(tr, axis=1)
    close = cum[:, -1] - c1 * np.abs(net[:, -1]); worst = np.minimum(np.minimum(cum.min(axis=1), close), 0.0)
    sel = G.keep & ~G.isA
    return close[sel], worst[sel]


def main():
    t0 = time.time(); L.setup(); sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    a = BR.SIZES['50K']; mu = days(1, 0)[0].mean()
    say(); say("=" * 160)
    say(f"ACCOUNT SIZING  e5g_account.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  Bulenox 50K (trailing drawdown ${a['dd']:,}, daily loss limit ${a['dll']:,}, target ${a['target']:,}, evaluation fee ${a['qual']}, "
        f"activation ${a['act']}); bootstrap of the 2019-26 sessions; worst points marked every 5 minutes (a little optimistic); $ at today's price, live cost on net orders")
    say("=" * 160)
    for sc, s_ in (('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0)):
        say(); say(f"model-leg edge {sc}: EVALUATION pass % a try (fees per pass, with resets) by micros - rows = model leg, columns = band leg")
        say("   model\\band " + "".join(f"{cb:>16d}" for cb in range(0, 7)))
        best = None
        for cm in range(0, 7):
            row = f"   {cm:>10d} "
            for cb in range(0, 7):
                if cm + cb == 0: row += f"{'-':>16s}"; continue
                u, w = days(cm, cb); u = u - s_ * cm * mu; w = w - s_ * cm * mu * (w < 0)
                e = BL.eval_stats(u, w, (u != 0).astype(float), 1, BL.SEED_REPORT, a['dd'], a['dll'], a['target'], a['qual'], resets=True)
                row += f"{e['p'] * 100:7.0f}% (${e['fees']:5,.0f})"
                if best is None or e['fees'] < best[0]: best = (e['fees'], cm, cb, e['p'])
            say(row)
        say(f"   cheapest pass: model {best[1]} + band {best[2]} micros: {best[3] * 100:.0f}% a try, ${best[0]:,.0f} fees a pass")
        say(f"model-leg edge {sc}: FUNDED account, 2 years (payouts to you | any payout | open after 1 yr / 2 yrs)")
        for cm, cb in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 3), (0, 1), (0, 2), (1, 0)):
            u, w = days(cm, cb); u = u - s_ * cm * mu; w = w - s_ * cm * mu * (w < 0)
            f = BL.funded_stats(BL.master_qm, u, w, (u != 0).astype(float), a, 1, BL.SEED_REPORT)
            say(f"   model {cm} + band {cb}: ${f['pay']:7,.0f} | {f['anyp'] * 100:3.0f}% | {f['open1'] * 100:3.0f}% / {f['open2'] * 100:3.0f}%")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
