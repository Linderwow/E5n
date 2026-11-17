r"""Fair expectations for the two-leg book: the same rules with the model leg on every clock offset and both model sets (information).
    python e5k_fair.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import G, say
X = L.X; U = L.U; PATHS = 20000


def day_arrays(net):
    M = C.M; c1 = X.SIDE(2.35) * U; tr = np.abs(np.diff(np.c_[np.zeros(M.n), net], axis=1))
    cum = np.cumsum(net * M.dpx, axis=1) / M.o930[:, None] * U - c1 * np.cumsum(tr, axis=1)
    close = cum[:, -1] - c1 * np.abs(net[:, -1]); worst = np.minimum(np.minimum(cum.min(axis=1), close), 0.0)
    sel = M.keep & ~G.isA
    return close[sel], worst[sel]


def year(u, rng):
    n = len(u); st = rng.integers(0, n - 21, (PATHS, 12)); y = u[(st[:, :, None] + np.arange(21)).reshape(PATHS, -1)]
    tot = y.sum(1); eq = y.cumsum(1); dd = (np.maximum.accumulate(np.maximum(eq, 0), 1) - eq).max(1)
    return np.median(tot), np.percentile(tot, 5), np.mean(tot > 0), np.median(dd), np.percentile(dd, 95), np.mean(dd > 2500)


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    a = BR.SIZES['50K']
    say(); say("=" * 160)
    say(f"FAIR EXPECTATIONS  e5k_fair.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  the book (1 micro a leg) with the model leg on each clock offset and model set; bootstrap of the 2019-26 sessions, 1 year; $ at today's price")
    say("=" * 160)
    rows = []
    for lab, f5, f15 in (('deployed models', M.g5, M.g15), ('retrained on every minute', M.a5, M.a15)):
        for o in range(5):
            mp = C.path(f5, f15, 'set', (M.dec % 5 == o) & (M.dec >= 15)); u, w = day_arrays(mp + M.band); r = year(u, np.random.default_rng(20261115))
            rows.append((lab, o, r, mp))
            say(f"   {lab:26s} offset {o}: median ${r[0]:+8,.0f} | 5% case ${r[1]:+8,.0f} | profitable year {r[2]:4.0%} | max drawdown typical ${r[3]:6,.0f}, bad case ${r[4]:6,.0f}, above $2,500 in {r[5]:4.0%}")
    dep = rows[0][2]; oth = np.array([r[2] for r in rows[1:]]).mean(0)
    say(f"   AS BACKTESTED (deployed models, own clock): median ${dep[0]:+,.0f}, max drawdown typical ${dep[3]:,.0f} / bad case ${dep[4]:,.0f}, above $2,500 in {dep[5]:.0%}")
    say(f"   FAIR (average of the other nine): median ${oth[0]:+,.0f}, 5% case ${oth[1]:+,.0f}, profitable year {oth[2]:.0%}, max drawdown typical ${oth[3]:,.0f} / bad case ${oth[4]:,.0f}, above $2,500 in {oth[5]:.0%}")
    bu, bw = day_arrays(M.band * 1.0); rb = year(bu, np.random.default_rng(20261115))
    say(f"   band leg alone for comparison: median ${rb[0]:+,.0f}, max drawdown typical ${rb[3]:,.0f} / bad case ${rb[4]:,.0f}, above $2,500 in {rb[5]:.0%}")
    say(); say("Bulenox 50K with the FAIR model leg (retrained models on the deployed clock; between the HALF and FULL cases of the sizing table):")
    fair = rows[5][3]
    for cm, cb in ((0, 6), (1, 5), (1, 6), (2, 4), (3, 3)):
        u, w = day_arrays(cm * fair + cb * M.band); e = BL.eval_stats(u, w, (u != 0).astype(float), 1, BL.SEED_REPORT, a['dd'], a['dll'], a['target'], a['qual'], resets=True)
        say(f"   evaluation, model {cm} + band {cb}: {e['p'] * 100:.0f}% a try, ${e['fees']:,.0f} fees a pass")
    for cm, cb in ((0, 1), (1, 1), (0, 2), (1, 2)):
        u, w = day_arrays(cm * fair + cb * M.band); f = BL.funded_stats(BL.master_qm, u, w, (u != 0).astype(float), a, 1, BL.SEED_REPORT)
        say(f"   funded, model {cm} + band {cb}: 2-year payouts ${f['pay']:,.0f} | any payout {f['anyp'] * 100:.0f}% | open after 1 yr {f['open1'] * 100:.0f}% / 2 yrs {f['open2'] * 100:.0f}%")
    say(); say("FORWARD-LEDGER LINES (1 micro a leg, cumulative $ after N sessions; bootstrap of 2019-26; 'fair' = the nine other clock / model combinations pooled):")
    rng = np.random.default_rng(20261116)

    def tot(u, h):
        n = len(u); st = rng.integers(0, n - 21, (PATHS // 4, -(-h // 21))); return u[(st[:, :, None] + np.arange(21)).reshape(PATHS // 4, -1)[:, :h]].sum(1)

    alt = [day_arrays(r[3] + M.band)[0] for r in rows[1:]]; altm = [day_arrays(r[3])[0] for r in rows[1:]]; bandu = day_arrays(M.band * 1.0)[0]
    for h in (21, 60, 126, 252):
        fb = np.concatenate([tot(u, h) for u in alt]); fm = np.concatenate([tot(u, h) for u in altm]); zm = np.concatenate([tot(u - u.mean(), h) for u in altm]); bo = tot(bandu, h)
        say(f"   after {h:3d} sessions | BOOK fair: 5% ${np.percentile(fb, 5):+8,.0f}, median ${np.median(fb):+8,.0f}, 95% ${np.percentile(fb, 95):+8,.0f} | "
            f"MODEL LEG (with check) fair: 5% ${np.percentile(fm, 5):+7,.0f}, median ${np.median(fm):+7,.0f}; with no edge: 5% ${np.percentile(zm, 5):+7,.0f}, 95% ${np.percentile(zm, 95):+7,.0f} | "
            f"BAND LEG: 5% ${np.percentile(bo, 5):+7,.0f}, median ${np.median(bo):+7,.0f}")
    say("   reading: a leg below its fair 5% line is doing worse than 19 of 20 fair outcomes; the model leg above the no-edge 95% line has an edge that chance alone rarely produces.")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
