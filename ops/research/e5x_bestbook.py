import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_train as TR

LINES = []
PATHS = 20000


def say(s=''):
    print(s, flush=True); LINES.append(s)


def mdd(x):
    eq = np.cumsum(x); return float((np.maximum(np.maximum.accumulate(eq), 0) - eq).max())


def main():
    t0 = time.time(); X.setup(); sys.path.insert(0, TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    e, _ = X.BASE[2.35]; f15 = np.load(os.path.join(X.CACHE, 'x5_nq.npy')); e15, _ = X.NQ.run(X.F_NQ, X.r_agree(f15), 2.35)
    x3 = pd.read_csv(os.path.join(X.CACHE, 'x3_daily.csv'), index_col=0, parse_dates=True)
    c = pd.concat([e.rename('e'), e15.rename('e15'), (x3.r_close * X.U).rename('n'), (x3.r_min * X.U).rename('nmin')], axis=1, join='inner')
    A, B = c[c.index <= X.JOIN], c[c.index > X.JOIN]; sh = X.sharpe
    say(); say("=" * 120); say(f"THE BEST COMBINATION  e5x_bestbook.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n + band leg; $ at today's price, real costs, every session counted"); say("=" * 120)
    say(f"{'mix (micros)':44s} Sharpe 2013-19  2019-26  2013-26 |  $/yr  | max drawdown | profit per $1 of max drawdown a year")
    mixes = [('E5n alone', 'e', 1, 0), ('band leg alone', 'e', 0, 1), ('1 E5n + 1 band leg', 'e', 1, 1), ('1 E5n + 2 band leg', 'e', 1, 2), ('2 E5n + 1 band leg', 'e', 2, 1),
             ('1 E5n with 15-minute check + 1 band leg', 'e15', 1, 1), ('1 E5n with 15-minute check + 2 band leg', 'e15', 1, 2)]
    best = None
    for lab, col, we, wn in mixes:
        s = we * c[col] + wn * c.n; yr = s.mean() * 252; d = mdd(s.values)
        say(f"{lab:44s}        {sh(we * A[col] + wn * A.n):5.2f}    {sh(we * B[col] + wn * B.n):5.2f}    {sh(s):5.2f} | {yr:6,.0f} | {d:9,.0f}    | {yr / d:5.2f}")
        if we and wn and (best is None or sh(we * A[col] + wn * A.n) > best[0]): best = (sh(we * A[col] + wn * A.n), lab, col, we, wn)
    _, lab, col, we, wn = best
    say(f"BEST MIX by the 2013-19 Sharpe (chosen without looking at 2019-26): {lab}")
    n = len(B); mu = B[col].mean(); rng = np.random.default_rng(20261115)
    def idx(h):
        st = rng.integers(0, n - 21, (PATHS, -(-h // 21))); return (st[:, :, None] + np.arange(21)).reshape(PATHS, -1)[:, :h]
    I = {h: idx(h) for h in (21, 63, 126, 252)}
    say(); say(f"PREDICTIONS for {lab} (bootstrap of the 2019-26 sessions; E5n's edge FULL / HALF / NONE, band leg as backtested)")
    for sc, s_ in (('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0)):
        u = (we * (B[col].values - s_ * mu) + wn * B.n.values)
        for h, nm in ((21, '1 month '), (63, '3 months'), (126, '6 months'), (252, '1 year  ')):
            y = u[I[h]]; tot = y.sum(1); eq = y.cumsum(1); dd = (np.maximum.accumulate(np.maximum(eq, 0), 1) - eq).max(1)
            say(f"   {sc:4s} {nm}: profit in {np.mean(tot > 0):4.0%} | median ${np.median(tot):+8,.0f} | 5-95% ${np.percentile(tot, 5):+8,.0f} .. ${np.percentile(tot, 95):+8,.0f} | "
                f"drawdown typical ${np.median(dd):6,.0f}, bad case ${np.percentile(dd, 95):6,.0f}, above $2,500 in {np.mean(dd > 2500):4.0%}")
    say(); say(f"BULENOX 50K for {lab} (E5n at close marks = optimistic; band leg with its intraday worst)")
    a = BR.SIZES['50K']
    for sc, s_ in (('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0)):
        ee = B[col].values - s_ * mu; u = we * ee + wn * B.n.values; w = we * ee + wn * B.nmin.values; nt = (u != 0).astype(float); parts = []
        for q in (1, 2, 3):
            ev = BL.eval_stats(u, w, nt, q, BL.SEED_REPORT, a['dd'], a['dll'], a['target'], a['qual'], resets=True); parts.append(f"x{q}: {ev['p'] * 100:.0f}% a try, ${ev['fees']:,.0f} fees a pass")
        f = BL.funded_stats(BL.master_qm, u, w, nt, a, 1, BL.SEED_REPORT)
        say(f"   {sc:4s} evaluation at " + " | ".join(parts) + f" || funded at x1: 2-yr payouts ${f['pay']:,.0f}, any payout {f['anyp'] * 100:.0f}%, open after 1 yr {f['open1'] * 100:.0f}% / 2 yrs {f['open2'] * 100:.0f}%")
    un = B.n.values; ev = BL.eval_stats(un, B.nmin.values, (un != 0).astype(float), 6, BL.SEED_REPORT, a['dd'], a['dll'], a['target'], a['qual'], resets=True)
    f = BL.funded_stats(BL.master_qm, un, B.nmin.values, (un != 0).astype(float), a, 1, BL.SEED_REPORT)
    say(f"   for comparison, band leg alone: evaluation at 6 micros {ev['p'] * 100:.0f}% a try, ${ev['fees']:,.0f} fees a pass || funded at 1 micro: 2-yr payouts ${f['pay']:,.0f}, open after 1 yr {f['open1'] * 100:.0f}%")
    say(f"run {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
    with open(X.OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")
