import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_train as TR

OUT = os.path.join(HERE, 'E5X_RESULTS.txt')
LINES = []
PATHS = 20000


def say(s=''):
    print(s, flush=True); LINES.append(s)


def boot_idx(n_days, n_hist, rng):
    nb = -(-n_days // 21); st = rng.integers(0, n_hist - 21, (PATHS, nb))
    return (st[:, :, None] + np.arange(21)).reshape(PATHS, -1)[:, :n_days]


def main():
    t0 = time.time(); X.setup()
    sys.path.insert(0, TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    e_all, _ = X.BASE[2.35]
    f15 = np.load(os.path.join(X.CACHE, 'x5_nq.npy')); e15_all, _ = X.NQ.run(X.F_NQ, X.r_agree(f15), 2.35)
    x3 = pd.read_csv(os.path.join(X.CACHE, 'x3_daily.csv'), index_col=0, parse_dates=True)
    c = pd.concat([e_all.rename('e'), e15_all.rename('e15'), (x3.r_close * X.U).rename('n'), (x3.r_min * X.U).rename('nmin')], axis=1, join='inner')
    c = c[c.index > X.JOIN]; n = len(c); mu = c.e.mean(); mu15 = c.e15.mean()
    say(); say("=" * 120)
    say(f"PREDICTIONS  e5x_predict.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  block bootstrap of the {n} MNQ sessions 2019-05..2026-10 (21-day blocks, {PATHS:,} paths,")
    say("the same days drawn for every strategy so their co-movement is kept); $ at today's price, 2.35 ticks + $0.61 a side; no prop rules in part 1")
    say("E5n edge scenarios: FULL = as backtested | HALF = half the average day (the usual discount; the 5-model ensemble points there) | NONE = no edge")
    say("=" * 120)
    ports = [('E5n, 1 micro', lambda s: c.e.values - s * mu, True), ('E5n + 15-minute check, 1 micro (near miss, not adopted)', lambda s: c.e15.values - s * mu15, True),
             ('band leg, 1 micro', lambda s: c.n.values, False), ('BOOK E5n + band leg, 1 micro each', lambda s: c.e.values - s * mu + c.n.values, True),
             ('BOOK 3 + 3 micros', lambda s: 3 * (c.e.values - s * mu + c.n.values), True)]
    rng = np.random.default_rng(20261115); IDX = {h: boot_idx(h, n, rng) for h in (21, 63, 252)}
    say("1. WHAT TO EXPECT")
    for lab, fn, has in ports:
        say(f" {lab}")
        for sc, s in ((('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0)) if has else (('as backtested', 0.0),)):
            u = fn(s); parts = []
            for h, nm in ((21, '1 month'), (63, '3 months'), (252, '1 year')):
                y = u[IDX[h]]; tot = y.sum(1); eq = y.cumsum(1); dd = (np.maximum.accumulate(np.maximum(eq, 0), 1) - eq).max(1)
                parts.append(f"{nm}: profit in {np.mean(tot > 0):.0%}, median ${np.median(tot):+,.0f} (5-95% ${np.percentile(tot, 5):+,.0f} .. ${np.percentile(tot, 95):+,.0f}), "
                             f"drawdown median ${np.median(dd):,.0f} / bad case ${np.percentile(dd, 95):,.0f}")
            say(f"   {sc:14s} " + " | ".join(parts))
    say(); say("2. BULENOX 50K (trailing $2,500, daily limit $1,100, target $3,000; evaluation with resets; E5n at close marks = optimistic)")
    a = BR.SIZES['50K']
    def bul(lab, u, w, q):
        nt = (u != 0).astype(float)
        e = BL.eval_stats(u, w, nt, q, BL.SEED_REPORT, a['dd'], a['dll'], a['target'], a['qual'], resets=True)
        f = BL.funded_stats(BL.master_qm, u, w, nt, a, 1, BL.SEED_REPORT)
        say(f"   {lab:44s} evaluation pass {e['p'] * 100:5.1f}% a try (fees per pass ${e['fees']:,.0f}) | funded at 1 unit: 2-yr payouts ${f['pay']:,.0f}, open after 1 yr {f['open1'] * 100:.0f}%")
    for sc, s in (('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0)):
        say(f"  E5n edge {sc}:")
        ee = c.e.values - s * mu
        bul('E5n alone, 3 micros in the evaluation', ee, ee, 3)
        bul('BOOK, 3 + 3 micros in the evaluation', ee + c.n.values, ee + c.nmin.values, 3)
    say("  no E5n at all:")
    bul('band leg alone, 6 micros in the evaluation', c.n.values, c.nmin.values, 6)
    bul('band leg alone, 3 micros in the evaluation', c.n.values, c.nmin.values, 3)
    say(); say("3. THE FORWARD TEST: WHERE THE LEDGER SHOULD BE (E5n, 1 micro, cumulative since 2026-10-16)")
    led = pd.read_csv(os.path.join(X.OPS, 'data', 'e5n_ledger.csv')); obs = float(led.net_usd_1micro.sum()); nd = len(led)
    say(f"   now: {nd} sessions, ${obs:+,.0f}")
    for h in (40, 60, 90, 126):
        idx = boot_idx(h, n, rng); q = {}
        for sc, s in (('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0)): q[sc] = (c.e.values - s * mu)[idx].sum(1)
        say(f"   after {h:3d} sessions: FULL median ${np.median(q['FULL']):+,.0f} (5% ${np.percentile(q['FULL'], 5):+,.0f}) | HALF median ${np.median(q['HALF']):+,.0f} | "
            f"NONE median ${np.median(q['NONE']):+,.0f} (80% ${np.percentile(q['NONE'], 80):+,.0f}, 95% ${np.percentile(q['NONE'], 95):+,.0f})")
    idx = boot_idx(nd, n, rng)
    say("   chance of the present ledger or worse: " + ", ".join(f"{sc} {np.mean((c.e.values - s * mu)[idx].sum(1) <= obs):.1%}" for sc, s in (('FULL', 0.0), ('HALF', 0.5), ('NONE', 1.0))))
    say(f"run {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")
