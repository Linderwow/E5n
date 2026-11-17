import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X

OUT = os.path.join(HERE, 'E5X_RESULTS.txt'); LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def run(bk, f, rule, day_stop=None, max_trades=None, week_stop=None, ticks=2.35):
    c1 = X.SIDE(ticks); n = len(bk.st); net = np.zeros(n); trd = np.zeros(n); entry = bk.entry
    wk = bk.dates.isocalendar(); wkey = (wk.year.values * 100 + wk.week.values); cur = -1; wsum = 0.0
    for d in range(n):
        if wkey[d] != cur: cur = wkey[d]; wsum = 0.0
        if week_stop is not None and wsum * X.U <= -week_stop: continue
        a, b = bk.st[d], bk.en[d]; o = bk.o930[a]; pos = 0; pnl = 0.0; sides = 0; last = 0.0; halted = False; rts = 0
        for i in range(a, b):
            fi = f[i]
            if fi != fi: continue
            if pos != 0: pnl += pos * (entry[i] - last); last = entry[i]
            if day_stop is not None and not halted and (pnl / o - sides * c1) * X.U <= -day_stop: halted = True
            new = 0 if halted else rule(pos, fi, i)
            if max_trades is not None and new != pos and new != 0 and rts >= max_trades: new = 0
            if new != pos:
                if new != 0: rts += 1
                sides += abs(new - pos); pos = new; last = entry[i]
        if pos != 0: pnl += pos * (bk.exitp[b - 1] - last); sides += abs(pos)
        net[d] = pnl / o - sides * c1; trd[d] = sides / 2; wsum += net[d]
    return pd.Series(net[bk.keep] * X.U, index=bk.dates[bk.keep]), float(trd[bk.keep].sum())


def mdd(x):
    eq = x.cumsum(); return float((np.maximum(eq.cummax(), 0) - eq).max())


def boot_dd(x, rng_seed=7):
    v = x[x.index > X.JOIN].values; rng = np.random.default_rng(rng_seed); st = rng.integers(0, len(v) - 21, (20000, 12))
    y = v[(st[:, :, None] + np.arange(21)).reshape(20000, -1)]; eq = y.cumsum(1); dd = (np.maximum.accumulate(np.maximum(eq, 0), 1) - eq).max(1)
    return float(np.median(dd)), float(np.percentile(dd, 95)), float(np.mean(dd > 2500)), float(np.median(y.sum(1)))


def main():
    t0 = time.time(); X.setup(); f, fe = X.F_NQ, X.F_ES; rb = X.r_base()
    base, nb = run(X.NQ, f, rb); chk, _ = X.BASE[2.35]; assert abs(base.sum() - chk.sum()) < 1e-6
    eb, _ = run(X.ES, fe, rb)
    A = lambda s: s[s.index <= X.JOIN]; B = lambda s: s[s.index > X.JOIN]
    say(); say("=" * 120); say(f"DRAWDOWN CONTROLS  e5x_dd.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5X_PLAN amendment D)  1 micro, today's price, 2.35 ticks + $0.61"); say("=" * 120)
    eq = base.cumsum(); dd = eq.cummax() - eq; worst = dd.sort_values(ascending=False)
    say("where the base's drawdowns come from (today's price):")
    seen = []
    for end in worst.index:
        if any(abs((end - s).days) < 120 for s in seen): continue
        seen.append(end); start = eq[:end].idxmax(); w = base[start:end]; w = w[1:]
        say(f"   {start.date()} -> {end.date()}: -${dd[end]:,.0f} over {len(w)} sessions; losing days {int((w < 0).sum())}, the 3 worst days = ${w.sort_values().head(3).sum():,.0f} "
            f"({w.sort_values().head(3).sum() / -dd[end]:.0%} of it); days without a trade {int((w == 0).sum())}")
        if len(seen) == 4: break
    yr = len(base) / 252; b_med, b_bad, b_p, b_y = boot_dd(base)
    hdr = f"{'':46s} $/yr   Sharpe  maxDD A  maxDD B  worst day | 1-year paths: typical DD / bad case / chance DD > $2,500 | ES/MES maxDD"
    say(); say(hdr)
    def row(lab, x, e, n_):
        med, bad, p, _ = boot_dd(x)
        say(f"{lab:46s} {x.sum() / yr:6,.0f}  {X.sharpe(x):5.2f}  {mdd(A(x)):7,.0f}  {mdd(B(x)):7,.0f}  {x.min():8,.0f} | {med:7,.0f} / {bad:6,.0f} / {p:4.0%}"
            f"{'':22s} | {mdd(e):,.0f}")
        return med, bad, p
    row('BASE E5n', base, eb, nb)
    f15 = np.load(os.path.join(X.CACHE, 'x5_nq.npy')); f15e = np.load(os.path.join(X.CACHE, 'x5_es.npy'))
    trials = [('D1a daily stop $400', dict(day_stop=400), rb, rb), ('D1b daily stop $700', dict(day_stop=700), rb, rb), ('D1c daily stop $1,000', dict(day_stop=1000), rb, rb),
              ('D2a at most 4 trades a day', dict(max_trades=4), rb, rb), ('D2b at most 6 trades a day', dict(max_trades=6), rb, rb),
              ('D3a weekly stop $1,000', dict(week_stop=1000), rb, rb), ('D3b weekly stop $1,500', dict(week_stop=1500), rb, rb),
              ('X5 15-minute agreement (shown again)', {}, X.r_agree(f15), X.r_agree(f15e)),
              ('X3 high-volatility gate (shown again)', {}, X.r_gate(X.NQ.volratio(), 1.0), X.r_gate(X.ES.volratio(), 1.0)),
              ('X5 + D1b (information, not registered)', dict(day_stop=700), X.r_agree(f15), X.r_agree(f15e))]
    helps = []
    for lab, kw, rn, re_ in trials:
        x, n_ = run(X.NQ, f, rn, **kw); e, _ = run(X.ES, fe, re_, **kw)
        med, bad, p = row(lab, x, e, n_)
        h = {'H1': mdd(A(x)) < mdd(A(base)) and mdd(B(x)) < mdd(B(base)), 'H2': bad <= 0.85 * b_bad, 'H3': x.sum() >= 0.8 * base.sum(),
             'H4': X.sharpe(x) >= X.sharpe(base) - 0.05, 'H5': mdd(e) < mdd(eb)}
        fails = [k for k, v in h.items() if not v]
        say(f"{'':6s}-> {'HELPS' if not fails else 'does not help (' + ', '.join(fails) + ')'}")
        if not fails and lab[0] == 'D': helps.append(lab)
    say(); say("CONTROLS THAT HELP: " + (", ".join(helps) if helps else "none")); say(f"run {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")
