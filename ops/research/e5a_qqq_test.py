r"""E5A A2 (E5A_PLAN.txt): does order flow in QQQ's own trades (free since 2016) add to E5n's 5-minute forecast? Information test.
    python e5a_qqq_test.py        needs cache\e5a_qqq_minutes.parquet (e5a_qqq.py fetch, build)
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5a_qqq as Q

RES = os.path.join(HERE, 'E5A_RESULTS.txt'); LINES = []
MEAS = {'QOF1': (1, 'buy', 'sell'), 'QOF5': (5, 'buy', 'sell'), 'QOF15': (15, 'buy', 'sell'), 'QLT5': (5, 'lbuy', 'lsell'), 'QSW5': (5, 'wbuy', 'wsell'), 'QRT15': (15, 'rbuy', 'rsell')}
TWIN = {'QOF1': (1, 'buy', 'sell'), 'QOF5': (5, 'buy', 'sell'), 'QOF15': (15, 'buy', 'sell'), 'QLT5': (5, 'lbuy', 'lsell'), 'QSW5': (5, 'buy', 'sell'), 'QRT15': (15, 'buy', 'sell')}


def say(s=''):
    print(s, flush=True); LINES.append(s)


def main():
    import e5x_lab as X, e5f_pilot as P
    t0 = time.time(); X.setup(); g = pd.read_parquet(Q.OUT).set_index('m'); g['net'] = g.buy - g.sell; g['ret'] = np.log(g['last']).groupby(g.index.normalize()).diff()
    have = np.unique(g.index.normalize().values.astype('datetime64[D]')); grid = pd.DatetimeIndex(np.load(os.path.join(Q.PATHS.CACHE, 'e5c_dates.npy'), allow_pickle=True))
    smp = grid[(np.arange(len(grid)) % 3 == 0) & (grid >= Q.FIRST) & (grid <= Q.LAST)].values.astype('datetime64[D]')
    say(); say("=" * 160); say(f"E5A A2  YEARS OF ORDER FLOW FROM QQQ'S OWN TRADES  e5a_qqq_test.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5A_PLAN.txt; information test only)"); say("=" * 160)
    T = X.NQ.T; f = X.F_NQ; td = T.date.values.astype('datetime64[D]'); sel = np.isin(td, have); T = T[sel].reset_index(drop=True); f = f[sel]; td = td[sel]; y = T.y.values; day = td.astype(np.int64)
    ends = T.date.values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]'); insmp = np.isin(td, smp); pil = (td >= np.datetime64(Q.PILOT[0])) & (td <= np.datetime64(Q.PILOT[1]))
    ok = np.isfinite(g.ret.values); san = float(np.corrcoef(g.net.values[ok], g.ret.values[ok])[0, 1])
    say(f"QQQ minute files: {len(have)} sessions, {g.n.sum() / 1e6:,.0f} million exchange trades; sampled sessions (every 3rd) with E5n decisions {len(np.unique(td[insmp]))} of {len(smp)} wanted, {int(insmp.sum()):,} decisions; "
        f"sanity: tick-rule buys minus sells vs the same minute's price change, correlation {san:+.2f}")
    x = {k: P.window(g, ends, w, a, b) for k, (w, a, b) in MEAS.items()}
    db = P.minute_flow(P.FILES['MNQ'][0]); cal = []
    for k, (w, a, b) in TWIN.items():
        z = P.window(db, ends, w, a, b); m = pil & np.isfinite(z) & np.isfinite(x[k]); cal.append(f"{k} {np.corrcoef(x[k][m], z[m])[0, 1]:+.2f}")
    say(f"calibration on the {len(np.unique(td[pil]))} pilot sessions - correlation with the MNQ measure from the bought trades (true buyer / seller side): " + ', '.join(cal))
    halves = (('all sampled sessions 2016-2026', insmp), ('2016-2021', insmp & (td < np.datetime64('2022-01-01'))), ('2022-2026', insmp & (td >= np.datetime64('2022-01-01')))); res = {}
    for lab, m in halves:
        say(f"{lab}: {len(np.unique(td[m]))} sessions, {int(m.sum()):,} decisions")
        for k in MEAS:
            tt, n, ic = P.ols_t(y[m], f[m], x[k][m], day[m]); pl = [P.ols_t(y[m], f[m], pd.Series(x[k][m]).shift(75 * s).values, day[m])[0] for s in (1, 2, 3)]; res[(lab, k)] = (tt, pl)
            say(f"   {k:6s} t {tt:+.2f} beyond E5n's forecast (n {n:,}; plain correlation with the next 5 minutes {ic:+.4f}) | placebos (1 / 2 / 3 sampled sessions earlier) t " + ' / '.join(f"{p:+.2f}" for p in pl))
    a, h1, h2 = (h[0] for h in halves); hit = []
    for k in MEAS:
        t_a, t_1, t_2 = res[(a, k)][0], res[(h1, k)][0], res[(h2, k)][0]
        if abs(t_a) >= 3.0 and np.sign(t_1) == np.sign(t_a) == np.sign(t_2) and min(abs(t_1), abs(t_2)) >= 2.0 and all(abs(p) < 2.0 for p in res[(a, k)][1]): hit.append(k)
    say("VERDICT A2: " + (f"PASS - {', '.join(hit)}: download the remaining sessions and test as a model column" if hit else "FAIL - no measure reaches |t| 3.0 with both halves at 2.0 and clean placebos"))
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    main()
