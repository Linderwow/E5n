r"""EX (EX_PLAN.txt): order tactics on the tick store (A) and trade rules at the real cost (B).
    python ex_exec.py orders
    python ex_exec.py rules
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import say

REAL = 4.5


def orders():
    import ut_tests as U, ut_timing2 as T2
    t0 = time.time(); L.OUT = os.path.join(HERE, 'EX_RESULTS.txt')
    dlog = pd.read_csv(U.DLOG, usecols=['date', 'time_et', 'net']); dlog = dlog[dlog.date >= T2.START].copy()
    dlog['prev'] = dlog.groupby('date').net.shift(1).fillna(0); od = dlog[(dlog.net != dlog.prev) & (dlog.time_et < '16:00')].copy()
    od['dir'] = np.sign(od.net - od.prev); od['T'] = pd.to_datetime(od.date + ' ' + od.time_et)
    TK = T2.tick_days(); ep = pd.Timestamp('1970-01-01')
    tactics = {'BASE market +0.4 s': None, 'A1 limit ref+0 / 10 s': (0, 10), 'A2 limit ref+1 / 10 s': (1, 10), 'A3 limit ref+2 / 10 s': (2, 10),
               'A4 limit ref+4 / 10 s': (4, 10), 'A5 limit ref+1 / 3 s': (1, 3), 'A6 limit ref+1 / 30 s': (1, 30)}
    rows = []
    for r in od.itertuples():
        v = TK.get(pd.Timestamp(r.date))
        if v is None: continue
        ms, px, _ = v; T = int((r.T - ep).total_seconds() * 1000); s0 = T + 400
        i_ref = np.searchsorted(ms, T, side='right') - 1; i0 = np.searchsorted(ms, s0, side='right') - 1
        if i_ref < 0 or i0 < 0: continue
        ref = px[i_ref]; p0 = px[i0]; d = r.dir; row = {'day': r.date}
        for nm, tc in tactics.items():
            if tc is None: fill = p0
            else:
                k, w = tc; lim = ref + d * k * 0.25
                if d * (p0 - lim) <= 0: fill = p0
                else:
                    a, b = np.searchsorted(ms, s0, side='right'), np.searchsorted(ms, s0 + w * 1000, side='right')
                    seg = px[a:b]; hit = np.flatnonzero(d * (seg - lim) < 0)
                    fill = lim if len(hit) else px[b - 1]
            row[nm] = d * (fill - ref) / 0.25
        rows.append(row)
    X = pd.DataFrame(rows); days = sorted(X.day.unique()); half = days[len(days) // 2]
    say(); say("=" * 150); say(f"EX PART A  ORDER TACTICS  ex_exec.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {len(X):,} E5n orders on {len(days)} sessions (tick store); ticks paid vs the bar close"); say("=" * 150)
    for nm in tactics:
        sv = X['BASE market +0.4 s'] - X[nm]; g = sv.groupby(X.day).mean(); t = g.mean() / g.std() * np.sqrt(len(g)) if g.std() > 0 else np.nan
        h1 = sv[X.day < half].mean(); h2 = sv[X.day >= half].mean(); ok = nm != 'BASE market +0.4 s' and sv.mean() >= 1.0 and t >= 3.0 and h1 > 0 and h2 > 0
        say(f"   {nm:24s} cost {X[nm].mean():+5.2f} ticks (median {X[nm].median():+.2f}, worst 5% {np.percentile(X[nm], 95):+.1f}) | saving vs market {sv.mean():+5.2f} (session t {t:+.2f}; halves {h1:+.2f} / {h2:+.2f})"
            + (" -> ADOPT CANDIDATE" if ok else ''))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


def hyst_path(f5, f15, x, thr):
    n = f5.shape[0]; pos = np.zeros(n); out = np.zeros((n, C.NM)); M = C.M
    for j in range(C.NM):
        if M.grid[j]:
            ok = ~np.isnan(f5[:, j]); t = C.target(f5[:, j], f15[:, j], thr)
            keep = (pos != 0) & (f5[:, j] * pos > x * thr)
            new = np.where(t != 0, t, np.where(keep, pos, 0.0))
            pos = np.where(ok, new, pos)
        out[:, j] = pos
    return out


def rules():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'EX_RESULTS.txt'); T0 = C.T0
    base_net = C.path(M.g5, M.g15, 'set', M.grid) + M.band
    V = {f'B{i + 1} exit hysteresis: hold until f5 below {x} x T0': hyst_path(M.g5, M.g15, x, T0) + M.band for i, x in enumerate((0.5, 0.25, 0.0))}
    V.update({f'B{i + 4} entry threshold {m} x T0': C.path(M.g5, M.g15, 'set', M.grid, m * T0) + M.band for i, m in enumerate((1.25, 1.5, 2.0))})
    b45, tb = C.simm(base_net, REAL); b23, _ = C.simm(base_net, 2.35); kb = L.stats(b45); kb23 = L.stats(b23)
    say(); say("=" * 150); say(f"EX PART B  TRADE RULES AT {REAL} TICKS A SIDE  ex_exec.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n book 1 + 1"); say("=" * 150)
    say(f"   B0 baseline at {REAL} ticks: 2013-19 Sharpe {kb['shA']:.2f} DD ${kb['ddA']:,.0f} | 2019-26 Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr DD ${kb['ddB']:,.0f} | {tb / (len(b45) / 252):,.0f} round trips/yr | (2.35 ticks: 2019-26 Sharpe {kb23['shB']:.2f})")
    for nm, net in V.items():
        s, tr = C.simm(net, REAL); s23, _ = C.simm(net, 2.35); k = L.stats(s); k23 = L.stats(s23)
        sel = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']; _, sb = L.halves(s); _, bb = L.halves(b45); d = (sb - bb).values; t = d.mean() / d.std() * np.sqrt(len(d))
        ok = sel and k['shB'] > kb['shB'] and k['ddB'] <= kb['ddB'] and t >= 2.0 and k23['shB'] >= kb23['shB']
        say(f"   {nm:46s} 2013-19 Sharpe {k['shA']:.2f} ({k['shA'] - kb['shA']:+.2f}) DD ${k['ddA']:,.0f} {'SEL' if sel else '   '}| 2019-26 Sharpe {k['shB']:.2f} ({k['shB'] - kb['shB']:+.2f}) "
            f"${k['yrB']:,.0f}/yr DD ${k['ddB']:,.0f} t {t:+.2f} | {tr / (len(s) / 252):,.0f} trips/yr | 2.35 ticks 2019-26 Sharpe {k23['shB']:.2f}" + (' CONFIRMED' if ok else ''))
    say(f"trial count 10,519; run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'orders': orders()
    elif a == 'rules': rules()
    else: print(__doc__)
