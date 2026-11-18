r"""NS (NS_PLAN.txt): overnight trades directed by market structure read at 18:00.
    python ns_structure.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import nt_night as N

N.OUT = os.path.join(HERE, 'NS_RESULTS.txt')


def bars(df, rule, offset=None):
    b = df.resample(rule, label='left', closed='left', offset=offset).agg({'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
    return b


def swings(b, n=2):
    h, l = b.h.values, b.l.values; sh = np.zeros(len(b), bool); sl = np.zeros(len(b), bool)
    for i in range(n, len(b) - n):
        sh[i] = h[i] > max(h[i - n:i].max(), h[i + 1:i + n + 1].max()); sl[i] = l[i] < min(l[i - n:i].min(), l[i + 1:i + n + 1].min())
    return sh, sl


def structure_at(b, rule_minutes, n=2):
    sh, sl = swings(b, n); t = b.index.values.astype('datetime64[m]').astype(np.int64)
    conf = t + (n + 1) * rule_minutes
    H = [(conf[i], b.h.values[i]) for i in np.flatnonzero(sh)]; L = [(conf[i], b.l.values[i]) for i in np.flatnonzero(sl)]
    return H, L, t + rule_minutes, b.c.values


def state(H, L, when, mode, closes=None):
    hs = [v for c, v in H if c <= when][-2:]; ls = [v for c, v in L if c <= when][-2:]
    if len(hs) < 2 or len(ls) < 2: return 0
    if mode == 'trend':
        if hs[1] > hs[0] and ls[1] > ls[0]: return 1
        if hs[1] < hs[0] and ls[1] < ls[0]: return -1
        return 0
    c = closes
    return 1 if c > hs[1] else -1 if c < ls[1] else 0


def signals(df):
    T1 = df.index.values.astype('datetime64[m]').astype(np.int64); C1 = df.c.values
    d24 = bars(df, '1D', offset='18h'); h4 = bars(df, '4h', offset='2h'); h1 = bars(df, '1h')
    S = {nm: structure_at(b, m) for nm, b, m in (('d', d24, 1440), ('h4', h4, 240), ('h1', h1, 60))}
    rth = df.between_time('09:30', '15:59'); rd = rth.groupby(rth.index.normalize()).agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    out = {}
    for day in range(int(T1[0] // 1440), int(T1[-1] // 1440) + 1):
        when = day * 1440 + 18 * 60; k = pd.Timestamp(day * 86400, unit='s')
        i = np.searchsorted(T1, day * 1440 + 16 * 60, side='right') - 1
        if i < 0 or k not in rd.index: continue
        c16 = C1[i]; pr = rd[rd.index < k]
        row = {f'S{j + 1}': state(*S[nm][:2], when, 'trend') for j, nm in enumerate(('d', 'h4', 'h1'))}
        row['S4'] = state(*S['d'][:2], when, 'bos', c16)
        row['S5'] = (1 if c16 > pr.h.iloc[-1] else -1 if c16 < pr.l.iloc[-1] else 0) if len(pr) else 0
        for j in (1, 2, 3): row[f'S{j}L'] = max(row[f'S{j}'], 0)
        out[day] = row
    return pd.DataFrame.from_dict(out, orient='index')


def run(df, f5, sig, col):
    T1 = df.index.values.astype('datetime64[m]').astype(np.int64); O1 = df.o.values; C1 = df.c.values; res = {}
    for day, sg in sig[col].items():
        s = day * 1440 + 18 * 60 + 5; e = (day + 1) * 1440 + 9 * 60 + 29
        i = np.searchsorted(T1, s); j = np.searchsorted(T1, e, side='right') - 1
        if i >= len(T1) or T1[i] - s > 10 or j < 0 or e - T1[j] > 10: continue
        res[pd.Timestamp((day + 1) * 86400, unit='s')] = 0.0 if sg == 0 else sg * (C1[j] - O1[i]) / O1[i] * N.NUSD - 2 * N.SIDE
    return pd.Series(res, dtype=float)


def main():
    t0 = time.time(); N.LINES.clear(); say = N.say
    say(); say("=" * 150); say(f"NS  OVERNIGHT TRADES DIRECTED BY MARKET STRUCTURE  ns_structure.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  18:05 -> 09:29, $5.72 a round trip"); say("=" * 150)
    nq = N.frame('nq'); sig = signals(nq[0]); sig['ALL'] = 1; alive = []
    N.grade(run(*nq, sig, 'ALL'), 'reference: ALWAYS long (no structure)', '2010-07-01', '2019-05-06')
    for col, lab in (('S1', 'S1 daily trend structure'), ('S2', 'S2 4-hour trend structure'), ('S3', 'S3 1-hour trend structure'), ('S4', 'S4 daily break of structure'),
                     ('S5', 'S5 close above / below prior day range'), ('S1L', 'S1L daily structure, long only'), ('S2L', 'S2L 4-hour structure, long only'), ('S3L', 'S3L 1-hour structure, long only')):
        x = run(*nq, sig, col); t, ly, ny, d5 = N.grade(x, f"{lab} (in market {(sig[col] != 0).mean():.0%})", '2010-07-01', '2019-05-06')
        ok = t >= 2.7 and ly <= ny / 2 and d5 > 0; say(f"      -> {'SURVIVES discovery' if ok else 'DEAD (kill rule)'}")
        if ok: alive.append((col, lab))
    if alive:
        mnq = N.frame('mnqx'); es = N.frame('es'); sm = signals(mnq[0]); se = signals(es[0]); sm['ALL'] = 1
        N.grade(run(*mnq, sm, 'ALL'), 'reference: ALWAYS long, HOLDOUT MNQ', '2019-05-06', '2026-11-01')
        for col, lab in alive:
            N.grade(run(*mnq, sm, col), lab + ' HOLDOUT MNQ', '2019-05-06', '2026-11-01'); N.grade(run(*es, se, col), lab + ' ES discovery', '2010-07-01', '2019-05-06')
    else: say("   no survivor - holdout not opened")
    say(f"   trial count +8; run {time.time() - t0:.0f} s")
    with open(N.OUT, 'a', encoding='utf-8') as f: f.write('\n'.join(N.LINES) + '\n')


if __name__ == '__main__':
    main()
