import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5d_more as D
import e5n_band as B
from e5b_lab import G, say
X = L.X; ND = L.ND
MK = {'MYM': (0.5, 1.0, 1), 'M2K': (5.0, 0.1, 1), 'MES': (5.0, 0.25, 1), 'MGC': (10.0, 0.1, 1), 'MCL': (100.0, 0.01, 1), 'M6E': (12500.0, 0.0001, 1), 'MBT': (0.1, 5.0, 3)}
NAMES = {'MYM': 'Dow', 'M2K': 'Russell 2000', 'MES': 'S&P 500', 'MGC': 'gold', 'MCL': 'crude oil', 'M6E': 'euro', 'MBT': 'bitcoin'}
SPLIT = pd.Timestamp('2023-01-01')


def tstat(s):
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std() > 0 else 0.0


def main():
    t0 = time.time(); L.setup(); m, b = G.model, G.band
    say(); say("=" * 160); say(f"E5B AMENDMENT E  e5h_markets.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  band exit time, entries by volatility, the band leg on other micro futures"); say("=" * 160)
    book = lambda t: L.sim(m, b, ticks=t); dec = G.dec
    Z = []
    for nm, lim in (('Z1', 375), ('Z2', 360), ('Z3', 330)):
        bz = np.where(dec[None, :] >= lim, 0.0, b); Z.append(dict(name=nm, desc=f'band leg flat at {(570 + lim) // 60}:{(570 + lim) % 60:02d}', fn=(lambda bz: lambda t: L.sim(net=m + bz, ticks=t))(bz)))
    D.judge('Z', Z, book, 'the netted book B0')
    v = G.m(G.T.vol30.values); q25 = np.full_like(v, np.nan); q75 = np.full_like(v, np.nan)
    for j in range(ND):
        s = pd.Series(v[:, j]); q25[:, j] = s.rolling(250, min_periods=120).quantile(0.25).shift(1).values; q75[:, j] = s.rolling(250, min_periods=120).quantile(0.75).shift(1).values

    def jm(M): return (M - 585) // 5

    def gate_low(date, M, new, regular):
        i = G.row.get(date)
        if i is None or M < 585 or M > 955: return True
        x, lo = v[i, jm(M)], q25[i, jm(M)]
        return not (x == x and lo == lo and x < lo)

    b_low = L.band_path(gate=gate_low)

    def boost(bp):
        out = bp.copy()
        for i in range(G.n):
            row = bp[i]; j = 0
            while j < ND:
                if row[j] == 0: j += 1; continue
                d = row[j]; j0 = j
                while j < ND and row[j] == d: j += 1
                je = max(j0 - 1, 0); x, hi = v[i, je], q75[i, je]
                if x == x and hi == hi and x > hi: out[i, j0:j] = 2 * d
        return out

    VB = [dict(name='VB1', desc='no band entry in the lowest volatility quartile', fn=lambda t: L.sim(net=m + b_low, ticks=t)),
          dict(name='VB2', desc='band trade doubled when entered in the highest volatility quartile', fn=(lambda bb: lambda t: L.sim(net=m + bb, ticks=t))(boost(b))),
          dict(name='VB3', desc='both', fn=(lambda bb: lambda t: L.sim(net=m + bb, ticks=t))(boost(b_low)))]
    D.judge('VB', VB, book, 'the netted book B0')
    say(); say("FAMILY O - the band leg exactly as deployed on other micro futures (2019-26 only; split at 2023-01-01; $ per micro of that market; cost 1 tick + $0.61 a side, bitcoin 3 ticks)")
    b0 = book(2.35); bB = b0[b0.index > L.JOIN]; mnq = L.sim(m, b, 0, 1); mnqB = mnq[mnq.index > L.JOIN]
    say(L.line('   MNQ band leg for reference (2019-26 in the middle column)', mnq)); cands = []
    for k, (code, (mult, tick, nt)) in enumerate(MK.items(), 1):
        f = os.path.join(L.TR.RESEARCH, 'regime_work', 'bars5m', code.lower() + '_5m.csv')
        df = pd.read_csv(f); df['time'] = pd.to_datetime(df.time); R = B.run(df.sort_values('time').reset_index(drop=True), live=True, warmup=B.LOOK)
        out = {}
        for lab, slip in (('1x', 1.0), ('2x', 2.0)):
            out[lab] = R.points * mult - R.sides * (slip * nt * tick * mult + 0.61)
        s = out['1x']; s2 = out['2x']; a, c = s[s.index < SPLIT], s[s.index >= SPLIT]; yrs = len(s) / 252
        top = s.sort_values(ascending=False); drop5 = s.drop(top.index[:5]); eq = s.cumsum(); dd = float((np.maximum(eq.cummax(), 0) - eq).max())
        j = pd.concat([bB.rename('book'), s.rename('leg'), mnqB.rename('mnq')], axis=1, join='inner'); j1, j2 = j[j.index < SPLIT], j[j.index >= SPLIT]
        up = all(X.sharpe(z.book + z.leg) > X.sharpe(z.book) for z in (j1, j2))
        ok = tstat(s) > 2.7 and a.sum() > 0 and c.sum() > 0 and drop5.sum() > 0 and up
        say(f"   O{k} {code} ({NAMES[code]}): {len(s):,} sessions, {R.sides.sum() / 2 / yrs:,.0f} trades/yr | ${s.sum() / yrs:+,.0f}/yr, t {tstat(s):+.2f}, Sharpe {X.sharpe(s):.2f} | halves ${a.mean() * 252:+,.0f} / ${c.mean() * 252:+,.0f} a year "
            f"(t {tstat(a):+.2f} / {tstat(c):+.2f}) | without its 5 best days ${drop5.sum() / yrs:+,.0f}/yr | max DD ${dd:,.0f} | at double slippage ${s2.sum() / yrs:+,.0f}/yr t {tstat(s2):+.2f} | "
            f"correlation with the MNQ band leg {j.leg.corr(j.mnq):+.2f}, with the book {j.leg.corr(j.book):+.2f} | book + 1 micro: Sharpe {X.sharpe(j1.book + j1.leg):.2f} / {X.sharpe(j2.book + j2.leg):.2f} vs {X.sharpe(j1.book):.2f} / {X.sharpe(j2.book):.2f} "
            f"-> {'FORWARD-TEST CANDIDATE' if ok else 'no'}")
        if ok: cands.append((code, s))
    if cands:
        j = pd.concat([bB.rename('book')] + [s.rename(c) for c, s in cands], axis=1, join='inner').fillna(0.0); tot = j.sum(axis=1)
        say(f"   book + 1 micro of each candidate ({', '.join(c for c, _ in cands)}), 2019-26 common days: ${tot.mean() * 252:,.0f}/yr, Sharpe {X.sharpe(tot):.2f}, max DD ${L.mdd(tot):,.0f} "
            f"vs the book alone ${j.book.mean() * 252:,.0f}/yr, Sharpe {X.sharpe(j.book):.2f}, max DD ${L.mdd(j.book):,.0f}")
    say("   CANDIDATES: " + (", ".join(c for c, _ in cands) if cands else "none")); say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
