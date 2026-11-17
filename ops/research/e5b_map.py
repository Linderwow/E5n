import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import G, say

C1 = None


def trades(p, leg):
    rows = []; px = G.px; ex = G.exit
    for i in np.flatnonzero(G.keep & G.isA):
        j = 0; row = p[i]
        while j < L.ND:
            if row[j] == 0: j += 1; continue
            d = row[j]; j0 = j
            while j < L.ND and row[j] == d: j += 1
            pe = px[i, j] if j < L.ND else ex[i]
            path = np.r_[px[i, j0:j], pe]; fav = d * (path - px[i, j0])
            rows.append((i, leg, int(d), j0, j, d * (pe - px[i, j0]) / G.o930[i] * L.U, fav.max() / G.o930[i] * L.U, fav.min() / G.o930[i] * L.U, j >= L.ND))
    t = pd.DataFrame(rows, columns=['i', 'leg', 'dir', 'j0', 'j1', 'gross', 'mfe', 'mae', 'eod'])
    t['net'] = t.gross - 2 * C1; t['mins'] = (t.j1 - t.j0) * 5; t['hour'] = (570 + G.dec[t.j0.values]) // 60
    return t


def tab(t, by, lab):
    g = t.groupby(by, observed=True).agg(n=('net', 'size'), win=('net', lambda x: (x > 0).mean()), avg=('net', 'mean'), total=('net', 'sum'),
                                         aw=('net', lambda x: x[x > 0].mean()), al=('net', lambda x: x[x <= 0].mean()))
    say(f"   by {lab}:")
    for k, r in g.iterrows():
        say(f"      {str(k):24s} {int(r.n):6,d} trades | win {r.win:4.0%} | avg win ${r.aw:6.0f}, avg loss ${r.al:6.0f} | ${r.avg:+6.1f} a trade | total ${r.total:+9,.0f}")


def main():
    global C1
    t0 = time.time(); L.setup(); C1 = L.X.SIDE(2.35) * L.U; T = G.T; m = G.m
    A = G.keep & G.isA; yrs = A.sum() / 252
    say(); say("=" * 150); say(f"E5B TRADE MAP  e5b_map.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  2013-01 .. 2019-05 ONLY (NQ; the 2019-26 half is not looked at here); $ per micro at today's price, live cost"); say("=" * 150)
    mt = trades(G.model, 'model'); bt = trades(G.band, 'band')
    vol = m(T.vol30.values); dv = T.dvol20.values[::L.ND]; gap = T.gap.values[::L.ND]
    o = G.px[:, 0]; hi = np.maximum(G.px.max(1), G.exit); lo = np.minimum(G.px.min(1), G.exit); move = (G.exit - G.o930) / G.o930; eff = np.abs(G.exit - G.o930) / np.maximum(hi - lo, 1e-9)
    for t, nm in ((mt, 'MODEL LEG (with the 15-minute check)'), (bt, 'BAND LEG')):
        say(); say(f"{nm}: {len(t):,} trades in {yrs:.1f} years ({len(t) / A.sum():.2f} a day) | win {np.mean(t.net > 0):.0%} | avg win ${t.net[t.net > 0].mean():.0f}, avg loss ${t.net[t.net <= 0].mean():.0f} | "
            f"${t.net.mean():+.1f} a trade | total ${t.net.sum():+,.0f} | gross ${t.gross.sum():+,.0f}, costs ${2 * C1 * len(t):,.0f}")
        say(f"   best excursion while open: avg ${t.mfe.mean():.0f}; kept at the exit: {t.gross.sum() / t.mfe.sum():.0%} of it | winners that were ahead by 2 costs and ended as losers: {np.mean((t.mfe > 4 * C1) & (t.net <= 0)):.0%} of trades")
        tab(t, 'dir', 'direction (+1 long, -1 short)'); tab(t, 'hour', 'entry hour (ET)')
        tab(t, pd.cut(t.mins, [0, 5, 10, 20, 40, 80, 160, 400]), 'minutes held')
        oth = (G.band if nm.startswith('MODEL') else G.model)[t.i.values, t.j0.values] * t.dir.values
        tab(t, pd.Series(np.where(oth > 0, 'other leg same side', np.where(oth < 0, 'other leg opposite', 'other leg flat')), index=t.index), 'other leg at entry')
        f15 = G.f15[t.i.values, t.j0.values] * t.dir.values; f60 = G.f60[t.i.values, t.j0.values] * t.dir.values; f30 = G.f30[t.i.values, t.j0.values] * t.dir.values
        if nm.startswith('BAND'):
            f5 = G.f5[t.i.values, t.j0.values] * t.dir.values
            tab(t, pd.Series(np.where(f5 > L.T0, '5-min forecast agrees', np.where(f5 < -L.T0, '5-min forecast opposes', '5-min forecast neutral')), index=t.index), 'model 5-minute forecast at entry')
            tab(t, pd.Series(np.where(f15 > 0, '15-min forecast agrees', '15-min forecast opposes'), index=t.index), 'model 15-minute forecast at entry')
            tab(t, t.eod.map({True: 'held to 16:00', False: 'exited earlier'}), 'how it ended')
        tab(t, pd.Series(np.where(f30 > 0, '30-min forecast agrees', '30-min forecast opposes'), index=t.index), 'model 30-minute forecast at entry')
        tab(t, pd.Series(np.where(f60 > 0, '60-min forecast agrees', '60-min forecast opposes'), index=t.index), 'model 60-minute forecast at entry')
        q = G.diag['q'][t.i.values, t.j0.values] * t.dir.values
        tab(t, pd.Series(np.where(q > 0, 'with EMA50/200 trend', np.where(q < 0, 'against EMA50/200 trend', 'no reading')), index=t.index), '5-minute EMA 50 / 200 trend at entry')
        vv = vol[t.i.values, t.j0.values]; tab(t, pd.qcut(vv, 4, labels=['quietest quarter', 'q2', 'q3', 'most volatile quarter']), '30-minute volatility at entry')
        tab(t, pd.qcut(dv[t.i.values], 3, labels=['calm weeks', 'middle', 'volatile weeks']), '20-day volatility')
        tab(t, pd.qcut(eff[t.i.values], 3, labels=['choppy day', 'middle', 'trend day']), 'day type (|open to close| / range, known only afterwards)')
    book = L.sim(G.model, G.band); mo = L.sim(G.model, G.band, 1, 0); ba = L.sim(G.model, G.band, 0, 1)
    bA, mA, aA = (x[x.index <= L.JOIN] for x in (book, mo, ba))
    say(); say(f"BOOK DAYS 2013-19: {len(bA):,} sessions | winning days {np.mean(bA > 0):.0%}, flat {np.mean(bA == 0):.0%} | avg win ${bA[bA > 0].mean():.0f}, avg loss ${bA[bA < 0].mean():.0f} | "
               f"best 20 days = {bA.sort_values().tail(20).sum() / bA.sum():.0%} of the profit | worst 20 days ${bA.sort_values().head(20).sum():,.0f}")
    worst = bA.sort_values().head(15)
    say("   15 worst book days: " + "; ".join(f"{d:%Y-%m-%d} {v:+,.0f} (model {mA[d]:+,.0f}, band {aA[d]:+,.0f})" for d, v in worst.items()))
    net = G.model + G.band; cum = np.cumsum(net * G.dpx, axis=1) / G.o930[:, None] * L.U; peak = np.maximum.accumulate(np.maximum(cum, 0), axis=1)
    pk = peak[:, -1][A]; end = cum[:, -1][A]; big = pk >= 300
    say(f"   give-back: on the {int(big.sum())} days the book was ahead by $300+ at some point (gross), it kept {end[big].sum() / pk[big].sum():.0%} of the peak; "
        f"{np.mean(end[big] <= 0):.0%} of those days ended at or below zero")
    low = np.minimum.accumulate(np.minimum(cum, 0), axis=1)[:, -1][A]; deep = low <= -300
    say(f"   recovery: on the {int(deep.sum())} days the book was behind by $300+ at some point (gross), the day ended at ${end[deep].mean():+.0f} on average vs ${low[deep].mean():+.0f} at the worst; "
        f"{np.mean(end[deep] > 0):.0%} of them ended positive")
    mv = np.abs(move[A]); bigd = mv >= 0.01; bookA = bA.values; cap = bookA[bigd] / (mv[bigd] * L.U)
    say(f"   big-move days (|open to close| >= 1%): {int(bigd.sum())} of {len(mv)}; the book made ${bookA[bigd].mean():+.0f} on average = {np.median(cap):.0%} of the move (median); "
        f"lost money on {np.mean(bookA[bigd] < 0):.0%} of them; band leg flat all day on {np.mean((np.abs(G.band[A][bigd]).sum(1) == 0)):.0%}")
    small = mv < 0.003
    say(f"   small-move days (|open to close| < 0.3%): {int(small.sum())}; the book made ${bookA[small].mean():+.0f} on average, lost on {np.mean(bookA[small] < 0):.0%}")
    st = np.where(G.model * G.band > 0, 'legs agree', np.where(G.model * G.band < 0, 'legs oppose', np.where(G.model != 0, 'model only', np.where(G.band != 0, 'band only', 'both flat'))))
    say("   time in each state and the gross $ made per 5 minutes of 1 micro of each leg held in it:")
    for s in ('both flat', 'model only', 'band only', 'legs agree', 'legs oppose'):
        k = (st == s) & A[:, None]; gm = (G.model * G.dpx)[k].sum() / (G.o930[:, None] * np.ones_like(G.dpx))[k].mean() * L.U if k.any() else 0
        gb = (G.band * G.dpx)[k].sum() / G.o930[A].mean() * L.U
        say(f"      {s:12s} {k.sum() / (A.sum() * L.ND):5.1%} of the session | model leg ${gm:+10,.0f} gross in total, band leg ${gb:+10,.0f}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
