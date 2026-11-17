r"""E5n exit statistics (descriptive, not a trial): every band-leg and model-leg trade 2013-26 by how it ended - peak open profit, result,
share of the peak kept, win rate, holding time. 1 micro, $ at today's price, before costs unless stated.
    python e5q_exits.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5n_band as B
from e5b_lab import G, say
X = L.X; U = L.U; COST = 2 * (0.5 * 2.35 + 0.61)


def band_trades():
    out = []
    for nm, sel in (('NQ', lambda d: d <= L.JOIN), ('MNQ', lambda d: d > L.JOIN)):
        df = G.df5[nm].sort_values('time').reset_index(drop=True); df['nxt'] = df.open.shift(-1)
        marks = []; B.run(df, live=True, warmup=B.LOOK, bars=marks); mk = pd.concat(marks, ignore_index=True)
        mk = mk.merge(df[['time', 'high', 'low', 'nxt']], on='time', how='left'); mk['day'] = mk.time.dt.normalize()
        for day, g in mk.groupby('day', sort=True):
            if not sel(day) or day.year < 2013: continue
            tg = g.target.values; n = len(tg); i = 0
            c, hi, lo, nx, at, tm = g.close.values, g.high.values, g.low.values, g.nxt.values, g.atr.values, g.time.values
            while i < n:
                if tg[i] == 0 or (i > 0 and tg[i] == tg[i - 1]): i += 1; continue
                j = i + 1
                while j < n and tg[j] == tg[i]: j += 1
                if j >= n: j = n - 1
                d = np.sign(tg[i]); ent = nx[i]; last = pd.Timestamp(tm[j]).hour * 60 + pd.Timestamp(tm[j]).minute >= 960
                ex = c[j] if last else nx[j]
                ext = ent
                for k in range(i + 1, j + 1): ext = max(ext, c[k]) if d > 0 else min(ext, c[k])
                trail = (d > 0 and c[j] < ext - B.CH * at[j]) or (d < 0 and c[j] > ext + B.CH * at[j])
                reason = 'session end 16:00' if last else ('3 x ATR trail' if trail else '4 closes back inside the band')
                seg_h = hi[i + 1:j + 1]; seg_l = lo[i + 1:j + 1]
                if not len(seg_h): i = j; continue
                mfe = (seg_h.max() - ent) if d > 0 else (ent - seg_l.min()); mae = (ent - seg_l.min()) if d > 0 else (seg_h.max() - ent)
                pk = int(np.argmax(seg_h) if d > 0 else np.argmin(seg_l))
                out.append(dict(day=day, side=d, reason=reason, atr=at[i], peak=max(mfe, 0) / ent * U, worst=max(mae, 0) / ent * U, res=d * (ex - ent) / ent * U,
                                hold=(j - i) * 5, to_peak=(pk + 1) * 5, entry_min=pd.Timestamp(tm[i]).hour * 60 + pd.Timestamp(tm[i]).minute, k=max(mfe, 0) / at[i]))
                i = j
    return pd.DataFrame(out)


def model_trades(path):
    M = C.M; out = []; keep = np.asarray(M.keep)
    for r in np.flatnonzero(keep):
        p = path[r]; px = M.px[r]; n = len(p); i = 0
        while i < n:
            if p[i] == 0 or (i > 0 and p[i] == p[i - 1]): i += 1; continue
            j = i + 1
            while j < n and p[j] == p[i]: j += 1
            d = np.sign(p[i]); ent = px[i]; ex = px[j] if j < n else M.exit[r]
            seg = np.r_[px[i + 1:j + 1] if j < n else np.r_[px[i + 1:], M.exit[r]]]
            if not len(seg): i = j; continue
            mfe = max(float((d * (seg - ent)).max()), 0.0); mae = max(float((-d * (seg - ent)).max()), 0.0)
            reason = 'session end 16:00' if j >= n else ('forecast flipped to the other side' if p[j] != 0 else 'forecast no longer strong enough')
            out.append(dict(day=M.dates[r], side=d, reason=reason, peak=mfe / ent * U, worst=mae / ent * U, res=d * (ex - ent) / ent * U, hold=j - i))
            i = j
    return pd.DataFrame(out)


def table(T, yrs, title, order):
    say(title)
    say(f"   {'how the trade ended':38s} | trades/yr | share | winners | avg peak | avg result | avg loss on losers | kept of the peak | avg hold")
    for r in order + ['ALL']:
        s = T if r == 'ALL' else T[T.reason == r]
        if not len(s): continue
        w = s[s.res > 0]; l = s[s.res <= 0]
        say(f"   {r:38s} | {len(s) / yrs:9.0f} | {len(s) / len(T) * 100:4.0f}% | {len(w) / len(s) * 100:6.0f}% | ${s.peak.mean():7,.0f} | ${s.res.mean():+9,.0f} | ${l.res.mean() if len(l) else 0:+17,.0f} | "
            f"{s.res.sum() / s.peak.sum() * 100:15.0f}% | {s.hold.mean():5.0f} min")
    w = T[T.res > 0]; l = T[T.res <= 0]
    say(f"   winners {len(w) / len(T) * 100:.0f}%: average win ${w.res.mean():+,.0f}, average loss ${l.res.mean():+,.0f}, payoff {w.res.mean() / -l.res.mean():.2f} to 1; per trade ${T.res.mean():+,.2f} before costs, "
        f"${T.res.mean() - COST:+,.2f} after ${COST:.2f} a round trip; ${T.res.sum() / yrs:+,.0f} a year before costs, ${(T.res.sum() - COST * len(T)) / yrs:+,.0f} after")


def main():
    t0 = time.time(); C.load(); M = C.M; keep = np.asarray(M.keep); yrs = keep.sum() / 252
    say(); say("=" * 160)
    say(f"EXIT STATISTICS  e5q_exits.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  every E5n trade 2013-01 .. 2026-10 by how it ended; 1 micro, $ at today's price, before costs unless stated; "
        f"peak = best open profit on the way (5-minute highs / lows for the band leg, minute prices for the model leg)")
    say("=" * 160)
    T = band_trades()
    table(T, yrs, "BAND LEG", ['3 x ATR trail', '4 closes back inside the band', 'session end 16:00'])
    say("   by size of the peak (band leg):")
    for lab, lo, hi in (('peak under $100', 0, 100), ('$100-300', 100, 300), ('$300-600', 300, 600), ('$600-1,000', 600, 1000), ('over $1,000', 1000, 1e9)):
        s = T[(T.peak >= lo) & (T.peak < hi)]
        if not len(s): continue
        q = np.percentile(s.res, [25, 50, 75])
        say(f"      {lab:16s}: {len(s) / yrs:4.0f} a year ({len(s) / len(T) * 100:2.0f}%), average result ${s.res.mean():+7,.0f}, middle half ${q[0]:+,.0f} .. ${q[2]:+,.0f}, kept of the peak {s.res.sum() / s.peak.sum() * 100:4.0f}%, "
            f"ended in a loss {np.mean(s.res < 0) * 100:3.0f}%, share of the leg's profit {s.res.sum() / T.res.sum() * 100:4.0f}%")
    big = T[T.peak >= 600]
    say(f"   trades that peaked at +$600 or more: gave back on average ${(big.peak - big.res).mean():,.0f} from the peak; the peak came {big.to_peak.mean():.0f} minutes after the entry, the exit {big.hold.mean():.0f} minutes after it; "
        f"ended via trail {np.mean(big.reason == '3 x ATR trail') * 100:.0f}%, back inside the band {np.mean(big.reason == '4 closes back inside the band') * 100:.0f}%, session end {np.mean(big.reason == 'session end 16:00') * 100:.0f}%")
    say(f"   worst open loss on the way (all band trades): average ${T.worst.mean():,.0f}, 95th percentile ${np.percentile(T.worst, 95):,.0f}; winners went ${T[T.res > 0].worst.mean():,.0f} against first on average")
    say("   by side: " + " | ".join(f"{'long' if s_ > 0 else 'short'} {len(g) / yrs:.0f}/yr, winners {np.mean(g.res > 0) * 100:.0f}%, ${g.res.mean():+,.0f} a trade, kept {g.res.sum() / g.peak.sum() * 100:.0f}% of the peak" for s_, g in T.groupby('side')))
    say("   by period: " + " | ".join(f"{lab} {len(g) / (n_ / 252):.0f}/yr, winners {np.mean(g.res > 0) * 100:.0f}%, ${g.res.mean():+,.0f} a trade, kept {g.res.sum() / g.peak.sum() * 100:.0f}%"
                                      for lab, g, n_ in (('2013-19', T[T.day <= L.JOIN], (keep & np.asarray(G.isA)).sum()), ('2019-26', T[T.day > L.JOIN], (keep & ~np.asarray(G.isA)).sum()))))
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); dep = C.path(M.g5, M.g15, 'set', clk0)
    Tm = model_trades(dep)
    say(); table(Tm, yrs, "MODEL LEG (as deployed, with the 15-minute check)", ['forecast no longer strong enough', 'forecast flipped to the other side', 'session end 16:00'])
    say("   by holding time (model leg): " + " | ".join(f"{lab} {len(g) / yrs:.0f}/yr ${g.res.mean():+,.1f} a trade" for lab, g in (('5 min', Tm[Tm.hold <= 5]), ('10-15 min', Tm[(Tm.hold > 5) & (Tm.hold <= 15)]),
                                                                                                                               ('20-60 min', Tm[(Tm.hold > 15) & (Tm.hold <= 60)]), ('over 60 min', Tm[Tm.hold > 60]))))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
