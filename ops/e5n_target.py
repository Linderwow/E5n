r"""E5n band-leg profit target, SHADOW ONLY (research E5Q: near miss, not in the live rules). For every band-leg trade: once its open
profit on a minute price reaches K band half-widths ((upper - lower band) / 2 at the entry), it would leave at the next minute's
price instead of the leg's own exit. Reports the difference to the leg as traded, per session.
    python e5n_target.py            rebuild ops\data\e5n_shadow_target.csv from the bars on this PC and print the summary
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd

K = 2.0; LAST_CHECK = 384; USD_PT = 2.0
LEDGER = os.path.join(HERE, 'data', 'e5n_shadow_target.csv'); COLS = ['date', 'trades', 'hits', 'diff_pts', 'diff_usd', 'recorded']
N1, N2, LINE = 252, 504, 500.0


def shadow(cont, marks, k=K):
    end = pd.DatetimeIndex(cont.end); dec = ((end.hour * 60 + end.minute) - 571).values; dayv = end.normalize().values
    o = cont.o.values.astype(float); c = cont.c.values.astype(float); rth = (dec >= 0) & (dec <= 389)
    minute = {}
    for d_, k_, o_, c_ in zip(dayv[rth], dec[rth], o[rth], c[rth]): minute.setdefault(d_, {})[int(k_)] = (o_, c_)
    mk = marks.sort_values('time'); t = pd.DatetimeIndex(mk.time); mk = mk.assign(day=t.normalize().values, dec0=((t.hour * 60 + t.minute) - 570).values)
    rows = []
    for d_, g in mk.groupby('day', sort=True):
        g = g[(g.dec0 >= 15) & (g.dec0 <= 385)]; px = minute.get(np.datetime64(d_), {})
        if not len(g) or not px: continue
        tg = g.target.values.astype(float); d0 = g.dec0.values.astype(int); half = ((g.ub - g.lb) / 2).values.astype(float)
        close16 = px[max(px)][1]; n = len(tg); i = 0; diff = 0.0; hits = 0; trades = 0
        while i < n:
            if tg[i] == 0 or (i > 0 and tg[i] == tg[i - 1]): i += 1; continue
            j = i + 1
            while j < n and tg[j] == tg[i]: j += 1
            x = int(d0[j]) if j < n else None; s = float(np.sign(tg[i])); e = int(d0[i]); W = half[i]; trades += 1
            if e in px and W == W and W > 0:
                ent = px[e][0]; last = min(x - 1 if x is not None else LAST_CHECK, LAST_CHECK)
                for m in range(e + 1, last + 1):
                    if m not in px or s * (px[m][0] - ent) < k * W: continue
                    if m + 1 in px and (x is None or m + 1 < x):
                        base_exit = px[x][0] if x is not None and x in px else close16
                        diff += s * (px[m + 1][0] - base_exit) * abs(tg[i]); hits += 1
                    break
            i = j
        rows.append((pd.Timestamp(d_).date(), trades, hits, round(diff, 2)))
    return pd.DataFrame(rows, columns=['date', 'trades', 'hits', 'diff_pts'])


def append_ledger(sh, start, now):
    old = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    have = set(pd.to_datetime(old.date).dt.date); new = sh[(pd.to_datetime(sh.date) > start) & ~sh.date.isin(have)]
    if len(new):
        new = new.assign(diff_usd=(new.diff_pts * USD_PT).round(2), recorded=now.strftime('%Y-%m-%d %H:%M'))[COLS]
        new.to_csv(LEDGER, mode='a', header=not os.path.exists(LEDGER), index=False)
    return (pd.read_csv(LEDGER) if os.path.exists(LEDGER) else old), int(len(new))


def summary(L):
    if L is None or not len(L): return dict(sessions=0)
    x = L.diff_usd.astype(float); h = L[L.hits > 0]; n = len(L)
    s = dict(sessions=n, trades=int(L.trades.sum()), hit_days=int(len(h)), better=int((h.diff_usd > 0).sum()), worse=int((h.diff_usd < 0).sum()), diff_usd=round(float(x.sum()), 2))
    if n >= N1 and x.sum() <= -LINE: s['status'] = f"DROPPED: the target cost ${-x.sum():,.0f} over {n} sessions"
    elif n >= N1 and x.sum() >= LINE and s['better'] >= 0.6 * max(s['hit_days'], 1): s['status'] = f"CANDIDATE: the target added ${x.sum():,.0f} over {n} sessions ({s['better']} of {s['hit_days']} hit days better) - re-run the E5Q gauntlet"
    elif n >= N2: s['status'] = f"NOT ADOPTED: ${x.sum():+,.0f} over {n} sessions - no decision line reached"
    else: s['status'] = f"collecting ({n} of {N1} sessions): ${x.sum():+,.0f} on {s['hit_days']} hit day(s)"
    return s


if __name__ == '__main__':
    import datetime as dt
    import e5n_daily as D
    a, keep = D.all_bars(); cont, offs = D.continuous(a, keep); bd, marks = D.band_days(cont)
    sh = shadow(cont, marks)
    if os.path.exists(LEDGER): os.remove(LEDGER)
    L, added = append_ledger(sh, D.TRAIN_END, dt.datetime.now()); print(f"{added} sessions written to {LEDGER}"); print(summary(L))
    al = sh[pd.to_datetime(sh.date) <= D.TRAIN_END]; yrs = len(al) / 252
    print(f"same rule on the bars before the forward test ({len(al):,} sessions): {al.hits.sum() / yrs:.0f} hits a year, ${al.diff_pts.sum() * USD_PT / yrs:+,.0f} a year a micro (actual dollars)")
