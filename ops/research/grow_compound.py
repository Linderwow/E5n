r"""GROW (GROW_PLAN.txt): E5n's book compounded all-in from $8,000, as in Reddit thread 1jj15c5 ($8k -> $546k -> $3k).
    python grow_compound.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import say
from e5q_capture import days

START = 8000.0
PER_UNIT = (8000, 4000, 2000, 1000, 500)
PATHS, HORIZON, BLOCK = 5000, 252, 20
RUIN = 1500.0


def run_path(pnl, worst, per):
    eq = START; peak = START; mdd = 0.0; top = START
    for p, w in zip(pnl, worst):
        if eq <= RUIN: return eq, mdd, top, True
        u = max(1.0, np.floor(eq / per))
        if eq + u * w <= 0: return 0.0, 1.0, top, True
        eq += u * p; peak = max(peak, eq); top = max(top, eq); mdd = max(mdd, 1 - eq / peak)
    return eq, mdd, top, eq <= RUIN


def boot(pnl, worst, per, rng):
    n = len(pnl); nb = HORIZON // BLOCK + 1
    s = rng.integers(0, n - BLOCK, (PATHS, nb)); idx = (s[:, :, None] + np.arange(BLOCK)).reshape(PATHS, -1)[:, :HORIZON]
    P, W = pnl[idx], worst[idx]
    eq = np.full(PATHS, START); peak = eq.copy(); top = eq.copy(); mdd = np.zeros(PATHS); dead = np.zeros(PATHS, bool)
    for j in range(HORIZON):
        dead |= eq <= RUIN; u = np.where(dead, 0.0, np.maximum(1.0, np.floor(eq / per)))
        wipe = ~dead & (eq + u * W[:, j] <= 0); eq = np.where(wipe, 0.0, eq); mdd = np.where(wipe, 1.0, mdd); dead |= wipe
        eq = np.where(dead, eq, eq + u * P[:, j]); peak = np.maximum(peak, eq); top = np.maximum(top, eq)
        mdd = np.maximum(mdd, 1 - eq / peak)
    return eq, mdd, top, dead | (eq <= RUIN)


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'GROW_RESULTS.txt'); rng = np.random.default_rng(20261117)
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0)
    pm, wm = days(fair); pb, wb = days(M.band); keep = np.asarray(M.keep)
    pm, wm, pb, wb = pm[keep], wm[keep], pb[keep], wb[keep]; dates = pd.to_datetime(np.asarray(M.dates)[keep])
    books = {'BACKTEST book (1 model + 1 band micro a unit)': (pm + pb, wm + wb),
             'FAIR book (model leg at 2/3 of its backtest)': (pm * 2 / 3 + pb, wm * 2 / 3 + wb),
             'NO MODEL EDGE (model leg mean removed)': (pm - pm.mean() + pb, wm - pm.mean() + wb)}
    say(); say("=" * 150); say(f"GROW  E5n COMPOUNDED ALL-IN FROM ${START:,.0f}  grow_compound.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  units = floor(equity / $ per unit), resized daily"); say("=" * 150)
    say(f"   sessions {len(pm):,} ({dates.min():%Y-%m-%d} .. {dates.max():%Y-%m-%d}); 1-year paths = {PATHS:,} block bootstraps of {BLOCK} sessions; at least 1 unit; ruin = equity at or below $1,500 or an intraday low wiping it out")
    for lab, (p, w) in books.items():
        say(); say(f"   {lab}: ${p.mean() * 252:,.0f}/yr a unit, Sharpe {p.mean() / p.std() * np.sqrt(252):.2f}")
        say(f"   {'$ per unit':>10s} | {'median end':>11s} {'10%':>9s} {'90%':>11s} | {'ruined':>6s} {'hit $100k':>9s} {'hit $500k':>9s} | {'median max DD':>13s} {'DD>50%':>7s} {'DD>80%':>7s} | {'peaked $500k+ then ended < $50k':>31s}")
        for per in PER_UNIT:
            e, dd, top, ru = boot(p, w, per, rng)
            gave = ((top >= 500_000) & (e < 50_000)).mean()
            say(f"   {per:>10,} | {np.median(e):>11,.0f} {np.percentile(e, 10):>9,.0f} {np.percentile(e, 90):>11,.0f} | {ru.mean():>6.0%} {(top >= 100_000).mean():>9.0%} {(top >= 500_000).mean():>9.0%} | {np.median(dd):>13.0%} {(dd > 0.5).mean():>7.0%} {(dd > 0.8).mean():>7.0%} | {gave:>31.1%}")
    say(); say("   ACTUAL HISTORY, start $8,000 each January 2 (backtest book):")
    p, w = books['BACKTEST book (1 model + 1 band micro a unit)']; yrs = dates.year
    say(f"   {'year':>6s} | " + " | ".join(f"${per:,}/unit end   maxDD" for per in PER_UNIT))
    for y in sorted(set(yrs)):
        m = yrs == y
        if m.sum() < 120: continue
        cells = []
        for per in PER_UNIT:
            e, dd, top, ru = run_path(p[m], w[m], per); cells.append(f"{e:>12,.0f} {dd:>6.0%}{'R' if ru else ' '}")
        say(f"   {y:>6d} | " + " | ".join(cells))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
