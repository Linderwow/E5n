import os, sys, glob, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, OPS); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_daily as D
import e5n_train as TR
from e5n_model import ModelSet

MICRO = os.path.join(TR.RESEARCH, 'regime_work', 'micro')
OUT = os.path.join(HERE, 'E5X_RESULTS.txt')
COMM = 0.61; REST = 30
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / (s / np.sqrt(len(x)))) if s > 0 else 0.0


def fill(z, s1, side, mode):
    bid, ask, px, bv, sv = z['bid'], z['ask'], z['px'], z['bv'], z['sv']
    s1 = min(s1, 23399)
    mkt = lambda s: ask[min(s, 23399)] if side > 0 else bid[min(s, 23399)]
    if mode == 'market' or s1 + REST + 1 > 23399: return mkt(s1), False
    L = bid[s1] if side > 0 else ask[s1]
    for s in range(s1 + 1, s1 + REST + 1):
        if mode == 'through': hit = (ask[s] <= L) if side > 0 else (bid[s] >= L)
        else: hit = (px[s] <= L and sv[s] > 0) if side > 0 else (px[s] >= L and bv[s] > 0)
        if hit: return L, True
    return mkt(s1 + REST + 1), False


def day_pnl(z, trail, mode):
    pos = 0; cash = 0.0; comm = 0.0; n = 0; rested = 0
    for minute, tgt in trail:
        q = tgt - pos
        if q == 0: continue
        p, r = fill(z, minute * 60 + 1, 1 if q > 0 else -1, mode)
        if not np.isfinite(p): return None
        cash -= q * p; comm += abs(q) * COMM; pos = tgt; n += 1; rested += r
    if pos != 0:
        p = z['bid'][23399] if pos > 0 else z['ask'][23399]
        if not np.isfinite(p): return None
        cash += pos * p; comm += abs(pos) * COMM; n += 1
    return cash * 2.0 - comm, n, rested


def main():
    t0 = time.time()
    M = ModelSet(D.MODELS); bars = D.minute_bars(); bars = bars[bars.date >= '2025-08-01']; S = D.Sessions(bars); py, dec = D.python_days(S, M)
    files = {os.path.basename(f)[:10]: f for f in glob.glob(os.path.join(MICRO, '*.npz'))}
    rows = []
    for d in sorted(py):
        key = str(d)
        if key not in files or not py[d]['trail']:
            if key in files: rows.append(dict(date=d, market=0.0, through=0.0, touch=0.0, bar=0.0, orders=0, rested_through=0, rested_touch=0))
            continue
        z = {k: v.astype(np.float64) for k, v in np.load(files[key]).items() if k != 'contract'}
        trail = [(int(a.split(':')[0]), int(a.split(':')[1])) for a in py[d]['trail'].split()]
        r = {m: day_pnl(z, trail, m) for m in ('market', 'through', 'touch')}
        if any(v is None for v in r.values()): continue
        rows.append(dict(date=d, market=r['market'][0], through=r['through'][0], touch=r['touch'][0],
                         bar=py[d]['gross_pts'] * 2 - py[d]['round_trips'] * 2 * (0.5 * 2.35 + COMM), orders=r['market'][1],
                         rested_through=r['through'][2], rested_touch=r['touch'][2]))
    R = pd.DataFrame(rows).set_index('date'); R.index = pd.to_datetime(R.index)
    say(); say("=" * 120); say(f"e5x_exec.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  X8 resting order first (30 s at the touch, then market) vs market orders; MNQ ticks, $0.61 a side"); say("=" * 120)
    n = int(R.orders.sum())
    say(f"{len(R)} sessions {R.index.min().date()}..{R.index.max().date()}, {n:,} orders | check: market orders at the quotes ${R.market.sum():,.0f} vs "
        f"bar-open fills with 2.35 ticks ${R.bar.sum():,.0f} (same decisions)")
    half = R.index <= R.index[len(R) // 2]
    for lab, col, rc in (('conservative (the market trades through the order)', 'through', 'rested_through'), ('optimistic (a trade prints at the order price)', 'touch', 'rested_touch')):
        dlt = R[col] - R.market
        say(f"{lab}: total ${R[col].sum():,.0f} vs market ${R.market.sum():,.0f} | saving ${dlt.sum():,.0f} = ${dlt.sum() / max(n, 1):.2f} an order "
            f"({dlt.sum() / max(n, 1) / 0.5:.2f} ticks) | day t {tt(dlt):+.2f} | first half ${dlt[half].sum():,.0f}, second half ${dlt[~half].sum():,.0f} | "
            f"filled while resting {R[rc].sum() / max(n, 1):.0%} of orders")
    dlt = R.through - R.market; ok = dlt.sum() > 0 and tt(dlt) >= 2.0 and dlt[half].sum() > 0 and dlt[~half].sum() > 0
    say(f"VERDICT X8: {'PASS - resting first saves money on this year of ticks (forward-test candidate; needs an order-handling build)' if ok else 'not an improvement by the registered bar'}")
    say(f"run {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")
