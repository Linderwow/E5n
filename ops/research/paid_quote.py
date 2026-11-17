r"""Exact Databento prices for the three paid-data candidates, using only the free cost check (nothing is bought).
    python paid_quote.py      key from D:\nt_logs\databento_key.txt (the key alone, or key=...), else DATABENTO_API_KEY
    -> PAID_QUOTE.txt
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import pandas as pd
import databento as db
import e5n_paths as P

KEYFILE = r'D:\nt_logs\databento_key.txt'
OUT = os.path.join(HERE, 'PAID_QUOTE.txt')


def key():
    if os.path.exists(KEYFILE):
        for ln in open(KEYFILE, encoding='utf-8').read().splitlines():
            ln = ln.strip()
            if ln.startswith('db-') or ln.lower().startswith('key='): return ln.split('=', 1)[-1].strip()
    return os.environ.get('DATABENTO_API_KEY', '')


def main():
    c = db.Historical(key()); lines = []
    say = lambda s='': (print(s), lines.append(s))
    stocks = sorted({f.split('_')[0] for f in os.listdir(P.STOCK_BARS)})
    say(f"PAID DATA QUOTE  paid_quote.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  Databento free cost check (metadata.get_cost); nothing bought")
    rng = {}
    for ds in ('XNAS.ITCH', 'GLBX.MDP3'):
        r = c.metadata.get_dataset_range(dataset=ds); rng[ds] = r; say(f"  {ds} history {str(r['start'])[:10]} .. {str(r['end'])[:10]}")
    cost = lambda **k: float(c.metadata.get_cost(**k))
    sess = pd.bdate_range('2025-10-01', '2025-10-31'); n = len(sess)
    say(); say("CLOSING-AUCTION IMBALANCE  XNAS.ITCH imbalance, Nasdaq-100 members + former members (%d symbols)" % len(stocks))
    one = cost(dataset='XNAS.ITCH', schema='imbalance', symbols=stocks, start='2025-10-01', end='2025-11-01')
    win = sum(cost(dataset='XNAS.ITCH', schema='imbalance', symbols=stocks, start=f'{d:%Y-%m-%d}T19:50', end=f'{d:%Y-%m-%d}T20:01') for d in sess[:5]) / 5 * n
    yrs = (pd.Timestamp('2026-11-01') - pd.Timestamp(str(rng['XNAS.ITCH']['start'])[:10]).tz_localize(None)).days / 365.25
    say(f"  October 2025 full days ${one:,.2f}; 15:50-16:00 ET only ${win:,.2f}")
    say(f"  whole history ({yrs:.1f} yr) full days ~${one * 12 * yrs:,.0f}; closing window only ~${win * 12 * yrs:,.0f}")
    say(); say("ORDER-BOOK DEPTH  GLBX.MDP3, one month (October 2025)")
    for sym in ('NQ.v.0', 'MNQ.v.0'):
        for sch in ('mbp-10', 'mbo', 'mbp-1', 'trades'):
            m = cost(dataset='GLBX.MDP3', schema=sch, symbols=[sym], stype_in='continuous', start='2025-10-01', end='2025-11-01')
            rth = sum(cost(dataset='GLBX.MDP3', schema=sch, symbols=[sym], stype_in='continuous', start=f'{d:%Y-%m-%d}T13:30', end=f'{d:%Y-%m-%d}T20:00') for d in sess[:3]) / 3 * n
            say(f"  {sym:8s} {sch:7s} month ${m:>9,.2f} (09:30-16:00 only ${rth:>9,.2f}) -> 1 yr ~${m * 12:>10,.0f} / ${rth * 12:>9,.0f}; 2 yr ~${m * 24:>10,.0f} / ${rth * 24:>9,.0f}")
            time.sleep(0.3)
    say(); say("Months differ in activity (+/- 30% or more); these are scale estimates from one month, the portal's Batch tool shows exact totals.")
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
