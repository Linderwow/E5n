r"""Earnings report dates of the largest Nasdaq-100 companies from Nasdaq's public earnings calendar (one request per weekday inside
the reporting windows, 6 at a time, resumable) -> cache\web\earn_calendar.csv (date, symbol, time).
    python e5y_earncal.py
"""
import os, sys, json, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

WEB = PATHS.WEB; OUT = os.path.join(WEB, 'earn_calendar.csv'); DONE = os.path.join(WEB, 'earn_calendar_days.txt')
SYMS = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'FB', 'NVDA', 'TSLA', 'NFLX', 'AVGO', 'INTC', 'CSCO', 'QCOM', 'ADBE', 'AMD', 'COST', 'PEP'}
LOCK = threading.Lock()


def in_window(d):
    m, x = d.month, d.day
    return (m in (1, 4, 7, 10) and x >= 14) or m in (2, 5, 8, 11) or (m in (3, 6, 9, 12) and x <= 5)


def fetch(k):
    for attempt in range(4):
        try:
            raw = subprocess.run(['curl', '-s', '--max-time', '30', '-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '-H', 'Accept: application/json',
                                  f"https://api.nasdaq.com/api/calendar/earnings?date={k}"], capture_output=True, timeout=45).stdout
            j = json.loads(raw.decode('utf-8', 'replace')); rows = ((j.get('data') or {}).get('rows')) or []
            with LOCK:
                with open(OUT, 'a') as fh:
                    for x in rows:
                        if x.get('symbol') in SYMS: fh.write(f"{k},{x['symbol']},{x.get('time', '')}\n")
                with open(DONE, 'a') as fh: fh.write(k + '\n')
            return True
        except Exception:
            time.sleep(2 + 3 * attempt)
    return False


def main():
    days = [f"{d:%Y-%m-%d}" for d in pd.bdate_range('2010-07-01', '2026-11-13') if in_window(d)]
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    if not os.path.exists(OUT): open(OUT, 'w').write('date,symbol,time\n')
    todo = [k for k in days if k not in done]
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(fetch, todo))
    bad = [k for k, ok in zip(todo, res) if not ok]
    print(f"{len(days)} days in the reporting windows, fetched now {sum(res)}, failed {len(bad)}: {', '.join(bad[:10])}")
    if not bad: open(os.path.join(WEB, 'earn_calendar_done.txt'), 'w').write('complete')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
