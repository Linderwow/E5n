r"""Free one-minute history of US stocks and ETFs from Dukascopy's public datafeed (no account, no key): per-day BID_candles_min_1.bi5
files (LZMA; 24-byte big-endian records: seconds from 00:00 UTC, open, close, low, high as int32 x 1000, volume float32).
    python e5s_duka.py probe                 which candidate symbols exist, and since when
    python e5s_duka.py fetch [SYM ...]       download 2019-01-02 .. yesterday, 09:30-16:00 ET only -> cache\stocks\<SYM>_<year>.parquet (resumable;
                                             default = the 8 largest Nasdaq-100 stocks; one request at a time, backing off when the feed refuses)
"""
import os, sys, lzma, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

OUT = os.path.join(PATHS.STOCKS, 'dukascopy')
URL = "https://datafeed.dukascopy.com/datafeed/{s}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
           'Referer': 'https://www.dukascopy.com/swiss/english/marketwatch/historical/'}
STOCKS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'FB', 'TSLA', 'AVGO', 'COST', 'NFLX', 'ADBE', 'PEP', 'CSCO', 'AMD', 'INTC', 'QCOM', 'TXN',
          'CMCSA', 'AMGN', 'INTU', 'PYPL', 'SBUX', 'BKNG', 'GILD', 'MDLZ', 'ADP', 'ISRG', 'MU', 'AMAT', 'LRCX', 'ASML', 'PLTR', 'TMUS', 'HON', 'LIN']
ETFS = ['QQQ', 'SPY', 'IWM', 'DIA', 'TLT', 'GLD', 'USO', 'XLK', 'SMH', 'VXX', 'XLF', 'XLE']
START = '2019-01-02'
MEGA = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'FB', 'AVGO', 'TSLA']
FAILED = []


def code(sym):
    return sym + 'USUSD'


PACE = [2.0, 0.0]


def raw_day(sym, day, tries=8):
    u = URL.format(s=code(sym), y=day.year, m=day.month - 1, d=day.day); cool = 30.0
    for k in range(tries):
        wait = PACE[0] - (time.time() - PACE[1])
        if wait > 0: time.sleep(wait)
        PACE[1] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=HEADERS), timeout=40) as r: b = r.read()
            PACE[0] = max(1.5, PACE[0] * 0.97)
            return lzma.decompress(b) if b else b''
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            PACE[0] = min(8.0, PACE[0] * 1.5); time.sleep(cool); cool = min(900.0, cool * 2)
        except Exception:
            time.sleep(cool); cool = min(900.0, cool * 2)
    FAILED.append((sym, str(day.date())))
    return b''


def parse(b, day):
    n = len(b) // 24
    if n == 0: return None
    a = np.frombuffer(b[:n * 24], dtype='>i4').reshape(n, 6); vol = np.frombuffer(b[:n * 24], dtype='>f4').reshape(n, 6)[:, 5]
    t = (pd.Timestamp(day, tz='UTC') + pd.to_timedelta(a[:, 0], unit='s')).tz_convert('America/New_York').tz_localize(None)
    df = pd.DataFrame({'time': t, 'open': a[:, 1] / 1000.0, 'close': a[:, 2] / 1000.0, 'low': a[:, 3] / 1000.0, 'high': a[:, 4] / 1000.0, 'volume': vol.astype(float)})
    m = df.time.dt.hour * 60 + df.time.dt.minute
    return df[(m >= 570) & (m < 960) & (df.volume > 0)]


def probe():
    for s in STOCKS + ETFS:
        hit = None
        for y in (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024):
            b = raw_day(s, pd.Timestamp(f'{y}-06-13') if y != 2020 else pd.Timestamp('2020-06-12'), tries=4)
            if b:
                d = parse(b, pd.Timestamp(f'{y}-06-13') if y != 2020 else pd.Timestamp('2020-06-12'))
                if d is not None and len(d) > 100: hit = (y, len(d), float(d.close.iloc[-1])); break
        print(f"{s:6s} " + (f"from {hit[0]} or earlier: {hit[1]} regular-session minutes that day, close {hit[2]:.2f}" if hit else "not available"), flush=True)


def fetch(syms):
    os.makedirs(OUT, exist_ok=True); end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    for s in syms:
        for y in range(pd.Timestamp(START).year, end.year + 1):
            path = os.path.join(OUT, f'{s}_{y}.parquet')
            if os.path.exists(path): continue
            days = [d for d in pd.date_range(max(pd.Timestamp(START), pd.Timestamp(f'{y}-01-01')), min(end, pd.Timestamp(f'{y}-12-31'))) if d.weekday() < 5]
            t0 = time.time(); before = len(FAILED)
            parts = []
            for n_, d in enumerate(days, 1):
                parts.append((d, raw_day(s, d)))
                if n_ % 60 == 0: print(f"   {s} {y}: {n_} of {len(days)} day files, {time.time() - t0:.0f} s, pace {PACE[0]:.1f} s", flush=True)
            frames = [f for f in (parse(x, d) for d, x in parts if x) if f is not None and len(f)]
            absent = sum(1 for _, x in parts if x is None); failed = len(FAILED) - before
            if failed > 0.2 * len(days):
                print(f"{s} {y}: {failed} of {len(days)} day files failed - not saved, will be retried on the next run", flush=True); time.sleep(120); continue
            df = pd.concat(frames, ignore_index=True).sort_values('time').drop_duplicates('time') if frames else pd.DataFrame(columns=['time', 'open', 'close', 'low', 'high', 'volume'])
            if y < end.year or len(df): df.to_parquet(path, index=False)
            print(f"{s} {y}: {len(df):,} minutes on {df.time.dt.normalize().nunique() if len(df) else 0} days, {absent} days absent, {failed} failed, {time.time() - t0:.0f} s", flush=True)
    print(f"finished; day files that failed after retries: {len(FAILED)}", flush=True)


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'probe': probe()
    elif a == 'fetch': fetch(sys.argv[2:] or MEGA)
    else: print(__doc__)
