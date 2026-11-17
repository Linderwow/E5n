r"""Free one-minute history of the Nasdaq-100 stocks from Alpaca's market-data API (free plan: bars since 2016, 200 requests a minute,
not the latest 15 minutes). Needs the user's own free API key - never typed into a chat:
    environment variables APCA_API_KEY_ID and APCA_API_SECRET_KEY, or the file D:\nt_logs\alpaca_key.txt (line 1 key id, line 2 secret).
    python e5s_alpaca.py check              is the key accepted (one small request)
    python e5s_alpaca.py fetch [SYM ...]    download 5-minute bars 2016-01-04 .. yesterday, 09:30-16:00 ET -> cache\stocks_alpaca_5m\<SYM>_<year>.parquet (resumable)
"""
import os, sys, json, time, threading, urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

OUT = PATHS.STOCK_BARS; TF = '5Min'; KEYFILE = r'D:\nt_logs\alpaca_key.txt'; BASE = 'https://data.alpaca.markets/v2/stocks/bars'
START = '2016-01-04'; PACE = 0.33; LOCK = threading.Lock()
NOW = ('AAPL MSFT NVDA AMZN GOOGL GOOG META TSLA AVGO COST NFLX ADBE PEP CSCO AMD INTC QCOM TXN CMCSA AMGN INTU PYPL SBUX BKNG GILD MDLZ ADP ISRG MU AMAT LRCX ASML PLTR '
       'TMUS HON LIN PANW VRTX REGN KLAC SNPS CDNS MELI CRWD ABNB MAR ORLY CTAS FTNT MRVL ADSK NXPI PDD WDAY ROP MNST PAYX AEP CPRT PCAR ROST KDP CHTR FAST ODFL EA VRSK '
       'EXC XEL CTSH KHC BKR GEHC DDOG TEAM IDXX CSGP LULU ON FANG DXCM TTWO ZS CDW BIIB MCHP GFS WBD MDB ARM APP MSTR AXON TTD CEG DASH CCEP AZN').split()
PAST = ('ATVI SGEN WBA ILMN SIRI DLTR ENPH MRNA ZM DOCU LCID RIVN JD BIDU NTES EBAY ALGN SWKS MTCH OKTA SPLK VRSN INCY CHKP TCOM XLNX MXIM CERN ALXN CTXS FISV FOXA FOX CELG '
        'NLOK WYNN UAL AAL EXPE ULTA HAS MYL LBTYA NTAP WDC STX TSCO HSIC JBHT XRAY HOLX ANSS QQQ').split()
_last = [0.0]


def keys():
    k, s = os.environ.get('APCA_API_KEY_ID'), os.environ.get('APCA_API_SECRET_KEY')
    if not (k and s) and os.path.exists(KEYFILE):
        vals = []
        for x in open(KEYFILE, encoding='utf-8-sig').read().splitlines():
            x = x.strip()
            if '=' in x: x = x.split('=', 1)[1].strip()
            x = x.strip('"').strip("'").strip()
            if x: vals.append(x)
        if len(vals) >= 2: k, s = vals[0], vals[1]
    return k, s


def get(params, k, s):
    with LOCK:
        wait = PACE - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
    req = urllib.request.Request(BASE + '?' + urllib.parse.urlencode(params), headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s, 'accept': 'application/json'})
    for n in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                try: why = e.read().decode('utf-8', 'replace')[:200]
                except Exception: why = ''
                raise SystemExit(f"the data service refused the request (HTTP {e.code}): {why or 'no reason given'} - if it says the key is invalid, check the key id and secret; a subscription / time-range message means the request, not the key")
            if e.code == 422: return {'bars': {}, 'next_page_token': None}
            time.sleep(20 if e.code == 429 else min(60, 2 ** n))
        except Exception:
            time.sleep(min(60, 2 ** n))
    raise SystemExit('the service did not answer after 8 tries - run the same command again later; finished files are kept')


def year(sym, y, end, k, s):
    a = max(pd.Timestamp(START), pd.Timestamp(f'{y}-01-01')); b = min(end, pd.Timestamp(f'{y}-12-31'))
    params = dict(symbols=sym, timeframe=TF, start=a.strftime('%Y-%m-%dT00:00:00Z'), end=(b + pd.Timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z'), limit=10000,
                  adjustment='split', feed='sip', sort='asc'); rows = []
    while True:
        j = get(params, k, s); rows += (j.get('bars') or {}).get(sym, [])
        tok = j.get('next_page_token')
        if not tok: break
        params['page_token'] = tok
    if not rows: return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'trades', 'vwap'])
    d = pd.DataFrame(rows); t = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = pd.DataFrame({'time': t, 'open': d.o.astype(float), 'high': d.h.astype(float), 'low': d.l.astype(float), 'close': d.c.astype(float), 'volume': d.v.astype(float),
                       'trades': d.get('n', pd.Series(np.nan, index=d.index)).astype(float), 'vwap': d.get('vw', pd.Series(np.nan, index=d.index)).astype(float)})
    m = df.time.dt.hour * 60 + df.time.dt.minute
    return df[(m >= 570) & (m < 960)].reset_index(drop=True)


def check():
    k, s = keys()
    if not (k and s): print(f"no key found: set APCA_API_KEY_ID and APCA_API_SECRET_KEY, or write {KEYFILE} (key=... on line 1, secret=... on line 2)"); return False
    j = get(dict(symbols='AAPL', timeframe='1Min', start='2016-06-13T13:30:00Z', end='2016-06-13T20:00:00Z', limit=1000, adjustment='split', feed='sip', sort='asc'), k, s)
    n = len((j.get('bars') or {}).get('AAPL', [])); print(f"key accepted; 2016-06-13 AAPL: {n} one-minute bars" + ('' if n > 300 else ' - fewer than expected, the plan may not include this history'))
    return n > 0


def fetch(syms, workers=32):
    k, s = keys()
    if not (k and s): print(f"no key found: set APCA_API_KEY_ID and APCA_API_SECRET_KEY, or write {KEYFILE} (key=... on line 1, secret=... on line 2)"); return
    os.makedirs(OUT, exist_ok=True); end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1); t0 = time.time()
    jobs = [(sym, y) for sym in syms for y in range(pd.Timestamp(START).year, end.year + 1) if not os.path.exists(os.path.join(OUT, f'{sym}_{y}.parquet'))]
    print(f"{len(jobs)} stock-years to download ({TF} bars, 09:30-16:00 ET kept)", flush=True); done = [0, 0]

    def job(a):
        sym, y = a; df = year(sym, y, end, k, s)
        if y < end.year or len(df): df.to_parquet(os.path.join(OUT, f'{sym}_{y}.parquet'), index=False)
        with LOCK:
            done[0] += 1; done[1] += len(df); n = done[0]
        if n % 50 == 0 or n == len(jobs): print(f"   {n} of {len(jobs)} stock-years, {done[1]:,} bars, {time.time() - t0:.0f} s", flush=True)

    with ThreadPoolExecutor(workers) as ex: list(ex.map(job, jobs))
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e9
    print(f"finished: {done[1]:,} new bars, {len(os.listdir(OUT))} files, {size:.2f} GB in {OUT}", flush=True)


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'check': check()
    elif a == 'fetch': fetch(sys.argv[2:] or NOW + PAST)
    else: print(__doc__)
