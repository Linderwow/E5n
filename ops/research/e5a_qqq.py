r"""E5A A2 (E5A_PLAN.txt): every QQQ trade since 2016 from the stock data service's free plan -> per-minute order-flow totals.
    python e5a_qqq.py fetch [all]     download the sampled sessions (every 3rd + the 66 pilot sessions; 'all' = every session), resumable
    python e5a_qqq.py build           join the session files into cache\e5a_qqq_minutes.parquet
"""
import os, sys, json, gzip, time, threading, urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS
import e5s_alpaca as AL

DIR = PATHS.QQQ_TRADES; OUT = os.path.join(PATHS.STOCKS, 'qqq_trades_minutes.parquet'); LOG = os.path.join(HERE, 'e5a_qqq_log.txt')
URL = 'https://data.alpaca.markets/v2/stocks/QQQ/trades'; PACE = 0.31; LOCK = threading.Lock(); _last = [0.0]; OKC = {'@', 'F', 'I', ' ', ''}
COLS = ['buy', 'sell', 'lbuy', 'lsell', 'wbuy', 'wsell', 'rbuy', 'rsell', 'n', 'shares', 'first', 'last']; BIG = 150000.0
PILOT = ('2026-01-02', '2026-04-09'); FIRST = '2016-01-04'; LAST = '2026-10-15'


def log(s):
    with LOCK:
        with open(LOG, 'a', encoding='utf-8') as fh: fh.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {s}\n")
    print(s, flush=True)


def sessions(everything=False):
    d = pd.DatetimeIndex(np.load(os.path.join(PATHS.CACHE, 'e5c_dates.npy'), allow_pickle=True)); pos = np.arange(len(d))
    m = (d >= FIRST) & (d <= LAST); pil = m & (d >= PILOT[0]) & (d <= PILOT[1]); smp = m & (pos % 3 == 0)
    pick = m if everything else (pil | smp)
    return [f"{x:%Y-%m-%d}" for x in d[pil]] + [f"{x:%Y-%m-%d}" for x in d[pick & ~pil]]


def page(params, k, s):
    with LOCK:
        wait = PACE - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
    req = urllib.request.Request(URL + '?' + urllib.parse.urlencode(params), headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s, 'accept': 'application/json', 'Accept-Encoding': 'gzip'})
    for n in range(10):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                raw = r.read()
                if r.headers.get('Content-Encoding', '') == 'gzip': raw = gzip.decompress(raw)
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403): raise SystemExit(f"the key was refused (HTTP {e.code})")
            time.sleep(25 if e.code == 429 else min(60, 2 ** n))
        except Exception:
            time.sleep(min(60, 2 ** n))
    raise RuntimeError('no answer after 10 tries')


def one(day, k, s):
    f = os.path.join(DIR, day + '.npy')
    if os.path.exists(f): return 0
    a = pd.Timestamp(day + ' 09:30', tz='America/New_York'); b = pd.Timestamp(day + ' 16:00', tz='America/New_York'); a_ns = a.value
    params = dict(start=a.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), end=b.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), limit=10000, feed='sip', sort='asc')
    A = np.zeros((390, len(COLS))); A[:, 10:] = np.nan; lastp = np.nan; lastsign = 0.0; total = 0
    while True:
        j = page(params, k, s); tr = j.get('trades') or []
        if tr:
            total += len(tr); ok = np.fromiter((set(x.get('c') or ['@']) <= OKC for x in tr), bool, len(tr))
            p = np.fromiter((x['p'] for x in tr), float, len(tr)); sz = np.fromiter((x['s'] for x in tr), float, len(tr))
            dk = np.fromiter((x['x'] == 'D' for x in tr), bool, len(tr)); sw = np.fromiter(('F' in (x.get('c') or ()) for x in tr), bool, len(tr))
            t = pd.to_datetime([x['t'] for x in tr], utc=True, format='ISO8601').as_unit('ns').asi8; mi = ((t - a_ns) // 60_000_000_000).astype(np.int64)
            inn = (mi >= 0) & (mi < 390) & ok; lit = inn & ~dk
            if lit.any():
                pl = p[lit]; sl = sz[lit]; ml = mi[lit]; d = np.sign(np.diff(np.r_[lastp, pl])); d[~np.isfinite(d)] = 0.0
                if d[0] == 0: d[0] = lastsign
                idx = np.where(d != 0, np.arange(len(d)), 0); np.maximum.accumulate(idx, out=idx); sg = d[idx]
                lastp = pl[-1]; lastsign = sg[-1]; val = pl * sl; big = val >= BIG; w = sw[lit]
                for c, v in ((0, sl * (sg > 0)), (1, sl * (sg < 0)), (2, val * big * (sg > 0)), (3, val * big * (sg < 0)), (4, sl * w * (sg > 0)), (5, sl * w * (sg < 0)), (8, np.ones(len(sl))), (9, sl)):
                    A[:, c] += np.bincount(ml, weights=v, minlength=390)
                fi = np.r_[True, ml[1:] != ml[:-1]]; la = np.r_[ml[1:] != ml[:-1], True]
                new = np.isnan(A[ml[fi], 10]); A[ml[fi][new], 10] = pl[fi][new]; A[ml[la], 11] = pl[la]
            dd = inn & dk
            if dd.any():
                fr = (p[dd] * 100.0) % 1.0; sd = sz[dd]; md = mi[dd]
                A[:, 6] += np.bincount(md, weights=sd * ((fr > 0.6) & (fr < 0.9999)), minlength=390); A[:, 7] += np.bincount(md, weights=sd * ((fr > 0.0001) & (fr < 0.4)), minlength=390)
        tok = j.get('next_page_token')
        if not tok: break
        params['page_token'] = tok
    tmp = f + '.tmp'
    with open(tmp, 'wb') as fh: np.save(fh, A)
    os.replace(tmp, f); return total


def fetch(everything=False):
    k, s = AL.keys()
    if not (k and s): print('no key found'); return 1
    os.makedirs(DIR, exist_ok=True); days = sessions(everything); todo = [d for d in days if not os.path.exists(os.path.join(DIR, d + '.npy'))]; t0 = time.time(); done = [0, 0, 0]
    log(f"fetch {'all' if everything else 'sample'}: {len(days)} sessions wanted, {len(todo)} to download")

    def job(d):
        try: n = one(d, k, s)
        except SystemExit: raise
        except Exception as e:
            with LOCK: done[2] += 1
            log(f"   {d} failed: {str(e)[:120]}"); return
        with LOCK:
            done[0] += 1; done[1] += n; c = done[0]
        if c % 25 == 0 or c == len(todo): log(f"   {c} of {len(todo)} sessions, {done[1] / 1e6:,.1f} million trades, {time.time() - t0:.0f} s")

    with ThreadPoolExecutor(6) as ex: list(ex.map(job, todo))
    log(f"finished: {done[0]} sessions, {done[1] / 1e6:,.1f} million trades, failed {done[2]}, {time.time() - t0:.0f} s"); return 0 if done[2] == 0 else 1


def build():
    parts = []
    for f in sorted(os.listdir(DIR)):
        if not f.endswith('.npy'): continue
        A = np.load(os.path.join(DIR, f)); df = pd.DataFrame(A, columns=COLS); df.insert(0, 'm', pd.Timestamp(f[:10] + ' 09:30') + pd.to_timedelta(np.arange(390), unit='m')); parts.append(df)
    g = pd.concat(parts, ignore_index=True); g.to_parquet(OUT, index=False)
    print(f"{len(parts)} sessions, {len(g):,} minutes, {g.n.sum() / 1e6:,.1f} million exchange trades -> {OUT}")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'fetch': sys.exit(fetch(len(sys.argv) > 2 and sys.argv[2] == 'all'))
    elif a == 'build': build()
    else: print(__doc__)
