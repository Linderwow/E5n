r"""E5A A6 (E5A_PLAN.txt amendment 1): QQQ best bid / ask size imbalance in the 10 seconds before each E5n decision, every 6th session since 2016.
    python e5a_quotes.py fetch     -> cache\e5a_qi10.parquet (resumable by session)
    python e5a_quotes.py run       information test
"""
import os, sys, json, gzip, time, threading, urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

OUT = os.path.join(PATHS.STOCKS, 'qqq_quote_imbalance.parquet'); DIR = os.path.join(PATHS.STOCKS, 'qqq_quote_parts'); RES = os.path.join(HERE, 'E5A_RESULTS.txt'); URL = 'https://data.alpaca.markets/v2/stocks/QQQ/quotes'
FIRST = '2016-01-04'; LAST = '2026-10-15'; PACE = 0.31; LOCK = threading.Lock(); _last = [0.0]; DECS = list(range(15, 390, 5)); LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def sessions():
    d = pd.DatetimeIndex(np.load(os.path.join(PATHS.CACHE, 'e5c_dates.npy'), allow_pickle=True)); pos = np.arange(len(d))
    return [f"{x:%Y-%m-%d}" for x in d[(d >= FIRST) & (d <= LAST) & (pos % 6 == 0)]]


def get(params, k, s):
    with LOCK:
        wait = PACE - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
    req = urllib.request.Request(URL + '?' + urllib.parse.urlencode(params), headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s, 'accept': 'application/json', 'Accept-Encoding': 'gzip'})
    for n in range(10):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get('Content-Encoding', '') == 'gzip': raw = gzip.decompress(raw)
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403): raise SystemExit(f"refused (HTTP {e.code})")
            time.sleep(25 if e.code == 429 else min(60, 2 ** n))
        except Exception:
            time.sleep(min(60, 2 ** n))
    raise RuntimeError('no answer after 10 tries')


def one(day, k, s):
    f = os.path.join(DIR, day + '.npy')
    if os.path.exists(f): return
    out = np.full((len(DECS), 2), np.nan)
    for i, dec in enumerate(DECS):
        t = pd.Timestamp(day + ' 09:30', tz='America/New_York') + pd.Timedelta(minutes=dec); a = (t - pd.Timedelta(seconds=10)).tz_convert('UTC'); b = t.tz_convert('UTC')
        j = get(dict(start=a.strftime('%Y-%m-%dT%H:%M:%SZ'), end=(b - pd.Timedelta(milliseconds=1)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), limit=10000, feed='sip', sort='asc'), k, s); q = j.get('quotes') or []
        if q:
            bs = np.fromiter((x['bs'] for x in q), float, len(q)); az = np.fromiter((x['as'] for x in q), float, len(q)); ok = (bs + az) > 0
            if ok.any(): out[i] = float(np.mean((bs[ok] - az[ok]) / (bs[ok] + az[ok]))), float(ok.sum())
    tmp = f + '.tmp'
    with open(tmp, 'wb') as fh: np.save(fh, out)
    os.replace(tmp, f)


def fetch():
    import e5s_alpaca as AL
    k, s = AL.keys(); os.makedirs(DIR, exist_ok=True); days = sessions(); todo = [d for d in days if not os.path.exists(os.path.join(DIR, d + '.npy'))]; t0 = time.time(); done = [0, 0]
    print(f"{len(days)} sessions wanted, {len(todo)} to download ({len(DECS)} requests each)", flush=True)

    def job(d):
        try: one(d, k, s)
        except SystemExit: raise
        except Exception as e:
            with LOCK: done[1] += 1
            print(f"   {d} failed: {str(e)[:100]}", flush=True); return
        with LOCK:
            done[0] += 1; c = done[0]
        if c % 25 == 0 or c == len(todo): print(f"   {c} of {len(todo)} sessions, {time.time() - t0:.0f} s", flush=True)

    with ThreadPoolExecutor(6) as ex: list(ex.map(job, todo))
    if done[1]: print(f"failed sessions {done[1]} - run again"); return 1
    rows = []
    for d in days:
        A = np.load(os.path.join(DIR, d + '.npy')); rows.append(pd.DataFrame({'date': pd.Timestamp(d), 'dec': DECS, 'qi10': A[:, 0], 'quotes': A[:, 1]}))
    a = pd.concat(rows, ignore_index=True); a.to_parquet(OUT, index=False)
    for f in os.listdir(DIR): os.remove(os.path.join(DIR, f))
    os.rmdir(DIR); print(f"{a.date.nunique()} sessions, {len(a):,} decisions, {a.qi10.notna().mean():.1%} with quotes, median quotes in 10 seconds {a.quotes.median():.0f} -> {OUT}"); return 0


def run():
    import e5x_lab as X, e5f_pilot as P
    t0 = time.time(); X.setup(); q = pd.read_parquet(OUT); T = X.NQ.T; f = X.F_NQ
    M = T[['date', 'dec']].merge(q, on=['date', 'dec'], how='left'); sel = np.isin(T.date.values, q.date.unique()); x = M.qi10.values[sel]; T = T[sel].reset_index(drop=True); f = f[sel]
    td = T.date.values.astype('datetime64[D]'); y = T.y.values; day = td.astype(np.int64)
    say(); say("=" * 160); say(f"E5A A6  QQQ QUOTE IMBALANCE  e5a_quotes.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5A_PLAN.txt amendment 1; information test only)"); say("=" * 160)
    cut = np.datetime64('2022-01-01'); parts = (('all sampled sessions 2016-2026', np.ones(len(T), bool)), ('2016-2021', td < cut), ('2022-2026', td >= cut)); res = {}
    say(f"{len(np.unique(td))} sessions (every 6th), {len(T):,} decisions, {np.isfinite(x).mean():.1%} with quotes; average imbalance {np.nanmean(x):+.3f}, standard deviation {np.nanstd(x):.3f}")
    for lab, m in parts:
        tt, n, ic = P.ols_t(y[m], f[m], x[m], day[m]); pl = [P.ols_t(y[m], f[m], pd.Series(x[m]).shift(75 * s).values, day[m])[0] for s in (1, 2, 3)]; res[lab] = (tt, pl)
        say(f"{lab}: {len(np.unique(td[m]))} sessions | QI10 t {tt:+.2f} beyond E5n's forecast (n {n:,}; plain correlation with the next 5 minutes {ic:+.4f}) | placebos (1 / 2 / 3 sampled sessions earlier) t " + ' / '.join(f"{p:+.2f}" for p in pl))
    a, h1, h2 = (p[0] for p in parts); t_a, t_1, t_2 = res[a][0], res[h1][0], res[h2][0]
    ok = abs(t_a) >= 3.0 and np.sign(t_1) == np.sign(t_a) == np.sign(t_2) and min(abs(t_1), abs(t_2)) >= 2.0 and all(abs(p) < 2.0 for p in res[a][1])
    say("VERDICT A6: " + ("PASS - quote imbalance adds information: download every session and test as a model column" if ok else "FAIL - |t| 3.0 with both halves at 2.0 and clean placebos not reached"))
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'fetch': sys.exit(fetch())
    elif a == 'run': run()
    else: print(__doc__)
