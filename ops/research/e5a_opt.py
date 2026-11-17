r"""E5A A4 (E5A_PLAN.txt amendment 1): same-day QQQ option flow since 2024-02 (free option bars) - does it add to E5n's 5-minute forecast?
    python e5a_opt.py fetch     5-minute bars of the options expiring that day, 31 strikes around the 09:30 price -> cache\e5a_opt_5m.parquet
    python e5a_opt.py run       information test
"""
import os, sys, glob, json, time, urllib.request, urllib.error, urllib.parse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

OUT = os.path.join(PATHS.STOCKS, 'qqq_options_0dte_5m.parquet'); RES = os.path.join(HERE, 'E5A_RESULTS.txt'); URL = 'https://data.alpaca.markets/v1beta1/options/bars'
FIRST = '2024-02-05'; LAST = '2026-10-15'; LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def get(params, k, s):
    req = urllib.request.Request(URL + '?' + urllib.parse.urlencode(params), headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s, 'accept': 'application/json'})
    for n in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403): raise SystemExit(f"refused (HTTP {e.code})")
            if e.code in (400, 422): return {}
            time.sleep(25 if e.code == 429 else min(60, 2 ** n))
        except Exception:
            time.sleep(min(60, 2 ** n))
    return {}


def fetch():
    import e5s_alpaca as AL, e5s_lab as ES
    k, s = AL.keys(); q = pd.concat([pd.read_parquet(f, columns=['time', 'open']) for f in sorted(glob.glob(os.path.join(ES.STOCKS, 'QQQ_*.parquet')))], ignore_index=True)
    q['time'] = pd.to_datetime(q.time); o = q.sort_values('time').groupby(q.time.dt.normalize()).open.first()
    grid = pd.DatetimeIndex(np.load(os.path.join(PATHS.CACHE, 'e5c_dates.npy'), allow_pickle=True)); days = [d for d in grid if pd.Timestamp(FIRST) <= d <= pd.Timestamp(LAST) and d in o.index]
    rows = []; t0 = time.time(); empty = 0
    for i, d in enumerate(days):
        c = int(round(float(o[d]))); syms = [f"QQQ{d:%y%m%d}{cp}{(c + j) * 1000:08d}" for j in range(-15, 16) for cp in 'CP']
        a = pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz='America/New_York'); b = pd.Timestamp(f"{d:%Y-%m-%d} 16:00", tz='America/New_York')
        params = dict(symbols=','.join(syms), timeframe='5Min', start=a.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), end=b.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), limit=10000); n0 = len(rows)
        while True:
            j = get(params, k, s)
            for sym, bars in (j.get('bars') or {}).items():
                for x in bars: rows.append((d, x['t'], sym[9], int(sym[10:]) / 1000.0, float(x['v']), float(x.get('vw', x['c'])), float(x.get('n', 0))))
            if not j.get('next_page_token'): break
            params['page_token'] = j['next_page_token']; time.sleep(0.6)
        empty += len(rows) == n0; time.sleep(0.6)
        if (i + 1) % 100 == 0: print(f"   {i + 1} of {len(days)} sessions, {len(rows):,} bars, {empty} sessions without an expiry that day, {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows, columns=['date', 't', 'cp', 'strike', 'v', 'vw', 'n'])
    df['m'] = pd.to_datetime(df.t, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None); df = df.drop(columns='t'); df.to_parquet(OUT, index=False)
    print(f"{len(days)} sessions asked, {df.date.nunique()} with same-day options, {len(df):,} bars, {df.v.sum() / 1e6:,.1f} million contracts -> {OUT}")


def run():
    import e5x_lab as X, e5f_pilot as P
    t0 = time.time(); X.setup(); df = pd.read_parquet(OUT); df['prem'] = df.v * df.vw * 100.0; hm = df.m.dt.hour * 60 + df.m.dt.minute; df = df[(hm >= 570) & (hm < 960)]
    g = df.pivot_table(index='m', columns='cp', values=['prem', 'v'], aggfunc='sum').fillna(0.0); g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.sort_index()
    have = np.unique(g.index.normalize().values.astype('datetime64[D]'))
    say(); say("=" * 160); say(f"E5A A4  SAME-DAY QQQ OPTION FLOW  e5a_opt.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5A_PLAN.txt amendment 1; information test only)"); say("=" * 160)
    T = X.NQ.T; f = X.F_NQ; td = T.date.values.astype('datetime64[D]'); sel = np.isin(td, have); T = T[sel].reset_index(drop=True); f = f[sel]; td = td[sel]; y = T.y.values; day = td.astype(np.int64)
    ends = T.date.values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]')
    x = {'OPC15': P.window(g, ends, 15, 'prem_C', 'prem_P'), 'OPCO': P.window(g, ends, 400, 'prem_C', 'prem_P'), 'OVC15': P.window(g, ends, 15, 'v_C', 'v_P')}
    r15 = T.r15.values if 'r15' in T.columns else np.full(len(T), np.nan); c = np.isfinite(x['OPC15']) & np.isfinite(r15)
    say(f"{len(have)} sessions with same-day options {pd.Timestamp(have.min()):%Y-%m-%d} .. {pd.Timestamp(have.max()):%Y-%m-%d}, {df.v.sum() / 1e6:,.0f} million contracts, {len(T):,} E5n decisions; "
        f"sanity: call-minus-put premium share over 15 minutes vs the index's own last 15 minutes, correlation {np.corrcoef(x['OPC15'][c], r15[c])[0, 1]:+.2f}; calls' share of premium {g.prem_C.sum() / (g.prem_C.sum() + g.prem_P.sum()):.1%}")
    cut = np.datetime64('2025-07-01'); parts = (('all sessions', np.ones(len(T), bool)), ('2024-02 .. 2025-06', td < cut), ('2025-07 .. 2026-10', td >= cut)); res = {}
    for lab, m in parts:
        say(f"{lab}: {len(np.unique(td[m]))} sessions, {int(m.sum()):,} decisions")
        for k in x:
            tt, n, ic = P.ols_t(y[m], f[m], x[k][m], day[m]); pl = [P.ols_t(y[m], f[m], pd.Series(x[k][m]).shift(75 * s).values, day[m])[0] for s in (1, 2, 3)]; res[(lab, k)] = (tt, pl)
            say(f"   {k:6s} t {tt:+.2f} beyond E5n's forecast (n {n:,}; plain correlation with the next 5 minutes {ic:+.4f}) | placebos (1 / 2 / 3 sessions earlier) t " + ' / '.join(f"{p:+.2f}" for p in pl))
    a, h1, h2 = (p[0] for p in parts); hit = []
    for k in x:
        t_a, t_1, t_2 = res[(a, k)][0], res[(h1, k)][0], res[(h2, k)][0]
        if abs(t_a) >= 3.0 and np.sign(t_1) == np.sign(t_a) == np.sign(t_2) and min(abs(t_1), abs(t_2)) >= 2.0 and all(abs(p) < 2.0 for p in res[(a, k)][1]): hit.append(k)
    say("VERDICT A4: " + (f"PASS - {', '.join(hit)}: start a nightly shadow" if hit else "FAIL - no measure reaches |t| 3.0 with both halves at 2.0 and clean placebos"))
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'fetch': fetch()
    elif a == 'run': run()
    else: print(__doc__)
