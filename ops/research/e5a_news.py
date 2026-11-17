r"""E5A A5 (E5A_PLAN.txt amendment 1): a news clock from free headlines since 2016 - does it add to E5n's 5-minute forecast?
    python e5a_news.py fetch     headline times for the 8 largest Nasdaq stocks and QQQ -> cache\e5a_news.parquet (resumable by year)
    python e5a_news.py run       the E5S column test with placebos
"""
import os, sys, json, time, urllib.request, urllib.error, urllib.parse
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

OUT = os.path.join(PATHS.WEB, 'news_times.parquet'); PARTS = os.path.join(PATHS.WEB, 'news_parts'); URL = 'https://data.alpaca.markets/v1beta1/news'
SYMS = 'AAPL,MSFT,NVDA,AMZN,GOOGL,GOOG,META,FB,AVGO,TSLA,QQQ'


def get(params, k, s):
    req = urllib.request.Request(URL + '?' + urllib.parse.urlencode(params), headers={'APCA-API-KEY-ID': k, 'APCA-API-SECRET-KEY': s, 'accept': 'application/json'})
    for n in range(10):
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403): raise SystemExit(f"refused (HTTP {e.code})")
            time.sleep(25 if e.code == 429 else min(60, 2 ** n))
        except Exception:
            time.sleep(min(60, 2 ** n))
    raise SystemExit('no answer after 10 tries - run again later')


def fetch():
    import e5s_alpaca as AL
    k, s = AL.keys(); os.makedirs(PARTS, exist_ok=True); t0 = time.time()
    for y in range(2016, 2027):
        f = os.path.join(PARTS, f'{y}.parquet')
        if os.path.exists(f): continue
        params = dict(symbols=SYMS, start=f'{y}-01-01T00:00:00Z', end=f'{y + 1}-01-01T00:00:00Z' if y < 2026 else '2026-10-16T00:00:00Z', limit=50, sort='asc', include_content='false'); rows = []; pages = 0
        while True:
            j = get(params, k, s); pages += 1
            for x in j.get('news') or []: rows.append((x['id'], x['created_at'], len(x.get('symbols') or [])))
            if not j.get('next_page_token'): break
            params['page_token'] = j['next_page_token']; time.sleep(0.45)
        df = pd.DataFrame(rows, columns=['id', 't', 'nsym']).drop_duplicates('id'); df.to_parquet(f, index=False)
        print(f"   {y}: {len(df):,} headlines, {pages} requests, {time.time() - t0:.0f} s", flush=True)
    a = pd.concat([pd.read_parquet(os.path.join(PARTS, f)) for f in sorted(os.listdir(PARTS))], ignore_index=True).drop_duplicates('id')
    a['m'] = pd.to_datetime(a.t, utc=True, format='ISO8601').dt.tz_convert('America/New_York').dt.tz_localize(None); a = a.drop(columns='t').sort_values('m').reset_index(drop=True); a.to_parquet(OUT, index=False)
    for f in os.listdir(PARTS): os.remove(os.path.join(PARTS, f))
    os.rmdir(PARTS); hm = a.m.dt.hour * 60 + a.m.dt.minute
    print(f"{len(a):,} headlines {a.m.min():%Y-%m-%d} .. {a.m.max():%Y-%m-%d}, {((hm >= 570) & (hm < 960)).mean():.0%} inside 09:30-16:00 ET -> {OUT}")


def run():
    import e5b_lab as L, e5s_lab as ES
    ES.OUT = os.path.join(HERE, 'E5A_RESULTS.txt'); L.setup(); now, opn = ES.times(); t = pd.read_parquet(OUT).m.values.astype('datetime64[s]'); w = now.astype('datetime64[s]')
    j = np.searchsorted(t, w, side='right'); n15 = j - np.searchsorted(t, w - np.timedelta64(15, 'm'), side='right'); n60 = j - np.searchsorted(t, w - np.timedelta64(60, 'm'), side='right')
    since = np.where(j > 0, (w - t[np.clip(j - 1, 0, None)]).astype('timedelta64[s]').astype(float) / 60.0, 240.0); since = np.minimum(since, 240.0)
    yr = L.G.T.date.dt.year.values; ok = yr >= 2016; fx = pd.DataFrame({'news15': np.where(ok, n15, np.nan), 'news60': np.where(ok, n60, np.nan), 'news_since': np.where(ok, since, np.nan)})
    L.say(f"A5 news clock: {len(t):,} headlines; per decision 2018-26: mean in the last 15 minutes {np.nanmean(fx.news15[yr >= 2018]):.2f}, in the last 60 {np.nanmean(fx.news60[yr >= 2018]):.2f}, "
          f"decisions with none in 60 minutes {np.mean(fx.news60[yr >= 2018] == 0):.0%}")
    ES.run('A5', 'news clock: headlines on the 8 largest Nasdaq stocks and QQQ - 3 columns', fx, first=2018, split=2022)


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'fetch': fetch()
    elif a == 'run': run()
    else: print(__doc__)
