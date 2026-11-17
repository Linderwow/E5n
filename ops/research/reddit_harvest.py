r"""Reddit research harvest through the Arctic Shift public archive (reddit.com itself refuses automated reading).
    python reddit_harvest.py posts       every post of the full subreddits, title-filtered posts of the large ones -> data\web\reddit\posts_<sub>.jsonl
    python reddit_harvest.py pick [N]    threads about index futures with rules / tests / results, ranked -> picked.csv (default 700)
    python reddit_harvest.py comments    comment trees of the picked threads -> threads\<id>.json (resumable)
    python reddit_harvest.py check       completeness (counts vs the archive's own totals), readability, order
    python reddit_harvest.py text        one readable text file per thread -> text\<rank>_<id>.txt
"""
import os, sys, json, time, re, urllib.request, urllib.parse, urllib.error
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import e5n_paths as P

OUT = os.path.join(P.WEB, 'reddit'); API = 'https://arctic-shift.photon-reddit.com/api/'; UA = {'User-Agent': 'Mozilla/5.0 research-client'}
FULL = ('algotrading', 'FuturesTrading', 'RealDayTrading', 'ninjatrader', 'Topstep', 'propfirm', 'FuturesTradingPros', 'NQ_Futures', 'ES_Futures')
TITLED = {'Daytrading': ('NQ', 'MNQ', 'ES ', 'MES', 'futures', 'nasdaq', 'NASDAQ', 'S&P', 'strategy', 'backtest', 'algo', 'edge'),
          'quant': ('futures', 'intraday', 'NQ', 'ES ', 'index', 'backtest'), 'thewallstreet': ('NQ', 'ES ', 'futures', 'strategy'),
          'Trading': ('NQ', 'MNQ', 'futures', 'nasdaq', 'strategy', 'backtest', 'edge'), 'tradingview': ('NQ', 'ES ', 'futures', 'strategy', 'backtest')}
DONE_TITLED = {'Daytrading': ('NQ', 'MNQ')}
FIELDS = 'id,title,score,num_comments,created_utc,selftext,author,link_flair_text,url'


def get(path, params, tries=10):
    url = API + path + '?' + urllib.parse.urlencode(params)
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r: d = json.load(r)
            if isinstance(d, dict) and d.get('error') is None and isinstance(d.get('data'), list): return d
            time.sleep(min(600, 30 * 2 ** a)); continue
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace')[:200]
            if e.code in (400, 422) and 'imeout' not in body and 'slow down' not in body: raise SystemExit(f"request refused ({e.code}): {body} {params}")
            time.sleep(min(600, 30 * 2 ** a))
        except Exception:
            time.sleep(min(600, 30 * 2 ** a))
    return None


def page_all(sub, title=None, year=None):
    fn = os.path.join(OUT, f"posts_{sub}{'_' + re.sub('[^A-Za-z]', '', title) if title else ''}{'_' + str(year) if year else ''}.jsonl"); before = None; n = 0
    floor = int(time.mktime((year, 1, 1, 0, 0, 0, 0, 0, 0))) if year else None
    if year: before = int(time.mktime((year + 1, 1, 1, 0, 0, 0, 0, 0, 0)))
    if os.path.exists(fn) and os.path.exists(fn + '.done'): return
    if os.path.exists(fn):
        good = []
        for line in open(fn, encoding='utf-8'):
            try: good.append(json.loads(line))
            except Exception: break
        if good: before = min(r['created_utc'] for r in good); n = len(good)
        with open(fn, 'w', encoding='utf-8') as fh:
            for r in good: fh.write(json.dumps(r) + chr(10))
    with open(fn, 'a', encoding='utf-8') as fh:
        while True:
            p = dict(subreddit=sub, limit=100, sort='desc', fields=FIELDS)
            if title: p['title'] = title
            if before: p['before'] = before
            if floor: p['after'] = floor
            d = get('posts/search', p, tries=3)
            if d is None: p['limit'] = 25; d = get('posts/search', p, tries=10)
            if d is None: print(f"{sub} {title}: gave up after retries at {before}", flush=True); return
            rows = d.get('data') or []
            if not rows: break
            for r in rows: r['sub'] = sub; fh.write(json.dumps(r) + '\n')
            n += len(rows); last = min(r['created_utc'] for r in rows)
            if before is not None and last >= before: break
            before = last; time.sleep(2.0)
    oldest = time.strftime('%Y-%m-%d', time.gmtime(before)) if before else '-'
    open(fn + '.done', 'w').write(json.dumps(dict(posts=n, oldest=oldest, finished=time.strftime('%Y-%m-%d %H:%M'))))
    print(f"{sub} {title or 'all'}: {n:,} posts back to {oldest}", flush=True)


def posts():
    os.makedirs(OUT, exist_ok=True)
    for s in FULL: page_all(s)
    for s, ts in TITLED.items():
        for t in ts:
            if s in DONE_TITLED and t in DONE_TITLED[s]: page_all(s, t)


KW_MKT = re.compile(r"\b(NQ|MNQ|ES|MES|nasdaq|nas100|us100|futures|s&p|spx|index futures|e-?mini|micros?)\b", re.I)
KW_SUB = re.compile(r"(strateg|backtest|edge|system|algo|rules|results|profitab|statistic|tested|win ?rate|sharpe|profit factor|funded|payout|track record|journal|setup|model)", re.I)


def pick(N=700):
    rows = {}
    for f in os.listdir(OUT):
        if f.startswith('posts_') and f.endswith('.jsonl'):
            for line in open(os.path.join(OUT, f), encoding='utf-8'):
                r = json.loads(line); rows[r['id']] = r
    keep = []
    for r in rows.values():
        txt = f"{r.get('title') or ''} {r.get('selftext') or ''}"
        if r.get('selftext') in ('[removed]', '[deleted]') and (r.get('num_comments') or 0) < 10: continue
        if not (KW_MKT.search(txt) and KW_SUB.search(txt)): continue
        sc = (r.get('score') or 0) + 2 * (r.get('num_comments') or 0) + (40 if len(r.get('selftext') or '') > 1500 else 0)
        keep.append((sc, r))
    keep.sort(key=lambda x: -x[0]); top = keep[:N]
    import csv
    with open(os.path.join(OUT, 'picked.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['rank', 'id', 'sub', 'score', 'num_comments', 'created', 'title'])
        for i, (sc, r) in enumerate(top): w.writerow([i + 1, r['id'], r['sub'], r.get('score'), r.get('num_comments'), time.strftime('%Y-%m-%d', time.gmtime(r['created_utc'])), (r.get('title') or '')[:200]])
    print(f"{len(rows):,} posts harvested, {len(keep):,} about index futures with rules / tests / results, {len(top)} picked")


def comments():
    import csv
    td = os.path.join(OUT, 'threads'); os.makedirs(td, exist_ok=True)
    ids = [r['id'] for r in csv.DictReader(open(os.path.join(OUT, 'picked.csv'), encoding='utf-8'))]; done = 0
    for i in ids:
        fn = os.path.join(td, i + '.json')
        if os.path.exists(fn): done += 1; continue
        d = get('comments/tree', dict(link_id=i, limit=2000))
        if d is None: print('FAIL', i, flush=True); continue
        tmp = fn + '.tmp'; json.dump(d['data'], open(tmp, 'w', encoding='utf-8')); os.replace(tmp, fn); done += 1
        if done % 25 == 0: print(f"{done} of {len(ids)} threads", flush=True)
        time.sleep(2.0)
    print(f"comment trees: {done} of {len(ids)}")


def flat(nodes, depth=0, out=None):
    out = [] if out is None else out
    for n in nodes or []:
        d = n.get('data', n) if isinstance(n, dict) else {}
        body = d.get('body')
        if body and body not in ('[removed]', '[deleted]'): out.append((d.get('score') or 0, depth, d.get('author'), body))
        kids = d.get('replies') or n.get('children') if isinstance(n, dict) else None
        if isinstance(kids, dict): kids = kids.get('data', {}).get('children')
        if kids: flat(kids, depth + 1, out)
    return out


def text():
    import csv
    rows = {}
    for f in os.listdir(OUT):
        if f.startswith('posts_') and f.endswith('.jsonl'):
            for line in open(os.path.join(OUT, f), encoding='utf-8'): r = json.loads(line); rows[r['id']] = r
    td = os.path.join(OUT, 'threads'); xd = os.path.join(OUT, 'text'); os.makedirs(xd, exist_ok=True); n = 0
    for r in csv.DictReader(open(os.path.join(OUT, 'picked.csv'), encoding='utf-8')):
        p = rows.get(r['id'], {}); fn = os.path.join(td, r['id'] + '.json')
        cm = flat(json.load(open(fn, encoding='utf-8'))) if os.path.exists(fn) else []
        cm.sort(key=lambda x: -x[0])
        with open(os.path.join(xd, f"{int(r['rank']):04d}_{r['id']}.txt"), 'w', encoding='utf-8') as fh:
            fh.write(f"r/{r['sub']}  score {r['score']}  comments {r['num_comments']}  {r['created']}  https://www.reddit.com/comments/{r['id']}\n")
            fh.write(f"TITLE: {p.get('title')}\nAUTHOR: {p.get('author')}\n\n{p.get('selftext') or ''}\n\n--- COMMENTS (by score) ---\n")
            for sc, dp, au, body in cm[:120]: fh.write(f"[{sc}] {au}: {body.strip()}\n\n")
        n += 1
    print(f"{n} thread files in {xd}")


def check():
    import csv
    meta = {}
    for s in FULL:
        d = get('subreddits/search', dict(subreddit=s))
        m = ((d or {}).get('data') or [{}])[0].get('_meta', {}) if d else {}
        meta[s] = m.get('num_posts'); time.sleep(2.0)
    bad = 0; print(f"{'file':42s} {'lines':>8s} {'unique':>8s} {'json errors':>11s} {'order ok':>8s} {'oldest':>10s} {'archive total':>13s}  verdict")
    for f in sorted(os.listdir(OUT)):
        if not (f.startswith('posts_') and f.endswith('.jsonl')): continue
        ids, ts, err = [], [], 0
        for line in open(os.path.join(OUT, f), encoding='utf-8'):
            try: r = json.loads(line); ids.append(r['id']); ts.append(r['created_utc'])
            except Exception: err += 1
        done = os.path.exists(os.path.join(OUT, f + '.done')); sub = f[6:-6].split('_')[0] if f[6:-6] not in FULL else f[6:-6]
        order = all(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)); uniq = len(set(ids)); total = meta.get(f[6:-6])
        complete = done and err == 0 and (total is None or uniq >= 0.97 * total)
        bad += not complete
        print(f"{f:42s} {len(ids):8,d} {uniq:8,d} {err:11d} {str(order):>8s} {time.strftime('%Y-%m-%d', time.gmtime(min(ts))) if ts else '-':>10s} {str(total or '-'):>13s}  {'OK' if complete else ('RUNNING / INCOMPLETE' if not done else 'SHORT')}")
    td = os.path.join(OUT, 'threads')
    if os.path.isdir(td):
        fs = [x for x in os.listdir(td) if x.endswith('.json')]; berr = 0
        for x in fs:
            try: json.load(open(os.path.join(td, x), encoding='utf-8'))
            except Exception: berr += 1
        print(f"comment trees: {len(fs)} files, {berr} unreadable")
        bad += berr
    print('VERDICT:', 'ALL COMPLETE AND READABLE' if bad == 0 else f'{bad} file(s) not complete yet or broken')


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'posts': posts()
    elif a == 'pick': pick(int(sys.argv[2]) if len(sys.argv) > 2 else 700)
    elif a == 'comments': comments()
    elif a == 'text': text()
    elif a == 'check': check()
    else: print(__doc__)
