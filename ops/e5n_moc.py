r"""Free forward record of the daily closing-auction (MOC) imbalance as published by FinancialJuice at about 15:50 ET (S&P 500,
Nasdaq 100, Dow 30, Mag 7, $ millions; + = buy). Windows task "E5n MOC recorder" starts it on weekdays at 15:49 ET; it watches the
public RSS feed until the day's "MOC Imbalance" item appears or 15:58 passes.
    python e5n_moc.py            one watch (the task)
    python e5n_moc.py page <id>  read one item page by id and print the parsed numbers (test)
"""
import os, re, sys, time, json, datetime as dt, urllib.request
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e5n_paths as PATHS

RSS = 'https://www.financialjuice.com/feed.ashx?xy=rss'; PAGE = 'https://www.financialjuice.com/News/{}/MOC-Imbalance.aspx'
OUT = os.path.join(PATHS.WEB, 'moc_financialjuice.csv'); LOG = os.path.join(PATHS.WEB, 'moc_financialjuice_log.txt')
HEAD = 'date_et,published_utc,item_id,spx_mln,ndx_mln,dow_mln,mag7_mln,recorded_et'
UA = {'User-Agent': 'Mozilla/5.0'}
KEYS = (('spx', r'S&(?:amp;)?P 500'), ('ndx', r'Nasdaq 100'), ('dow', r'Dow 30'), ('mag7', r'Mag(?:nificent)? 7'))


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r: return r.read().decode('utf-8', 'replace')


def parse_page(html):
    out = {}
    for k, pat in KEYS:
        m = re.search(pat + r'\s*:\s*([+-]?[\d,]+(?:\.\d+)?)\s*(mln|bln|million|billion)?', html, re.I)
        if m:
            v = float(m.group(1).replace(',', '')); out[k] = v * 1000 if (m.group(2) or '').lower().startswith('b') else v
    return out


def items():
    x = get(RSS); res = []
    for it in re.findall(r'<item>(.*?)</item>', x, re.S):
        t = re.search(r'<title>(.*?)</title>', it, re.S); g = re.search(r'<guid[^>]*>(\d+)</guid>', it); p = re.search(r'<pubDate>(.*?)</pubDate>', it)
        if t and g and p and re.search(r'\bMOC\b.*imbalance', t.group(1), re.I):
            res.append((g.group(1), dt.datetime.strptime(p.group(1).strip(), '%a, %d %b %Y %H:%M:%S GMT')))
    return res


def known():
    if not os.path.exists(OUT): return set()
    return {ln.split(',')[2] for ln in open(OUT, encoding='utf-8').read().splitlines()[1:] if ln.count(',') >= 2}


def record(item_id, pub, now):
    v = parse_page(get(PAGE.format(item_id)))
    if 'ndx' not in v: return False
    new = not os.path.exists(OUT)
    with open(OUT, 'a', encoding='utf-8') as f:
        if new: f.write(HEAD + '\n')
        f.write(f"{now:%Y-%m-%d},{pub:%Y-%m-%d %H:%M:%S},{item_id}," + ','.join('' if v.get(k) is None else f"{v[k]:g}" for k, _ in KEYS) + f",{now:%Y-%m-%d %H:%M:%S}\n")
    return True


def watch():
    os.makedirs(PATHS.WEB, exist_ok=True); start = dt.datetime.now(); end = start.replace(hour=15, minute=58, second=0); msg = 'no MOC item by 15:58'
    while True:
        now = dt.datetime.now()
        try:
            have = known()
            for item_id, pub in items():
                if item_id in have or (now - (pub - dt.timedelta(hours=5))).days > 0: continue
                if record(item_id, pub, now): msg = f"recorded item {item_id} published {pub:%H:%M:%S} UTC"; break
            if msg.startswith('recorded'): break
        except Exception as e:
            msg = f"error {type(e).__name__}: {str(e)[:120]}"
        if now >= end: break
        time.sleep(20)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | {msg}\n")


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'page': print(json.dumps(parse_page(get(PAGE.format(sys.argv[2])))))
    elif len(sys.argv) > 1: print(__doc__)
    else: watch()
