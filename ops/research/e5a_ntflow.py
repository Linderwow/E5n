r"""E5A A1 + A7 (E5A_PLAN.txt): order-flow tests on NinjaTrader's own tick store (every trade with the trade-at-ask flag).
    python e5a_ntflow.py build [NQ]   read the tick store (shared read, nothing written there) -> cache\e5a_ntflow_minutes.parquet (MNQ) or
                                      cache\e5a_ntflow_nq_minutes.parquet (NQ)
    python e5a_ntflow.py run          A1: MNQ, the E5F pilot's measures + calibration against the bought trades of the pilot months
    python e5a_ntflow.py runnq        A7: the big contract's order flow and its difference from the micro's
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

NT = os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8'); RESEARCH = os.path.join(NT, 'agent_exports', 'review', 'research'); sys.path.insert(0, RESEARCH)
TICKS = os.path.join(NT, 'db', 'tick'); RES = os.path.join(HERE, 'E5A_RESULTS.txt')
OUTS = {'MNQ': os.path.join(PATHS.CACHE, 'e5a_ntflow_minutes.parquet'), 'NQ': os.path.join(PATHS.CACHE, 'e5a_ntflow_nq_minutes.parquet')}; OUT = OUTS['MNQ']
EPOCH = 621355968000000000; MIN = 600_000_000; PILOT = ('2026-01-02', '2026-04-09'); LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def one_file(path):
    import ncd_reader as R, ncd_export as E
    try:
        d, meta = R.decode_tick_bytes(E._read_shared(path))
    except Exception as e:
        return None, f"{os.path.basename(path)}: {str(e)[:80]}"
    lt = d['local_ticks']
    if len(lt) == 0: return None, None
    keep = ~R.is_sub_ms(lt); lt = lt[keep]; vol = d['volume'][keep].astype(float); ask = d['at_ask'][keep]; px = d['price_ticks'][keep] * meta['tick_size']
    if len(lt) == 0: return None, None
    mi = (lt - EPOCH) // MIN; ms = (lt - EPOCH) // 10_000
    new = np.r_[True, (ms[1:] != ms[:-1]) | (ask[1:] != ask[:-1])]; gid = np.cumsum(new) - 1; gv = np.bincount(gid, weights=vol)[gid]
    big = vol >= 10; gbig = gv >= 10; b = ask.astype(float); s = 1.0 - b
    df = pd.DataFrame({'mi': mi, 'buy': vol * b, 'sell': vol * s, 'lbuy': vol * b * big, 'lsell': vol * s * big, 'gbuy': vol * b * gbig, 'gsell': vol * s * gbig, 'px': px})
    g = df.groupby('mi', sort=True).agg(buy=('buy', 'sum'), sell=('sell', 'sum'), lbuy=('lbuy', 'sum'), lsell=('lsell', 'sum'), gbuy=('gbuy', 'sum'), gsell=('gsell', 'sum'), first=('px', 'first'), last=('px', 'last'))
    return g.reset_index(), None


def build(root='MNQ'):
    t0 = time.time(); jobs = []; now = time.time(); skipped = 0
    for inst in sorted(os.listdir(TICKS)):
        if not inst.startswith(root + ' '): continue
        for fn in sorted(os.listdir(os.path.join(TICKS, inst))):
            p = os.path.join(TICKS, inst, fn)
            if not fn.endswith('.Last.ncd'): continue
            if now - os.stat(p).st_mtime > 120: jobs.append((inst, p))
            else: skipped += 1
    print(f"{len(jobs)} hourly tick files in {len({i for i, _ in jobs})} contracts ({skipped} written in the last 2 minutes left out)", flush=True); parts = []; bad = []
    with ProcessPoolExecutor(8) as ex:
        for (inst, p), (g, err) in zip(jobs, ex.map(one_file, [p for _, p in jobs], chunksize=20)):
            if err: bad.append(err)
            if g is not None: g['contract'] = inst; parts.append(g)
    a = pd.concat(parts, ignore_index=True); a = a.groupby(['contract', 'mi'], sort=True).agg(buy=('buy', 'sum'), sell=('sell', 'sum'), lbuy=('lbuy', 'sum'), lsell=('lsell', 'sum'), gbuy=('gbuy', 'sum'), gsell=('gsell', 'sum'),
                                                                                             first=('first', 'first'), last=('last', 'last')).reset_index()
    a['m'] = pd.to_datetime(a.mi.values * 60, unit='s'); a['d'] = a.m.dt.normalize(); hm = a.m.dt.hour * 60 + a.m.dt.minute; rth = a[(hm >= 570) & (hm < 960)].copy(); rth['v'] = rth.buy + rth.sell
    dv = rth.groupby(['d', 'contract']).v.sum().reset_index().sort_values(['d', 'v']); front = dv.groupby('d').tail(1)[['d', 'contract']]
    out = rth.merge(front, on=['d', 'contract']).drop(columns=['mi', 'v']).sort_values('m').reset_index(drop=True); out.to_parquet(OUTS[root], index=False)
    per = out.groupby('d').size()
    print(f"{out.d.nunique()} sessions {out.d.min():%Y-%m-%d} .. {out.d.max():%Y-%m-%d}, {len(out):,} regular-session minutes (sessions with fewer than 380: {int((per < 380).sum())}), {int((out.buy + out.sell).sum()):,} contracts; "
          f"unreadable files {len(bad)} {bad[:3]}; {time.time() - t0:.0f} s -> {OUTS[root]}")


def run():
    import e5x_lab as X, e5f_pilot as P
    t0 = time.time(); X.setup(); g = pd.read_parquet(OUT).set_index('m'); g['net'] = g.buy - g.sell; g['ret'] = g['last'] - g['first']
    say(); say("=" * 160); say(f"E5A A1  THE ORDER-FLOW PILOT ON NINJATRADER'S TICK STORE  e5a_ntflow.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5A_PLAN.txt; information test only)"); say("=" * 160)
    san = float(g.net.corr(g.ret)); T = X.NQ.T; f = X.F_NQ; days = np.unique(g.d.values.astype('datetime64[D]'))
    sel = np.isin(T.date.values.astype('datetime64[D]'), days); T = T[sel].reset_index(drop=True); f = f[sel]; y = T.y.values; day = T.date.values.astype('datetime64[D]').astype(np.int64)
    ends = T.date.values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]')
    meas = {'OF1': P.window(g, ends, 1, 'buy', 'sell'), 'OF5': P.window(g, ends, 5, 'buy', 'sell'), 'OF15': P.window(g, ends, 15, 'buy', 'sell'), 'LT5': P.window(g, ends, 5, 'lbuy', 'lsell'), 'LG5': P.window(g, ends, 5, 'gbuy', 'gsell')}
    pil = (T.date.values >= np.datetime64(PILOT[0])) & (T.date.values <= np.datetime64(PILOT[1]))
    say(f"tick store: {len(days)} sessions {pd.Timestamp(days.min()):%Y-%m-%d} .. {pd.Timestamp(days.max()):%Y-%m-%d}; on the research grid (to 2026-10-15) {T.date.nunique()} sessions, {len(T):,} E5n decisions "
        f"({T.date[pil].nunique()} pilot sessions, {T.date[~pil].nunique()} new); sanity: buys minus sells vs the same minute's price change, correlation {san:+.2f} (must be clearly positive)")
    share = {k: float((g[a].sum() + g[b].sum()) / (g.buy.sum() + g.sell.sum())) for k, (a, b) in {'LT5': ('lbuy', 'lsell'), 'LG5': ('gbuy', 'gsell')}.items()}
    say(f"share of volume in prints of 10+ contracts {share['LT5']:.1%}; in orders of 10+ after joining same-millisecond prints {share['LG5']:.1%}")
    db_g = P.minute_flow(P.FILES['MNQ'][0]); db_g['net'] = db_g.buy - db_g.sell; j = g[['net']].join(db_g[['net']], how='inner', rsuffix='_db')
    lt_db = P.window(db_g, ends, 5, 'lbuy', 'lsell'); of_db = P.window(db_g, ends, 5, 'buy', 'sell')
    c = lambda a, b: float(np.corrcoef(a[np.isfinite(a) & np.isfinite(b) & pil], b[np.isfinite(a) & np.isfinite(b) & pil])[0, 1])
    say(f"calibration on the pilot sessions against the bought trades: minute net flow correlation {float(j.net.corr(j.net_db)):+.3f} ({len(j):,} minutes); at the decisions OF5 {c(meas['OF5'], of_db):+.3f}, "
        f"LT5 {c(meas['LT5'], lt_db):+.3f}, LG5 vs the bought LT5 {c(meas['LG5'], lt_db):+.3f}")
    res = {}
    for lab, m in (('NEW sessions (not in the pilot) - PRIMARY', ~pil), ('pilot sessions (the store, for comparison with E5F)', pil), ('all sessions', np.ones(len(T), bool))):
        say(f"{lab}: {T.date[m].nunique()} sessions, {int(m.sum()):,} decisions")
        for nm, x in meas.items():
            tt, n, ic = P.ols_t(y[m], f[m], x[m], day[m]); pl = [P.ols_t(y[m], f[m], pd.Series(x[m]).shift(75 * s).values, day[m])[0] for s in (1, 2, 3)]
            res[(lab[:3], nm)] = (tt, pl)
            say(f"   {nm:5s} t {tt:+.2f} beyond E5n's forecast (n {n:,}; plain correlation with the next 5 minutes {ic:+.4f}) | placebos (1 / 2 / 3 sessions earlier) t " + ' / '.join(f"{p:+.2f}" for p in pl))
    bar = {'LT5': 2.5, 'LG5': 2.8, 'OF1': 2.8, 'OF5': 2.8, 'OF15': 2.8}
    hit = [nm for nm in meas if res[('NEW', nm)][0] >= bar[nm] and all(abs(p) < 2.0 for p in res[('NEW', nm)][1])]
    say("VERDICT A1: " + (f"PASS - {', '.join(hit)} add information beyond E5n's forecast on sessions the pilot never saw" if hit else
                          "FAIL - no measure reaches its line on the new sessions (LT5 2.5, the others 2.8)"))
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


def runnq():
    import e5x_lab as X, e5f_pilot as P
    t0 = time.time(); X.setup(); g = pd.read_parquet(OUTS['NQ']).set_index('m'); gm = pd.read_parquet(OUTS['MNQ']).set_index('m'); g['net'] = g.buy - g.sell; g['ret'] = g['last'] - g['first']
    say(); say("=" * 160); say(f"E5A A7  ORDER FLOW OF THE BIG CONTRACT (NQ)  e5a_ntflow.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5A_PLAN.txt amendment 2; information test only)"); say("=" * 160)
    per = g.groupby('d').size(); full = per.index[per >= 380].values.astype('datetime64[D]'); days = np.intersect1d(full, np.unique(gm.d.values.astype('datetime64[D]')))
    san = float(g.net.corr(g.ret)); T = X.NQ.T; f = X.F_NQ; sel = np.isin(T.date.values.astype('datetime64[D]'), days); T = T[sel].reset_index(drop=True); f = f[sel]; y = T.y.values
    td = T.date.values.astype('datetime64[D]'); day = td.astype(np.int64); ends = T.date.values.astype('datetime64[m]') + (570 + T.dec.values).astype('timedelta64[m]')
    j = g[['net', 'last']].join(gm[['buy', 'sell', 'last']], how='inner', rsuffix='_m'); j['net_m'] = j.buy - j.sell
    vol_nq = float((g.buy + g.sell).sum() / g.d.nunique()); vol_m = float((gm.buy + gm.sell).sum() / gm.d.nunique())
    say(f"NQ store: {g.d.nunique()} sessions {g.d.min():%Y-%m-%d} .. {g.d.max():%Y-%m-%d} ({len(full)} with 380+ minutes); used = on the research grid and in both stores: {len(np.unique(td))} sessions, {len(T):,} E5n decisions")
    say(f"sanity: NQ buys minus sells vs the same minute's price change, correlation {san:+.2f}; NQ vs MNQ minute net flow {float(j.net.corr(j.net_m)):+.2f}; NQ price minus MNQ price, median {float((j['last'] - j.last_m).median()):+.2f} points; "
        f"contracts a session NQ {vol_nq:,.0f} (= {vol_nq * 10:,.0f} micros), MNQ {vol_m:,.0f}")
    sh = {k: float((g[a].sum() + g[b].sum()) / (g.buy.sum() + g.sell.sum())) for k, (a, b) in {'NLT5': ('lbuy', 'lsell'), 'NLG5': ('gbuy', 'gsell')}.items()}
    say(f"share of NQ volume in prints of 10+ contracts {sh['NLT5']:.1%}; in orders of 10+ after joining same-millisecond prints {sh['NLG5']:.1%}")
    nof5 = P.window(g, ends, 5, 'buy', 'sell')
    x = {'NOF1': P.window(g, ends, 1, 'buy', 'sell'), 'NOF5': nof5, 'NOF15': P.window(g, ends, 15, 'buy', 'sell'), 'NLT5': P.window(g, ends, 5, 'lbuy', 'lsell'), 'NLG5': P.window(g, ends, 5, 'gbuy', 'gsell'),
         'NDV5': nof5 - P.window(gm, ends, 5, 'buy', 'sell')}
    mid = np.unique(td)[len(np.unique(td)) // 2]; parts = (('all sessions', np.ones(len(T), bool)), (f'first half (to {pd.Timestamp(mid):%Y-%m-%d})', td < mid), ('second half', td >= mid)); res = {}
    for lab, m in parts:
        say(f"{lab}: {len(np.unique(td[m]))} sessions, {int(m.sum()):,} decisions")
        for k in x:
            tt, n, ic = P.ols_t(y[m], f[m], x[k][m], day[m]); pl = [P.ols_t(y[m], f[m], pd.Series(x[k][m]).shift(75 * s).values, day[m])[0] for s in (1, 2, 3)]; res[(lab, k)] = (tt, pl)
            say(f"   {k:5s} t {tt:+.2f} beyond E5n's forecast (n {n:,}; plain correlation with the next 5 minutes {ic:+.4f}) | placebos (1 / 2 / 3 sessions earlier) t " + ' / '.join(f"{p:+.2f}" for p in pl))
    a, h1, h2 = (p[0] for p in parts); hit = []
    for k in x:
        t_a, t_1, t_2 = res[(a, k)][0], res[(h1, k)][0], res[(h2, k)][0]
        if abs(t_a) >= 2.8 and np.sign(t_1) == np.sign(t_a) == np.sign(t_2) and min(abs(t_1), abs(t_2)) >= 1.5 and all(abs(p) < 2.0 for p in res[(a, k)][1]): hit.append(k)
    say("VERDICT A7: " + (f"PASS - {', '.join(hit)}: start a nightly shadow" if hit else "FAIL - no measure reaches |t| 2.8 with both halves at 1.5 and clean placebos"))
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build': build(sys.argv[2] if len(sys.argv) > 2 else 'MNQ')
    elif a == 'run': run()
    elif a == 'runnq': runnq()
    else: print(__doc__)
