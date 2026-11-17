r"""UT (UT_PLAN.txt): pre / post-holiday sessions (T1, T2) and decision timing on NinjaTrader's tick store (T3).
    python ut_tests.py holiday
    python ut_tests.py timing
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import datetime as dt
import numpy as np, pandas as pd
import e5n_calendar as CAL
import red_rules as R

NT = os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8'); sys.path.insert(0, os.path.join(NT, 'agent_exports', 'review', 'research'))
TICKS = os.path.join(NT, 'db', 'tick'); DLOG = os.path.join(NT, 'e5n', 'logs', 'E5n_decisions_MNQ_Backtest.csv')
OUT = os.path.join(HERE, 'UT_RESULTS.txt'); LINES = []; NUSD = 55750.0; RT = 3.57; EPOCH = 621355968000000000


def say(s=''):
    print(s, flush=True); LINES.append(s)


def flush():
    with open(OUT, 'a', encoding='utf-8') as f: f.write('\n'.join(LINES) + '\n')


def sessions(name):
    d = R.load(name); md = d['md']; t = d['tday']
    df = pd.DataFrame({'tday': t, 'md': md, 'o': d['o'], 'c': d['c']})
    o = df[df.md == 570].groupby('tday').o.first(); c = df[df.md == 959].groupby('tday').c.last()
    s = pd.concat([o, c], axis=1).dropna(); s.index = pd.to_datetime(s.index.values * 86400, unit='s').normalize()
    s['ret'] = (s.c / s.o - 1) * NUSD - RT; s['gross'] = (s.c / s.o - 1) * NUSD
    return s


def nyse_holidays(y):
    out = set(CAL.closed_days(y)) | {CAL._nth_weekday(y, 1, 0, 3), CAL._nth_weekday(y, 2, 0, 3), CAL._last_weekday(y, 5, 0), CAL._observed(dt.date(y, 7, 4)),
                                     CAL._nth_weekday(y, 9, 0, 1), CAL._nth_weekday(y, 11, 3, 4)}
    if y >= 2022: out.add(CAL._observed(dt.date(y, 6, 19)))
    return out


def tags(idx):
    closed = set()
    for y in range(idx.min().year - 1, idx.max().year + 2): closed |= {pd.Timestamp(x) for x in nyse_holidays(y)}
    full = np.array([CAL.full_session(x.date()) for x in idx])
    pre = np.array([any(x + pd.Timedelta(days=k) in closed for k in (1, 3) if (x + pd.Timedelta(days=k)).weekday() < 5 and all((x + pd.Timedelta(days=j)).weekday() >= 5 for j in range(1, k))) for x in idx])
    post = np.array([any(x - pd.Timedelta(days=k) in closed for k in (1, 3) if (x - pd.Timedelta(days=k)).weekday() < 5 and all((x - pd.Timedelta(days=j)).weekday() >= 5 for j in range(1, k))) for x in idx])
    return pre & full, post & full


def grade(x, lab, ctrl=None):
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan
    yrs = x.groupby(x.index.year).sum(); d5 = x.sum() - x.nlargest(5).sum()
    c = f" | minus all other sessions ${x.mean() - ctrl.mean():+.1f}" if ctrl is not None else ''
    say(f"   {lab:42s} n {len(x):4d}  ${x.mean():+7.1f} a session  t {t:+5.2f}  losing years {int((yrs < 0).sum())}/{len(yrs)}  drop-5 ${d5:+,.0f}{c}")
    return t, int((yrs < 0).sum()), len(yrs), d5


def holiday():
    say(); say("=" * 150); say(f"UT T1 / T2  PRE- AND POST-HOLIDAY SESSIONS  ut_tests.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  long 09:30 -> 16:00, $ at MNQ 27,875, $3.57 a round trip"); say("=" * 150)
    nq = sessions('nq'); nq = nq[(nq.index >= '2010-07-01') & (nq.index <= '2019-05-03')]
    pre, post = tags(nq.index); ok = []
    say(f"   DISCOVERY NQ {nq.index.min():%Y-%m-%d} .. {nq.index.max():%Y-%m-%d}, {len(nq):,} sessions")
    for lab, m in (('T1 pre-holiday', pre), ('T2 post-holiday', post)):
        t, ly, ny, d5 = grade(nq.ret[m], lab, nq.ret[~(pre | post)]); grade(nq.gross[m], f'{lab} (zero cost)')
        alive = t >= 3.0 and ly <= ny / 2 and d5 > 0; say(f"      -> {'SURVIVES discovery' if alive else 'DEAD (kill rule)'}")
        if alive: ok.append((lab, m))
    for lab, _ in ok:
        for nm in ('mnqx', 'es'):
            s = sessions(nm); s = s[s.index >= ('2019-05-06' if nm == 'mnqx' else '2010-07-01')]; p, q = tags(s.index)
            grade(s.ret[p if lab.startswith('T1') else q], f"{lab} on {nm}")
    say(f"   trial count 10,488"); flush()


def one_day(args):
    import ncd_reader as RD, ncd_export as E
    files = args
    ts, px = [], []
    for p in files:
        try: d, meta = RD.decode_tick_bytes(E._read_shared(p))
        except Exception: continue
        lt = d['local_ticks']
        if len(lt): ts.append((lt - EPOCH) // 10_000); px.append(d['price_ticks'] * meta['tick_size'])
    if not ts: return None
    ms = np.concatenate(ts); pr = np.concatenate(px); o = np.argsort(ms, kind='stable'); return ms[o], pr[o]


def timing():
    t0 = time.time(); dlog = pd.read_csv(DLOG, usecols=['date', 'time_et', 'net'])
    dlog = dlog[dlog.date >= '2025-10-01'].copy(); dlog['prev'] = dlog.groupby('date').net.shift(1).fillna(0); orders = dlog[(dlog.net != dlog.prev) & (dlog.time_et < '16:00')].copy()
    orders['dir'] = np.sign(orders.net - orders.prev); orders['T'] = pd.to_datetime(orders.date + ' ' + orders.time_et)
    jobs = {}
    for inst in sorted(os.listdir(TICKS)):
        if not inst.startswith('MNQ '): continue
        for fn in os.listdir(os.path.join(TICKS, inst)):
            if fn.endswith('.Last.ncd') and fn[:8] >= '20251001': jobs.setdefault((inst, fn[:8]), []).append(os.path.join(TICKS, inst, fn))
    days = sorted({d for _, d in jobs}); res = []; D = (10, 5, 2, 1, 0.5, 0); dd = (0.2, 0.4, 0.7, 1, 2)
    keys = sorted(jobs)
    with ProcessPoolExecutor(8) as ex: data = dict(zip(keys, ex.map(one_day, [sorted(jobs[k]) for k in keys])))
    for day in days:
        best = None
        for inst in {i for i, d in keys if d == day}:
            v = data.get((inst, day))
            if v is not None and (best is None or len(v[0]) > len(best[0])): best = v
        if best is None: continue
        ms, pr = best; dstr = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        for r in orders[orders.date == dstr].itertuples():
            T = int((r.T - pd.Timestamp('1970-01-01')).total_seconds() * 1000)
            at = lambda t_: pr[np.searchsorted(ms, t_, side='right') - 1] if np.searchsorted(ms, t_, side='right') > 0 else np.nan
            row = {'day': dstr, 'dir': r.dir}
            for a in D: row[f'b{a}'] = at(T - int(a * 1000))
            for b in dd: row[f'a{b}'] = at(T + int(b * 1000))
            res.append(row)
    X = pd.DataFrame(res).dropna()
    say(); say("=" * 150); say(f"UT T3  DECISION TIMING  ut_tests.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n orders (NinjaTrader decision log) on NinjaTrader's MNQ tick store"); say("=" * 150)
    say(f"   {len(X):,} orders on {X.day.nunique()} sessions ({X.day.min()} .. {X.day.max()}); move IN the order's direction, ticks (positive = paying more)")
    def sess_t(v):
        g = pd.Series(v.values, index=X.day.values).groupby(level=0).mean(); return g.mean() / g.std() * np.sqrt(len(g)) if g.std() > 0 else np.nan
    for b in (0.2, 0.4, 0.7, 1, 2):
        v = X.dir * (X[f'a{b}'] - X['b0']) / 0.25; say(f"   bar close -> +{b}s: {v.mean():+.2f} ticks (sessions t {sess_t(v):+.2f})")
    say("   deciding D seconds before the close, filled then, vs filled 0.4 s after the close (saving if positive) | stability = price at T-D within 1 tick of the close")
    for a in (10, 5, 2, 1, 0.5):
        v = X.dir * (X['a0.4'] - X[f'b{a}']) / 0.25; stab = (np.abs(X[f'b{a}'] - X['b0']) <= 0.25).mean(); ok = v.mean() >= 1.0 and sess_t(v) >= 3.0 and stab >= 0.90
        say(f"   D = {a:>4}s: saving {v.mean():+.2f} ticks a side (sessions t {sess_t(v):+.2f}), stability {stab:.0%} -> {'ADOPT CANDIDATE' if ok else 'no'}")
    say(f"run {time.time() - t0:.0f} s"); flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'holiday': holiday()
    elif a == 'timing': timing()
    else: print(__doc__)
