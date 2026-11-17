r"""FIX (FIX_PLAN.txt): scheduled benchmark flows - currency 4 pm London fix, gold fixes, Treasury month-end, crude settlement.
    python fix_flows.py
"""
import os, sys, math, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_paths as P

RES = os.path.join(HERE, 'FIX_RESULTS.txt'); LINES = []
SPEC = {'m6e': (12500.0, 0.0001, 0.61), 'mgc': (10.0, 0.1, 0.61), 'zn': (1000.0, 1 / 64, 2.00), 'mcl': (100.0, 0.01, 0.61)}
SPLIT = {'m6e': ('2019-07-01', '2022-12-31', '2023-01-01', '2026-10-15'), 'mgc': ('2019-07-01', '2022-12-31', '2023-01-01', '2026-10-15'),
         'zn': ('2019-07-01', '2022-12-31', '2023-01-01', '2026-10-15'), 'mcl': ('2021-07-12', '2023-12-31', '2024-01-01', '2026-10-15')}


def say(s=''):
    print(s, flush=True); LINES.append(s)


class Mk:
    def __init__(self, name):
        z = np.load(P.bars_1m(name)); self.t = z['t'].astype('datetime64[m]').astype(np.int64); self.o = z['o'].astype(float); self.c = z['c'].astype(float)
        self.iid = z['iid']; self.name = name; self.pos = {int(v): i for i, v in enumerate(self.t)}
        self.days = pd.DatetimeIndex(np.unique((self.t // 1440) * 86400 * 10 ** 9).astype('datetime64[ns]'))

    def at(self, ts):
        i = self.pos.get(int(ts))
        return i


def ny_minute(date, hh, mm, tz):
    x = pd.Timestamp(f"{date:%Y-%m-%d} {hh:02d}:{mm:02d}", tz=tz).tz_convert('America/New_York').tz_localize(None)
    return int(x.value // 60_000_000_000)


def trade(m, sig_from, sig_to, entry, exit_, side_fn):
    a, b, e, x = (m.at(v) for v in (sig_from, sig_to, entry, exit_))
    if None in (a, b, e, x) or not (m.iid[a] == m.iid[b] == m.iid[e] == m.iid[x]): return None
    s = side_fn(m.c[b] - m.o[a])
    if s == 0: return None
    return s * (m.o[x] - m.o[e])


def fixed(m, entry, exit_, side):
    e, x = m.at(entry), m.at(exit_)
    if None in (e, x) or m.iid[e] != m.iid[x]: return None
    return side * (m.o[x] - m.o[e])


def last_days(days):
    s = pd.Series(days, index=days); g = s.groupby([days.year, days.month])
    return set(g.max()), set(g.apply(lambda v: v.iloc[-2] if len(v) > 1 else pd.NaT).dropna())


def tstat(x):
    s = x.std(); return float(x.mean() / s * math.sqrt(len(x))) if s > 0 else 0.0


def main():
    t0 = time.time(); M = {k: Mk(k) for k in ('m6e', 'mgc', 'zn', 'mcl')}; mes = Mk('mes')
    mes_close = {}
    for d in mes.days:
        i = mes.at(int(pd.Timestamp(f"{d:%Y-%m-%d} 15:59").value // 60_000_000_000))
        if i is not None: mes_close[d] = mes.c[i]
    mc = pd.Series(mes_close).sort_index()
    rows = {k: [] for k in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7')}
    fade = lambda mv: -np.sign(mv)
    for d in M['m6e'].days:
        if d.weekday() >= 5: continue
        f = ny_minute(d, 16, 0, 'Europe/London'); r = trade(M['m6e'], f - 30, f - 1, f, f + 30, fade)
        if r is not None: rows['F1'].append((d, r))
    lastd, _ = last_days(M['m6e'].days[M['m6e'].days.weekday < 5])
    for d in sorted(lastd):
        prev = mc[mc.index < d]; pm = mc[mc.index < pd.Timestamp(d.year, d.month, 1)]
        if len(prev) == 0 or len(pm) == 0: continue
        side = 1.0 if prev.iloc[-1] > pm.iloc[-1] else -1.0
        r = fixed(M['m6e'], int(pd.Timestamp(f"{d:%Y-%m-%d} 09:30").value // 60_000_000_000), ny_minute(d, 16, 0, 'Europe/London'), side)
        if r is not None: rows['F2'].append((d, r))
    for d in M['mgc'].days:
        if d.weekday() >= 5: continue
        for code, hh, mm in (('F3', 15, 0), ('F4', 10, 30)):
            f = ny_minute(d, hh, mm, 'Europe/London'); r = trade(M['mgc'], f - 30, f - 1, f, f + 30, fade)
            if r is not None: rows[code].append((d, r))
    zd = M['zn'].days[M['zn'].days.weekday < 5]; lastz, prevz = last_days(zd)
    for code, sel in (('F5', lastz), ('F6', prevz)):
        for d in sorted(sel):
            r = fixed(M['zn'], int(pd.Timestamp(f"{d:%Y-%m-%d} 08:20").value // 60_000_000_000), int(pd.Timestamp(f"{d:%Y-%m-%d} 15:00").value // 60_000_000_000), 1.0)
            if r is not None: rows[code].append((d, r))
    for d in M['mcl'].days:
        if d.weekday() >= 5: continue
        base = int(pd.Timestamp(f"{d:%Y-%m-%d} 14:00").value // 60_000_000_000)
        r = trade(M['mcl'], base, base + 27, base + 31, base + 60, fade)
        if r is not None: rows['F7'].append((d, r))
    inst = {'F1': 'm6e', 'F2': 'm6e', 'F3': 'mgc', 'F4': 'mgc', 'F5': 'zn', 'F6': 'zn', 'F7': 'mcl'}
    names = {'F1': 'euro: fade the 30 minutes into the 4 pm London fix, 30 minutes after', 'F2': 'euro: month-end hedge flow, 09:30 -> fix, side from US stocks month-to-date',
             'F3': 'gold: fade into the 15:00 London PM fix', 'F4': 'gold: fade into the 10:30 London AM fix', 'F5': 'ZN long 08:20 -> 15:00 on the last day of the month',
             'F6': 'ZN long on the day before the last', 'F7': 'crude: fade 14:00 -> 14:28 after settlement, 14:31 -> 15:00'}
    say(); say("=" * 165); say(f"FIX  SCHEDULED BENCHMARK FLOWS  fix_flows.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (FIX_PLAN.txt) $ per 1 contract, cost 1 tick + commission a side"); say("=" * 165)
    surv = []
    for code, ev in rows.items():
        k = inst[code]; mult, tick, com = SPEC[k]; a0, a1, b0, b1 = SPLIT[k]; cost = 2 * (tick * mult + com)
        s = pd.Series({d: r for d, r in ev}).sort_index() * mult; days = M[k].days[M[k].days.weekday < 5]
        def part(x0, x1):
            dd = days[(days >= x0) & (days <= x1)]; x = s.reindex(dd).fillna(0.0); n = (s.index >= x0) & (s.index <= x1)
            net = x - cost * pd.Series(s.index.isin(dd), index=s.index).reindex(dd).fillna(0).astype(float) if False else x.copy()
            ev_ = s[(s.index >= x0) & (s.index <= x1)]; netd = (ev_ - cost).reindex(dd).fillna(0.0)
            yy = netd.groupby(netd.index.year).sum(); d5 = netd.copy(); d5[netd.nlargest(5).index] = 0
            return dict(n=int(len(ev_)), yr=len(ev_) / (len(dd) / 252), t=tstat(netd), t0=tstat(x), usd=float(netd.sum() / (len(dd) / 252)), win=float((ev_ > cost).mean()) if len(ev_) else np.nan,
                        lose=int((yy < 0).sum()), nyr=len(yy), d5=tstat(d5))
        A = part(a0, a1)
        ok = A['t'] >= 2.7 and A['lose'] <= A['nyr'] / 2
        say(f"   {code} {names[code]:78s} | discovery {A['n']:4d} events ({A['yr']:4.0f}/yr) t {A['t']:+5.2f} (zero cost {A['t0']:+5.2f}) ${A['usd']:+7,.0f}/yr, "
            f"winners {A['win']:.0%}, losing yrs {A['lose']}/{A['nyr']} -> {'PASSES DISCOVERY' if ok else 'dead'}")
        if ok: surv.append((code, part, b0, b1))
    say()
    if not surv: say("HOLDOUT: not opened - nothing passed discovery.")
    for code, part, b0, b1 in surv:
        B = part(b0, b1); okB = B['t'] >= 2.0 and B['d5'] >= 2.0
        say(f"   HOLDOUT {code}: {B['n']} events t {B['t']:+.2f} (zero cost {B['t0']:+.2f}) ${B['usd']:+,.0f}/yr, losing yrs {B['lose']}/{B['nyr']}, drop-5 t {B['d5']:+.2f} -> "
            f"{'CANDIDATE' if okB else 'dead'}")
    say(f"run {time.time() - t0:.0f} s")
    with open(RES, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    main()
