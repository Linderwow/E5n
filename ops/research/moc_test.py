r"""MOC (MOC_PLAN.txt): NYSE closing-auction imbalance at 3:50 pm -> 15:51 open to 16:00 close on MNQ / NQ (MES / ES replication).
    python moc_test.py
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_paths as P
import e5n_calendar as CAL
import red_rules as R

SRC = os.path.join(P.WEB, 'moc', 'OrderImbalanceInflowOutflow_20261117.csv'); OUT = os.path.join(HERE, 'MOC_RESULTS.txt'); LINES = []
NUSD = 55750.0; RT = 3.57; DISC = ('2015-10-02', '2021-12-31'); HOLD = ('2022-01-03', '2026-11-16'); JOIN = pd.Timestamp('2019-05-06')


def say(s=''):
    print(s, flush=True); LINES.append(s)


def window(names):
    parts = []
    for nm, lo, hi in names:
        d = R.load(nm); df = pd.DataFrame({'tday': d['tday'], 'md': d['md'], 'o': d['o'], 'c': d['c']})
        o = df[df.md == 951].groupby('tday').o.first(); c = df[df.md == 959].groupby('tday').c.last(); s = pd.concat([o, c], axis=1).dropna()
        s.index = pd.to_datetime(s.index.values * 86400, unit='s').normalize(); parts.append(s[(s.index >= lo) & (s.index < hi)])
    s = pd.concat(parts); s = s[[CAL.full_session(x.date()) for x in s.index]]
    s['r'] = (s.c / s.o - 1) * NUSD; return s


def grade(x, lab):
    t = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan; yrs = x.groupby(x.index.year).sum()
    say(f"   {lab:44s} n {len(x):5d} ${x.mean():+6.2f} a day  t {t:+5.2f}  losing years {int((yrs < 0).sum())}/{len(yrs)}  drop-5 ${x.sum() - x.nlargest(5).sum():+,.0f}  total ${x.sum():+,.0f}")
    return t, int((yrs < 0).sum()), len(yrs), x.sum() - x.nlargest(5).sum()


def main():
    m = pd.read_csv(SRC, encoding='utf-8-sig'); m['Date'] = pd.to_datetime(m['Date']); N = m.set_index('Date')['Net Notional'].astype(float)
    med = N.abs().rolling(60, min_periods=40).median().shift(1)
    nq = window((('nq', '2015-01-01', JOIN), ('mnqx', JOIN, '2027-01-01'))); es = window((('es', '2015-01-01', JOIN), ('mes', JOIN, '2027-01-01')))
    say(); say("=" * 150); say(f"MOC  NYSE CLOSING IMBALANCE (3:50 PM) -> 15:51 OPEN TO 16:00 CLOSE  moc_test.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  $ at MNQ 27,875, $3.57 a round trip"); say("=" * 150)
    say(f"   imbalance days {len(N):,} ({N.index.min():%Y-%m-%d} .. {N.index.max():%Y-%m-%d}); net > 0 on {(N > 0).mean():.0%}; median |net| ${N.abs().median() / 1e6:,.0f} M")
    def pos(rule, idx):
        n = N.reindex(idx); s = np.sign(n)
        if rule == 'M2': s = np.where(n.abs() >= med.reindex(idx), s, 0.0)
        if rule == 'M3': s = -s
        return pd.Series(np.nan_to_num(s), index=idx)
    def pnl(px, rule, lo, hi, cost=True):
        p = px[(px.index >= lo) & (px.index <= hi)]; p = p[p.index.isin(N.index)]; s = pos(rule, p.index)
        return s * p.r - (np.abs(s) * RT if cost else 0.0)
    say(f"   DISCOVERY {DISC[0]} .. {DISC[1]} on NQ / MNQ (sessions with both data: {len(nq[(nq.index >= DISC[0]) & (nq.index <= DISC[1]) & nq.index.isin(N.index)]):,})")
    alive = []
    for rule, lab in (('M1', 'M1 continuation'), ('M2', 'M2 strong continuation (|net| >= 60-day median)'), ('M3', 'M3 fade')):
        t, ly, ny, d5 = grade(pnl(nq, rule, *DISC), lab); grade(pnl(nq, rule, *DISC, cost=False), f'{rule} zero cost')
        ok = t >= 2.5 and ly <= ny / 2 and d5 > 0; say(f"      -> {'SURVIVES discovery' if ok else 'DEAD (kill rule)'}")
        if ok: alive.append(rule)
    for rule in alive:
        say(f"   HOLDOUT {HOLD[0]} .. {HOLD[1]} for {rule}:"); th, *_ = grade(pnl(nq, rule, *HOLD), f'{rule} MNQ holdout'); te, *_ = grade(pnl(es, rule, *DISC), f'{rule} ES / MES discovery')
        grade(pnl(es, rule, *HOLD), f'{rule} ES / MES holdout')
    if not alive: say("   no survivor - holdout not opened")
    say("   trial count 10,522")
    with open(OUT, 'a', encoding='utf-8') as f: f.write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
