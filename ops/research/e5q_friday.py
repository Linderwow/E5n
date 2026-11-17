r"""Friday 2026-11-13: E5n's band-leg trades on NinjaTrader's own bars - entry, best open profit, exit (1 micro).
    python e5q_friday.py [YYYY-MM-DD]
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
import e5n_daily as D
import e5n_band as B

day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2026-11-13')
a, keep = D.all_bars(); cont, offs = D.continuous(a, keep)
marks = []; B.run(B.five(cont), live=True, warmup=D.WARMUP, bars=marks)
mk = pd.concat(marks, ignore_index=True); mk = mk[mk.time.dt.normalize() == day].reset_index(drop=True)
m1 = cont[pd.DatetimeIndex(cont.end).normalize() == day].copy(); m1['t'] = pd.DatetimeIndex(m1.end)
prev = 0.0; trades = []
for i, r in mk.iterrows():
    if r.target != prev:
        if prev != 0: trades[-1]['exit_t'] = r.time
        if r.target != 0: trades.append(dict(side=int(np.sign(r.target)), entry_t=r.time))
        prev = r.target
for tr in trades:
    xt = tr.get('exit_t', mk.time.iloc[-1] + pd.Timedelta(minutes=5))
    seg = m1[(m1.t > tr['entry_t']) & (m1.t <= xt)]
    if not len(seg): continue
    after = m1[m1.t > xt]
    ent = seg.o.iloc[0]; ex = after.o.iloc[0] if len(after) else seg.c.iloc[-1]; s = tr['side']
    best = (seg.h.max() - ent) if s > 0 else (ent - seg.l.min()); fin = s * (ex - ent)
    bt = seg.t.iloc[int(np.argmax(seg.h.values) if s > 0 else np.argmin(seg.l.values))]
    print(f"{'LONG ' if s > 0 else 'SHORT'} signal {tr['entry_t']:%H:%M}, filled ~{ent:,.2f}; best open profit {best:+.2f} pts (${best * 2:+,.0f} a micro) at {bt:%H:%M}; "
          f"exit signal {xt:%H:%M} ~{ex:,.2f}: {fin:+.2f} pts (${fin * 2:+,.0f}); gave back {best - fin:.2f} pts (${(best - fin) * 2:,.0f})")
print(f"session range {m1.l.min():,.2f} .. {m1.h.max():,.2f}; 5-min ATR at 10:00 {mk.atr.iloc[min(6, len(mk) - 1)]:.1f} pts")
