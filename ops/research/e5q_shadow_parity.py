r"""E5Q: the nightly's target shadow (ops\e5n_target.py, NinjaTrader bars) against the research overlay Q2 on the common sessions, and
Friday 2026-11-13 under the rule.
    python e5q_shadow_parity.py
"""
import os, sys
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
import e5n_daily as D
import e5n_target as TG
from e5b_lab import G, say


def main():
    C.load(); M = C.M
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    ctx = NS(atr=slot('atr'), vw=slot('vw'), half=(slot('ub') - slot('lb')) / 2, mdl=np.zeros_like(M.band))
    p2 = Q.overlay(lambda S: (S.op >= 2.0 * S.W, np.zeros_like(S.mfe, bool)), ctx)
    res = pd.Series(((p2 - M.band) * M.dpx).sum(1), index=M.dates)
    a, keep = D.all_bars(); cont, offs = D.continuous(a, keep); bd, marks = D.band_days(cont); sh = TG.shadow(cont, marks)
    nt = pd.Series(sh.diff_pts.values, index=pd.to_datetime(sh.date))
    j = pd.DataFrame({'research': res, 'nightly': nt}).dropna(); j = j[j.index > L.JOIN]
    hit = j[(j.research != 0) | (j.nightly != 0)]; both = hit[(hit.research != 0) & (hit.nightly != 0)]
    say(); say("=" * 160)
    say(f"TARGET SHADOW PARITY  e5q_shadow_parity.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  nightly shadow (NinjaTrader bars) vs the research overlay Q2, {len(j):,} common sessions {j.index[0].date()} .. {j.index[-1].date()}")
    say("=" * 160)
    say(f"   days with a hit: research {int((j.research != 0).sum())}, nightly {int((j.nightly != 0).sum())}, both {len(both)}; on those the difference in points: correlation {both.research.corr(both.nightly):.3f}, "
        f"equal within 5 points on {np.mean(np.abs(both.research - both.nightly) <= 5) * 100:.0f}%")
    say(f"   total difference to the leg as traded: research {j.research.sum():+,.0f} points, nightly {j.nightly.sum():+,.0f} points ({j.nightly.sum() / j.research.sum() * 100:.0f}% of it)")
    ok = len(both) >= 0.85 * max((j.research != 0).sum(), 1) and both.research.corr(both.nightly) >= 0.9 and 0.75 <= j.nightly.sum() / j.research.sum() <= 1.25
    say(f"   {'PARITY PASS' if ok else 'PARITY FAIL'} (needs: 85% of the research hit days found, correlation >= 0.9, total within 25%)")
    fr = sh[sh.date == pd.Timestamp('2026-11-13').date()]
    if len(fr):
        r = fr.iloc[0]; b = bd.get(r.date, dict(pts=0.0))
        say(f"   Friday 2026-11-13: band leg as traded {b['pts']:+.2f} points (${b['pts'] * 2:+,.0f} a micro before costs); with the target {b['pts'] + r.diff_pts:+.2f} points "
            f"(${(b['pts'] + r.diff_pts) * 2:+,.0f}); hits {int(r.hits)}, difference ${r.diff_pts * 2:+,.0f}")
    L.flush(); return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
