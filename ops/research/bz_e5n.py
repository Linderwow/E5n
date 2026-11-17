r"""BZ F1 / F2 (BZ_PLAN.txt): swing and flip zones on 60 / 240-minute and daily bars as filters on E5n's two legs, inside the book.
    python bz_e5n.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import bz_zones as BZ
from e5b_lab import G, say
from e5q_capture import days
X = L.X


def zone_grid():
    M = C.M; n, nd = M.px.shape; near_s = np.zeros((n, nd), bool); near_r = np.zeros((n, nd), bool); cover = np.zeros(n, bool)
    for name, sel in (('nq', np.asarray(G.isA)), ('mnqx', ~np.asarray(G.isA))):
        d = BZ.load(name); sets, w60, _ = BZ.zone_sets(d, (60, 240, 1440))
        S = np.column_stack([sets[k][0] for k in sets if k.startswith(('Z1', 'Z2'))]); R = np.column_stack([sets[k][1] for k in sets if k.startswith(('Z1', 'Z2'))])
        pc = d.c; S = np.where(S <= pc[:, None], S, np.nan); R = np.where(R >= pc[:, None], R, np.nan)
        with np.errstate(all='ignore'):
            sn = np.nanmax(S, axis=1); rn = np.nanmin(R, axis=1)
        ns = np.isfinite(sn) & (pc - sn <= w60); nr = np.isfinite(rn) & (rn - pc <= w60)
        for r in np.flatnonzero(sel):
            want = np.datetime64(M.dates[r].date()) + np.timedelta64(570, 'm') + (M.dec - 1).astype('timedelta64[m]')
            j = np.searchsorted(d.t, want.astype(np.int64), side='right') - 1; good = (j >= 0) & ((want.astype(np.int64) - d.t[np.clip(j, 0, None)]) <= 5)
            if good.mean() < 0.9: continue
            cover[r] = True; jj = np.clip(j, 0, None); near_s[r] = good & ns[jj]; near_r[r] = good & nr[jj]
    return near_s, near_r, cover


def main():
    t0 = time.time(); C.load(); M = C.M; ns, nr, cover = zone_grid()
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); band = M.band
    keep = np.asarray(M.keep); A_ = keep & np.asarray(G.isA); B_ = keep & ~np.asarray(G.isA)
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    say(); say("=" * 160)
    say(f"BZ F1 / F2  ZONES AS E5n FILTERS  bz_e5n.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (BZ_PLAN.txt) fair model leg 1 + band leg 1, live cost; select 2013-19, confirm 2019-26")
    say("=" * 160)
    say(f"coverage {cover[keep].mean():.1%} of sessions; decision minutes within one 60-minute w above support {ns[keep].mean():.1%}, below resistance {nr[keep].mean():.1%}")
    sig = np.sign(np.where(np.abs(M.a5) > L.T0, M.a5, 0.0)); block = ((sig < 0) & ns) | ((sig > 0) & nr)
    f2 = C.path(np.where(block, 0.0, M.a5), M.a15, 'set', clk0)
    p = band.copy(); prev = np.zeros(M.n); skip = np.zeros(M.n, bool)
    for j in range(band.shape[1]):
        bj = band[:, j]; new = bj != prev; start = new & (bj != 0)
        skip = np.where(new, start & (((bj < 0) & ns[:, j]) | ((bj > 0) & nr[:, j])), skip); p[:, j] = np.where(skip, 0.0, bj); prev = bj
    ub, _ = days(fair + band)
    def row(lab, u): return f"   {lab:62s} | 2013-19 Sharpe {sh(u[A_]):.2f} ${u[A_].mean() * 252:+7,.0f}/yr DD ${dd(u[A_]):6,.0f} | 2019-26 Sharpe {sh(u[B_]):.2f} ${u[B_].mean() * 252:+7,.0f}/yr DD ${dd(u[B_]):6,.0f}"
    say(row('BASELINE (E5n as built)', ub)); res = {}
    for code, desc, m_, b_ in (('F1', 'band leg: skip entries selling into support / buying into resistance', fair, p), ('F2', 'model leg: no signal into a zone', f2, band)):
        u, _ = days(m_ + b_); res[code] = u; say(row(f"{code} {desc}", u) + ("  SELECTED" if sh(u[A_]) >= sh(ub[A_]) + 0.05 else ""))
    sel = [c for c in res if sh(res[c][A_]) >= sh(ub[A_]) + 0.05]; bar = 2.0 if len(sel) <= 1 else 2.4
    for c in sel:
        dl = res[c][B_] - ub[B_]; t = dl.mean() / dl.std() * np.sqrt(len(dl))
        say(f"   CONFIRM {c}: Sharpe {sh(res[c][B_]):.2f} vs {sh(ub[B_]):.2f}, ${dl.mean() * 252:+,.0f}/yr, paired t {t:+.2f} (bar {bar}) -> {'CONFIRMED' if sh(res[c][B_]) > sh(ub[B_]) and t >= bar else 'not confirmed'}")
    if not sel: say("   nothing selected on 2013-19")
    say(f"run {time.time() - t0:.0f} s"); L.OUT = os.path.join(HERE, 'BZ_RESULTS.txt'); L.flush()


if __name__ == '__main__':
    main()
