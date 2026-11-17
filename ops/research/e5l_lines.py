r"""E5L (E5L_PLAN.txt): NinjaTrader's Trend lines(5) and MAMA(0.5, 0.05) as filters and exits of E5n's own legs, inside the book.
    python e5l_lines.py build     indicators on the back-adjusted 1-minute series -> data\research\e5l_<nq|mnqx>.npz
    python e5l_lines.py run       the 12 trials -> E5B_RESULTS.txt
"""
import os, sys, time, shutil
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import G, say
X = L.X; U = L.U; P = X.PATHS
OLD = os.path.join(L.TR.RESEARCH, 'regime_work')


def ind_file(name):
    return os.path.join(P.CACHE, f'e5l_{name}.npz')


def build_one(name):
    out = ind_file(name)
    if os.path.exists(out): return
    src = os.path.join(OLD, f'ntline_{name}.npz')
    if os.path.exists(src):
        shutil.copy2(src, out); print(f"{name}: copied the indicator values computed for the stand-alone test", flush=True); return
    sys.path.insert(0, L.TR.RESEARCH)
    import trendline_port as TP, mama_port as MP
    z = np.load(P.bars_1m(name)); sh = z['shift']; ao, ah, al, ac = z['o'] + sh, z['h'] + sh, z['l'] + sh, z['c'] + sh
    t0 = time.time(); bars = list(zip(ao.tolist(), ah.tolist(), al.tolist(), ac.tolist()))
    val, kinds, _ = TP.trend_lines(bars, 5); kind = np.array([1 if k == 'high' else -1 if k == 'low' else 0 for k in kinds], np.int8)
    Mm, Fm = MP.mama([(None, a, b, c, d) for a, b, c, d in bars], 0.5, 0.05)
    np.savez_compressed(out, val=np.array(val, float), kind=kind, mama=np.array(Mm), fama=np.array(Fm))
    print(f"{name}: indicators on {len(bars):,} bars in {time.time() - t0:.0f} s", flush=True)


def states():
    M = C.M; n, nd = M.px.shape
    S = {k: np.zeros((n, nd), np.int8) for k in ('mup', 'xdn', 'xup', 'agl', 'ags', 'bdn', 'bup')}; cover = np.zeros(n, bool)
    for name, sel in (('nq', np.asarray(G.isA)), ('mnqx', ~np.asarray(G.isA))):
        z = np.load(P.bars_1m(name)); y = np.load(ind_file(name)); t = z['t'].astype('datetime64[m]'); ac = z['c'] + z['shift']
        val, kind, ma, fa = y['val'], y['kind'], y['mama'], y['fama']
        mup = ma > fa; xdn = np.r_[False, mup[:-1] & ~mup[1:]]; xup = np.r_[False, ~mup[:-1] & mup[1:]]
        ok = np.isfinite(val); agl = ok & (kind == 1) & (ac < val); ags = ok & (kind == -1) & (ac > val)
        v0, v1, c0, c1 = val[1:], val[:-1], ac[1:], ac[:-1]; k1 = kind[1:]; o2 = np.isfinite(v0) & np.isfinite(v1)
        bup = np.r_[False, o2 & (c1 <= v1) & (c0 > v0) & (k1 == 1)]; bdn = np.r_[False, o2 & (c1 >= v1) & (c0 < v0) & (k1 == -1)]
        rows = np.flatnonzero(sel)
        for r in rows:
            want = np.datetime64(M.dates[r].date()) + np.timedelta64(570, 'm') + (M.dec - 1).astype('timedelta64[m]')
            j = np.searchsorted(t, want, side='right') - 1; good = (j >= 0) & ((want - t[np.clip(j, 0, None)]) <= np.timedelta64(5, 'm'))
            jj = np.clip(j, 0, None)
            if good.mean() < 0.9: continue
            cover[r] = True
            for k, a in (('mup', mup), ('xdn', xdn), ('xup', xup), ('agl', agl), ('ags', ags), ('bdn', bdn), ('bup', bup)): S[k][r] = np.where(good, a[jj], 0)
    return S, cover


def leg_overlay(p, rule):
    M = C.M; px = M.px; n, nd = p.shape; out = np.zeros_like(p); prev = np.zeros(n); d = np.zeros(n); ent = np.ones(n); best = np.zeros(n); off = np.zeros(n, bool)
    for j in range(nd):
        pj = p[:, j]; new = pj != prev; start = new & (pj != 0)
        d = np.where(new, np.sign(pj), d); ent = np.where(start, px[:, j], ent); best = np.where(start, px[:, j], best); off = np.where(new, False, off)
        out[:, j] = np.where(off, 0.0, pj); prev = pj
        live = (d != 0) & ~off; q = px[:, j]
        best = np.where(live & (d > 0), np.maximum(best, q), np.where(live & (d < 0), np.minimum(best, q), best))
        if rule is not None: off = off | (live & rule(j, d, q, ent, best))
    return out


def main():
    t0 = time.time(); C.load(); M = C.M; S, cover = states()
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); band = M.band
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); atr = np.where((M.dec >= 15)[None, :], G.diag['atr'][:, gi], np.nan)
    from e5q_capture import days
    keep = np.asarray(M.keep); A_ = keep & np.asarray(G.isA); B_ = keep & ~np.asarray(G.isA)
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    say(); say("=" * 160)
    say(f"E5L  TREND LINES + MAMA INSIDE E5n  e5l_lines.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5L_PLAN.txt) 12 trials; fair model leg 1 + band leg 1, one net position, live cost; "
        f"select 2013-19, confirm 2019-26")
    say("=" * 160)
    say(f"indicator coverage: {cover[keep].mean():.1%} of sessions 2013-26; MAMA above FAMA on {S['mup'][keep].mean():.0%} of minutes; price against an unbroken line: "
        f"below a descending one {S['agl'][keep].mean():.0%}, above an ascending one {S['ags'][keep].mean():.0%}; breaks a session: {(S['bup'][keep].sum(1) + S['bdn'][keep].sum(1)).mean():.1f}")
    assert np.array_equal(leg_overlay(band, None), band) and np.array_equal(leg_overlay(fair, None), fair)
    mup = S['mup'].astype(bool); agl = S['agl'].astype(bool); ags = S['ags'].astype(bool)
    sig = np.sign(np.where(np.abs(M.a5) > L.T0, M.a5, 0.0))
    def mfilter(block):
        f5 = np.where(block, 0.0, M.a5); return C.path(f5, M.a15, 'set', clk0)
    b_mama = ((sig > 0) & ~mup) | ((sig < 0) & mup); b_line = ((sig > 0) & agl) | ((sig < 0) & ags)
    def bfilter(skip_long, skip_short):
        p = band.copy(); prev = np.zeros(M.n); skip = np.zeros(M.n, bool)
        for j in range(band.shape[1]):
            bj = band[:, j]; new = bj != prev; start = new & (bj != 0)
            skip = np.where(new, start & (((bj > 0) & skip_long[:, j]) | ((bj < 0) & skip_short[:, j])), skip)
            p[:, j] = np.where(skip, 0.0, bj); prev = bj
        return p
    R = 15.0 / 27875.0
    trials = {
        'L1': ('model leg: signal only with MAMA on its side', lambda: (mfilter(b_mama), band)),
        'L2': ('model leg: no signal against the trend line', lambda: (mfilter(b_line), band)),
        'L3': ('model leg: both L1 and L2', lambda: (mfilter(b_mama | b_line), band)),
        'L4': ('band leg: skip entries with MAMA against', lambda: (fair, bfilter(~mup, mup))),
        'L5': ('band leg: skip entries against the trend line', lambda: (fair, bfilter(agl, ags))),
        'L6': ('band leg: skip if L4 or L5', lambda: (fair, bfilter(~mup | agl, mup | ags))),
        'L7': ('band leg: exit at a MAMA cross against', lambda: (fair, leg_overlay(band, lambda j, d, q, e, b: ((d > 0) & (S['xdn'][:, j] > 0)) | ((d < 0) & (S['xup'][:, j] > 0))))),
        'L8': ('band leg: exit at a trend-line break against', lambda: (fair, leg_overlay(band, lambda j, d, q, e, b: ((d > 0) & (S['bdn'][:, j] > 0)) | ((d < 0) & (S['bup'][:, j] > 0))))),
        'L9': ('band leg: L7 or L8 after 1 ATR of open profit', lambda: (fair, leg_overlay(band, lambda j, d, q, e, b: (d * (b - e) >= np.nan_to_num(atr[:, j], nan=1e9)) & (
            ((d > 0) & ((S['xdn'][:, j] > 0) | (S['bdn'][:, j] > 0))) | ((d < 0) & ((S['xup'][:, j] > 0) | (S['bup'][:, j] > 0))))))),
        'L10': ('model leg: exit at a trend-line break against', lambda: (leg_overlay(fair, lambda j, d, q, e, b: ((d > 0) & (S['bdn'][:, j] > 0)) | ((d < 0) & (S['bup'][:, j] > 0))), band)),
        'L11': ('band leg: 15-point target + 15-point stop', lambda: (fair, leg_overlay(band, lambda j, d, q, e, b: np.abs(d * (q - e) / e) >= R))),
        'L12': ('model leg: 15-point target + 15-point stop', lambda: (leg_overlay(fair, lambda j, d, q, e, b: np.abs(d * (q - e) / e) >= R), band)),
    }
    ub, _ = days(fair + band); vb, _ = days(band); yrs = keep.sum() / 252
    def row(lab, u, m_, b_):
        tr = np.abs(np.diff(np.c_[np.zeros(M.n), m_ + b_, np.zeros(M.n)], axis=1)).sum(1)[keep].sum() / 2 / yrs
        return (f"   {lab:50s} | book 2013-19 Sharpe {sh(u[A_]):.2f} ${u[A_].mean() * 252:+7,.0f}/yr DD ${dd(u[A_]):6,.0f} | 2019-26 Sharpe {sh(u[B_]):.2f} ${u[B_].mean() * 252:+7,.0f}/yr "
                f"DD ${dd(u[B_]):6,.0f} | {tr:5.0f} round trips/yr")
    say(); say(row('BASELINE (E5n as built)', ub, fair, band)); res = {}
    for code, (desc, mk) in trials.items():
        m_, b_ = mk(); u, _ = days(m_ + b_); res[code] = u
        say(row(f"{code} {desc}", u, m_, b_) + ("  SELECTED" if sh(u[A_]) >= sh(ub[A_]) + 0.05 else ""))
    sel = [c for c in trials if sh(res[c][A_]) >= sh(ub[A_]) + 0.05]; bar = 2.0 if len(sel) == 1 else 2.4 if len(sel) <= 5 else 2.6
    say(); say(f"CONFIRMATION 2019-26 for {len(sel)} selected (bar: paired t >= {bar}):" if sel else "CONFIRMATION: nothing selected on 2013-19.")
    for c in sel:
        dl = res[c][B_] - ub[B_]; t = dl.mean() / dl.std() * np.sqrt(len(dl)) if dl.std() > 0 else 0.0
        say(f"   {c} {trials[c][0]}: Sharpe {sh(res[c][B_]):.2f} vs {sh(ub[B_]):.2f}, ${dl.mean() * 252:+,.0f}/yr, max DD ${dd(res[c][B_]):,.0f} vs ${dd(ub[B_]):,.0f}, paired t {t:+.2f} -> "
            f"{'CONFIRMED (gauntlet next)' if sh(res[c][B_]) > sh(ub[B_]) and t >= bar else 'not confirmed'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build':
        for nm in ('nq', 'mnqx'): build_one(nm)
    elif a == 'run': main()
    else: print(__doc__)
