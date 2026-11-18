r"""LT (LT_PLAN.txt): resting limit profit targets on the model and band legs, 1+1 and 2+2, 4.5 ticks a side.
    python lt_limit.py build
    python lt_limit.py run
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from types import SimpleNamespace as NS
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
from e5b_lab import G, say

TK = 4.5
TICK = 0.25
HL = C.C + 'hl.npz'


def build():
    t0 = time.time(); C.load(); M = C.M; L2, _, _, _ = C.TR.research()
    hi = np.full((M.n, C.NM), np.nan); lo = np.full((M.n, C.NM), np.nan); op = np.full((M.n, C.NM), np.nan)
    for nm, sel in (('nq', lambda d: d <= L.X.JOIN), ('mnqx', lambda d: d > L.X.JOIN)):
        R = C.TR.Raw(L2, nm); idx = {pd.Timestamp(d): i for i, d in enumerate(R.D.date)}
        for r, d in enumerate(M.dates):
            if sel(d) and d in idx:
                k = idx[d]; hi[r] = R.H[k, M.dec]; lo[r] = R.L[k, M.dec]; op[r] = R.O[k, M.dec]
        print(f"{nm}: {time.time() - t0:.0f} s", flush=True)
    ok = ~np.isnan(op); mism = np.abs(op[ok] - M.px[ok]).max(); miss = (~ok.any(1)).sum()
    print(f"rows without bars {miss}; max |open - cached entry| {mism:.4f}")
    assert miss == 0 and mism < 1e-6, 'minute grid does not match the lab cache'
    np.savez_compressed(HL, hi=hi, lo=lo); print(f"saved {HL}")


def limit_leg(leg, A, W, k, unit):
    M = C.M; px = M.px; n = px.shape[0]; z = np.load(HL); hi, lo = z['hi'], z['lo']
    pos = np.zeros_like(px); prev = np.zeros(n); d = np.zeros(n); out = np.zeros(n, bool); tg = np.full(n, np.nan)
    add = np.zeros(n); fills = np.zeros((n, C.NM), bool)
    for j in range(C.NM):
        bj = np.sign(leg[:, j]); new = bj != prev; start = new & (bj != 0)
        d = np.where(new, bj, d); out = np.where(new, False, out)
        lvl = (A[:, j] if unit == 'atr' else W[:, j]) * k
        tg = np.where(start, px[:, j] + bj * lvl, tg)
        live = (d != 0) & ~out & np.isfinite(tg)
        hit = live & (((d > 0) & (hi[:, j] >= tg + TICK)) | ((d < 0) & (lo[:, j] <= tg - TICK)))
        fill = np.where(d > 0, np.maximum(tg, px[:, j]), np.minimum(tg, px[:, j]))
        add = add + np.where(hit, d * (fill - px[:, j]), 0.0)
        out = out | hit; fills[:, j] = hit
        pos[:, j] = np.where(out, 0.0, bj); prev = bj
    return pos, add, fills


def sim(net, add, fills, ticks):
    M = C.M; X = L.X
    pnl = (net * M.dpx).sum(1) + add
    dn = np.abs(np.diff(np.c_[np.zeros(M.n), net, np.zeros(M.n)], axis=1)); sides = dn.sum(1)
    cred = np.minimum(fills.astype(float), dn[:, :C.NM]).sum(1)
    cost = sides * X.SIDE(ticks) - cred * (X.SIDE(ticks) - X.SIDE(0)) if ticks is not None else 0.0
    r = (pnl / M.o930 - cost) * X.U
    return pd.Series(r[M.keep], index=M.dates[M.keep]), float(sides[M.keep].sum() / 2), int(fills[M.keep].sum())


def run():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'LT_RESULTS.txt')
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    A = slot('atr'); W = (slot('ub') - slot('lb')) / 2
    model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band.copy()
    ctx = NS(atr=A, vw=slot('vw'), half=W, mdl=model)
    def capped(leg, rule):
        keepb = M.band; M.band = leg
        try: return Q.overlay(rule, ctx)
        finally: M.band = keepb
    gb = lambda S: ((S.mfe >= 1 * S.A) & (S.op <= 0.5 * S.mfe), S.op * 0 > 1)
    Z = np.zeros(M.n); NF = np.zeros((M.n, C.NM), bool)
    a26 = np.nanmedian(A[M.dates.year == 2026][:, M.grid]); w26 = np.nanmedian(W[M.dates.year == 2026][:, M.grid])
    mL = {k: limit_leg(model, A, W, k, 'atr') for k in (0.5, 1.0, 1.5)}; mG = capped(model, gb)
    bQ = limit_leg(band, A, W, 2.0, 'half'); b3 = limit_leg(band, A, W, 3.0, 'atr')
    F1 = {'L1 model limit target 0.5 ATR': (mL[0.5][0] + band, mL[0.5][1], mL[0.5][2]),
          'L2 model limit target 1 ATR': (mL[1.0][0] + band, mL[1.0][1], mL[1.0][2]),
          'L3 model limit target 1.5 ATR': (mL[1.5][0] + band, mL[1.5][1], mL[1.5][2]),
          'G1 model give back 50% after 1 ATR': (mG + band, Z, NF)}
    F2 = {'L4 model 2nd ct limit 0.5 ATR': (model + mL[0.5][0] + 2 * band, mL[0.5][1], mL[0.5][2]),
          'L5 model 2nd ct limit 1 ATR': (model + mL[1.0][0] + 2 * band, mL[1.0][1], mL[1.0][2]),
          'L6 model 2nd ct limit 1.5 ATR': (model + mL[1.5][0] + 2 * band, mL[1.5][1], mL[1.5][2]),
          'G2 model 2nd ct give back 50% / 1 ATR': (model + mG + 2 * band, Z, NF),
          'B1 band 2nd ct limit 2 half-widths': (2 * model + band + bQ[0], bQ[1], bQ[2]),
          'B2 band 2nd ct limit 3 ATR': (2 * model + band + b3[0], b3[1], b3[2])}
    rat = lambda k: k['yrB'] / k['ddB'] if k['ddB'] > 0 else np.nan
    say(); say("=" * 156); say(f"LT  RESTING LIMIT PROFIT TARGETS  lt_limit.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {TK} ticks a side (limit fills: commission only); ratio = $/yr per $ max DD 2019-26"); say("=" * 156)
    say(f"   2026 medians: ATR {a26:.1f} points, band half-width {w26:.1f} points  ->  0.5 / 1 / 1.5 ATR = {0.5 * a26:.0f} / {a26:.0f} / {1.5 * a26:.0f} points")
    for title, base_net, V in (('FRAME 1+1 (today: Contracts 1)', model + band, F1), ('FRAME 2+2 (Contracts 2)', 2 * model + 2 * band, F2)):
        base, tb, _ = sim(base_net, Z, NF, TK); kb = L.stats(base); b0, _, _ = sim(base_net, Z, NF, None); k0 = L.stats(b0)
        say(); say(f"   {title}: baseline 2013-19 Sharpe {kb['shA']:.2f} ${kb['yrA']:,.0f}/yr DD ${kb['ddA']:,.0f} | 2019-26 Sharpe {kb['shB']:.2f} ${kb['yrB']:,.0f}/yr DD ${kb['ddB']:,.0f} ratio {rat(kb):.2f} | zero-cost 2019-26 ${k0['yrB']:,.0f}/yr")
        for nm, (net, add, fl) in V.items():
            s, tr, nf = sim(net, add, fl, TK); k = L.stats(s); s0, _, _ = sim(net, add, fl, None); kz = L.stats(s0)
            sel = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']; _, sb = L.halves(s); _, bb = L.halves(base); dd = (sb - bb).values
            t = dd.mean() / dd.std() * np.sqrt(len(dd)) if dd.std() > 0 else 0.0
            ok = sel and k['shB'] > kb['shB'] and k['ddB'] <= kb['ddB'] and t >= 2.0
            yrs = len(s) / 252
            say(f"   {nm:38s} 2013-19 Sh {k['shA']:.2f} ({k['shA'] - kb['shA']:+.2f}) DD ${k['ddA']:,.0f} {'SEL' if sel else '   '}| 2019-26 Sh {k['shB']:.2f} ({k['shB'] - kb['shB']:+.2f}) "
                f"${k['yrB']:,.0f}/yr ({k['yrB'] - kb['yrB']:+,.0f}) DD ${k['ddB']:,.0f} ratio {rat(k):.2f} t {t:+.2f} | 0-cost {kz['yrB'] - k0['yrB']:+,.0f} | fills {nf / yrs:,.0f}/yr" + (' CONFIRMED' if ok else ''))
    say(f"10 trials; run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build': build()
    elif a == 'run': run()
    else: print(__doc__)
