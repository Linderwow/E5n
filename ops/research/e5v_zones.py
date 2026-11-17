r"""E5V (E5V_PLAN.txt): stop loss and profit take at fair value gaps and zones of interest, as overlays on E5n's two legs in the book.
    python e5v_zones.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5v_struct as ST
from e5b_lab import G, say
X = L.X; U = L.U

TR = {'F1': ('band', [('fvg', 1)], [], None, 'FVG profit take, 1-minute'), 'F2': ('band', [('fvg', 5)], [], None, 'FVG profit take, 5-minute'),
      'F3': ('band', [('fvg', 15)], [], None, 'FVG profit take, 15-minute'), 'F4': ('band', [('fvg', 60)], [], None, 'FVG profit take, 60-minute'),
      'F5': ('band', [], [('fvg', 1)], None, 'FVG stop, 1-minute'), 'F6': ('band', [], [('fvg', 5)], None, 'FVG stop, 5-minute'),
      'F7': ('band', [], [('fvg', 15)], None, 'FVG stop, 15-minute'), 'F8': ('band', [], [('fvg', 60)], None, 'FVG stop, 60-minute'),
      'F9': ('band', [('fvg', 5)], [('fvg', 5)], None, 'FVG stop + profit take, 5-minute'), 'F10': ('band', [('fvg', 15)], [('fvg', 15)], None, 'FVG stop + profit take, 15-minute'),
      'F11': ('band', [], [], 5, 'FVG trailing stop, 5-minute'), 'F12': ('band', [], [], 15, 'FVG trailing stop, 15-minute'),
      'Z1': ('band', [('kl',)], [], None, 'key-level profit take'), 'Z2': ('band', [], [('kl',)], None, 'key-level stop'), 'Z3': ('band', [('kl',)], [('kl',)], None, 'key-level stop + profit take'),
      'Z4': ('band', [('sw',)], [], None, 'swing profit take'), 'Z5': ('band', [], [('sw',)], None, 'swing stop'), 'Z6': ('band', [], [], 'sw', 'swing trailing stop'),
      'Z7': ('band', [('fvg', 15), ('kl',), ('sw',)], [('fvg', 15), ('kl',), ('sw',)], None, 'everything: nearest gap / level / swing'),
      'Y1': ('model', [('fvg', 1)], [], None, 'model leg: FVG profit take, 1-minute'), 'Y2': ('model', [('fvg', 5)], [], None, 'model leg: FVG profit take, 5-minute'),
      'Y3': ('model', [], [('fvg', 1)], None, 'model leg: FVG stop, 1-minute'), 'Y4': ('model', [], [('fvg', 5)], None, 'model leg: FVG stop, 5-minute'),
      'Y5': ('model', [('kl',), ('sw',)], [], None, 'model leg: level / swing profit take'), 'Y6': ('model', [], [('kl',), ('sw',)], None, 'model leg: level / swing stop')}


def days(net, adj=None):
    M = C.M; c1 = X.SIDE(2.35) * U; tr = np.abs(np.diff(np.c_[np.zeros(M.n), net], axis=1))
    cum = np.cumsum(net * M.dpx, axis=1) / M.o930[:, None] * U - c1 * np.cumsum(tr, axis=1)
    close = cum[:, -1] - c1 * np.abs(net[:, -1])
    if adj is not None: close = close + adj / M.o930 * U
    return close, np.minimum(np.minimum(cum.min(axis=1), close), 0.0)


def trades_of(path, ERA, day0, isA, keep, tm=1.0, sm=0.5):
    M = C.M; nd = path.shape[1]; out = []; miss = 0
    for r in np.flatnonzero(keep):
        p = path[r]; ch = np.flatnonzero(np.diff(np.r_[0.0, p]) != 0); E = ERA['A' if isA[r] else 'B']
        for a_, j0 in enumerate(ch):
            if p[j0] == 0: continue
            j1 = int(ch[a_ + 1]) if a_ + 1 < len(ch) else nd; d = float(np.sign(p[j0]))
            tau = int(day0[r]) + 575 + int(j0); tau1 = int(day0[r]) + 575 + j1 if j1 < nd else int(day0[r]) + 960
            i = np.searchsorted(E.t, tau)
            if i >= len(E.t) or E.t[i] != tau: miss += 1; continue
            price = E.o[i]; A = ST.atr5(E, tau); lv = ST.levels(E, tau, d, price, A, tm, sm)
            if lv is None: miss += 1; continue
            out.append(dict(r=r, j0=int(j0), j1=j1, d=d, tau=tau, tau1=tau1, off=M.px[r, j0] - price, lv=lv, E=E))
    return out, miss


def apply(path, trades, tsrc, ssrc, trail, day0, market=False):
    M = C.M; nd = path.shape[1]; p2 = path.copy(); adj = np.zeros(M.n); nT = nS = 0
    for q in trades:
        d = q['d']; lv = q['lv']
        T = ST.nearest(np.array([lv[('T',) + s] for s in tsrc], float), d, True) if tsrc else np.nan
        S = ST.nearest(np.array([lv[('S',) + s] for s in ssrc], float), d, False) if ssrc else np.nan
        ev = ST.trail_events(q['E'], q['tau'], q['tau1'], d, trail) if trail is not None else None
        if not (T == T or S == S or (ev is not None and len(ev[0]))): continue
        x = ST.sim_exit(q['E'], q['tau'], q['tau1'], d, T, S, ev, market)
        if x is None: continue
        te, F, kind = x; r = q['r']; m = min(te - int(day0[r]) - 575, nd - 1)
        if m < q['j0'] - 1: continue
        p2[r, m + 1:q['j1']] = 0.0
        adj[r] += d * (F + q['off'] - (M.px[r, m + 1] if m + 1 < nd else M.exit[r]))
        if kind == 'target': nT += 1
        else: nS += 1
    return p2, adj, nT, nS


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import src2_lib as L2
    ERA = {'A': ST.load(L2, 'nq'), 'B': ST.load(L2, 'mnqx')}
    keep = np.asarray(M.keep); isA = np.asarray(G.isA); A_ = keep & isA; B_ = keep & ~isA; yrs = keep.sum() / 252
    day0 = M.dates.values.astype('datetime64[m]').astype(np.int64)
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0)
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    say(); say("=" * 160)
    say(f"STOPS AND PROFIT TAKES AT GAPS AND ZONES  e5v_zones.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5V_PLAN.txt) overlays on E5n's legs, resting orders at the level; fair model leg 1 + band leg 1, "
        f"live cost, $ at today's price; select 2013-19, confirm 2019-26")
    say("=" * 160)
    n5 = {k: len(ERA[k].fvg[5].t) / (len(np.unique(ERA[k].dd)) / 365.25 * 252 / 252) for k in ERA}
    say("   gaps found per trading day (1 / 5 / 15 / 60 / 240-minute): " + " | ".join(f"{'NQ 2010-19' if k == 'A' else 'MNQ 2019-26'} " + " / ".join(f"{len(ERA[k].fvg[tf].t) / len(ERA[k].kl):.1f}" for tf in (1, 5, 15, 60, 240)) for k in ERA))
    legs = {'band': M.band, 'model': fair}; TRD = {}
    for nm, path in legs.items():
        TRD[nm], miss = trades_of(path, ERA, day0, isA, keep)
        chk = [abs((M.px[q['r'], q['j0'] + 1] - q['off']) - q['E'].o[np.searchsorted(q['E'].t, q['tau'] + 1)]) for q in TRD[nm][:3000] if q['j0'] + 1 < path.shape[1] and q['E'].t[np.searchsorted(q['E'].t, q['tau'] + 1)] == q['tau'] + 1]
        have = lambda key: np.mean([q['lv'][key] == q['lv'][key] for q in TRD[nm]]) * 100
        say(f"   {nm} leg: {len(TRD[nm]):,} trades with data ({miss} without); price grid vs 1-minute data one minute later: median difference {np.median(chk):.2f} points; a zone ahead / behind exists for "
            f"1-min gaps {have(('T', 'fvg', 1)):.0f}% / {have(('S', 'fvg', 1)):.0f}%, 5-min {have(('T', 'fvg', 5)):.0f}% / {have(('S', 'fvg', 5)):.0f}%, 15-min {have(('T', 'fvg', 15)):.0f}% / {have(('S', 'fvg', 15)):.0f}%, "
            f"60-min {have(('T', 'fvg', 60)):.0f}% / {have(('S', 'fvg', 60)):.0f}%, key levels {have(('T', 'kl')):.0f}% / {have(('S', 'kl')):.0f}%, swings {have(('T', 'sw')):.0f}% / {have(('S', 'sw')):.0f}%")
    b0, a0, _, _ = apply(M.band, TRD['band'], [], [], None, day0); assert np.array_equal(b0, M.band) and not a0.any()
    ub, _ = days(fair + M.band); vb = {'band': days(M.band)[0], 'model': days(fair)[0]}

    def run(code, trd=None, market=False, mdl=None):
        leg, ts, ss, trail, desc = TR[code]; trd = trd or TRD; mdl = fair if mdl is None else mdl
        p2, adj, nT, nS = apply(legs[leg], trd[leg], ts, ss, trail, day0, market)
        net = (mdl + p2) if leg == 'band' else (p2 + M.band)
        if leg == 'model' and mdl is not fair: net = p2 + M.band
        u, w = days(net, adj); v, _ = days(p2, adj)
        return u, v, nT, nS, p2, adj

    say(); say(f"   {'BASELINE (legs as built)':47s} | book 2013-19 Sharpe {sh(ub[A_]):.2f} ${ub[A_].mean() * 252:+7,.0f}/yr DD ${dd(ub[A_]):6,.0f} | 2019-26 Sharpe {sh(ub[B_]):.2f} ${ub[B_].mean() * 252:+7,.0f}/yr DD ${dd(ub[B_]):6,.0f}"
               f" | band alone ${vb['band'][A_].mean() * 252:+6,.0f} / ${vb['band'][B_].mean() * 252:+6,.0f}, model alone ${vb['model'][A_].mean() * 252:+6,.0f} / ${vb['model'][B_].mean() * 252:+6,.0f} a year")
    res = {}
    for code in TR:
        u, v, nT, nS, p2, adj = run(code); res[code] = (u, v); leg = TR[code][0]
        sel = sh(u[A_]) >= sh(ub[A_]) + 0.05
        say(f"   {code:3s} {TR[code][4]:43s} | book 2013-19 Sharpe {sh(u[A_]):.2f} ${u[A_].mean() * 252:+7,.0f}/yr DD ${dd(u[A_]):6,.0f} | 2019-26 Sharpe {sh(u[B_]):.2f} ${u[B_].mean() * 252:+7,.0f}/yr DD ${dd(u[B_]):6,.0f}"
            f" | the leg alone ${v[A_].mean() * 252:+6,.0f} / ${v[B_].mean() * 252:+6,.0f} a year | profit takes {nT / yrs:4.0f}/yr, stops {nS / yrs:4.0f}/yr" + ("  SELECTED" if sel else ""))
    selected = [c for c in TR if sh(res[c][0][A_]) >= sh(ub[A_]) + 0.05]
    bar = 2.0 if len(selected) == 1 else 2.4 if len(selected) <= 5 else 2.6
    say(); say(f"CONFIRMATION 2019-26 for {len(selected)} selected rule(s) (bar: paired t >= {bar}):" if selected else "CONFIRMATION: nothing selected on 2013-19.")
    conf = []
    for c in selected:
        dlt = res[c][0][B_] - ub[B_]; t = tt(dlt); ok = sh(res[c][0][B_]) > sh(ub[B_]) and t >= bar
        say(f"   {c} {TR[c][4]}: Sharpe {sh(res[c][0][B_]):.2f} vs {sh(ub[B_]):.2f}, ${dlt.mean() * 252:+,.0f}/yr vs baseline, max DD ${dd(res[c][0][B_]):,.0f} vs ${dd(ub[B_]):,.0f}, paired t {t:+.2f} -> {'CONFIRMED' if ok else 'not confirmed'}")
        if ok: conf.append(c)
    adopted = []
    for c in conf:
        leg = TR[c][0]; u, v = res[c]; say(f"GAUNTLET {c} {TR[c][4]}:"); oks = []
        k1 = True
        for tm, sm in ((0.5, 0.25), (2.0, 1.0)):
            trd = {leg: trades_of(legs[leg], ERA, day0, isA, keep, tm, sm)[0]}; u1 = run(c, trd)[0]
            good = sh(u1[A_]) >= sh(ub[A_]) and sh(u1[B_]) >= sh(ub[B_]); k1 &= good
            say(f"   K1 distances {tm} A / {sm} A: book Sharpe {sh(u1[A_]):.2f} / {sh(u1[B_]):.2f} vs {sh(ub[A_]):.2f} / {sh(ub[B_]):.2f} -> {'ok' if good else 'below'}")
        oks.append(('K1', k1))
        d = pd.Series((u - ub)[keep], index=M.dates[keep]); yr = d.groupby(d.index.year).sum(); dB = d[d.index > L.JOIN]; yB = dB.groupby(dB.index.year).sum()
        k2 = (yr > 0).sum() >= 8 and dB[dB.index.year != yB.idxmax()].sum() > 0; oks.append(('K2', k2))
        say(f"   K2 years: positive in {(yr > 0).sum()} of {len(yr)}; 2019-26 ${dB.sum():+,.0f}, without {yB.idxmax()} ${dB[dB.index.year != yB.idxmax()].sum():+,.0f} -> {'PASS' if k2 else 'FAIL'}")
        k3 = dB.drop(dB.nlargest(5).index).sum() >= 0; oks.append(('K3', k3)); say(f"   K3 without the 5 best days: ${dB.drop(dB.nlargest(5).index).sum():+,.0f} -> {'PASS' if k3 else 'FAIL'}")
        dv = v - vb[leg]; k4 = tt(dv[B_]) >= 1.5 and dv[A_].mean() * 252 >= -300; oks.append(('K4', k4))
        say(f"   K4 the leg alone: ${dv[A_].mean() * 252:+,.0f} / ${dv[B_].mean() * 252:+,.0f} a year, paired t {tt(dv[A_]):+.2f} / {tt(dv[B_]):+.2f} -> {'PASS' if k4 else 'FAIL'}")
        if leg == 'band':
            db, _ = days(dep + M.band); u5 = run(c, mdl=dep)[0]; k5 = sh(u5[A_]) >= sh(db[A_]) and sh(u5[B_]) >= sh(db[B_]); oks.append(('K5', k5))
            say(f"   K5 model leg as backtested: book Sharpe {sh(u5[A_]):.2f} vs {sh(db[A_]):.2f} / {sh(u5[B_]):.2f} vs {sh(db[B_]):.2f} -> {'PASS' if k5 else 'FAIL'}")
        u6 = run(c, market=True)[0]; k6 = sh(u6[A_]) >= sh(ub[A_]) and sh(u6[B_]) >= sh(ub[B_]); oks.append(('K6', k6))
        say(f"   K6 market orders instead of resting ones: book Sharpe {sh(u6[A_]):.2f} / {sh(u6[B_]):.2f} vs {sh(ub[A_]):.2f} / {sh(ub[B_]):.2f} -> {'PASS' if k6 else 'FAIL'}")
        bad = [n for n, o in oks if not o]; say(f"   -> {'ALL PASS: candidate' if not bad else 'FAILED ' + ', '.join(bad) + ': near miss, not adopted'}")
        if not bad: adopted.append(c)
    say(); say(f"VERDICT (exits at gaps and zones): {'CANDIDATE ' + ', '.join(adopted) if adopted else 'none adopted' + (' (near miss: ' + ', '.join(conf) + ')' if conf else '')}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
