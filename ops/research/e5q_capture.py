r"""E5Q (E5Q_PLAN.txt + amendment 1): profit-capture mechanisms on E5n's band leg, inside the two-leg book, selected on 2013-19 and
confirmed on 2019-26.
    python e5q_capture.py
"""
import os, sys, time
from types import SimpleNamespace as NS
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import G, say
X = L.X; U = L.U
ROUND1 = ('G1', 'G2', 'G3', 'G4', 'G5', 'E1', 'E2', 'T1', 'T2', 'T3', 'C1', 'C2', 'C3', 'S1', 'S2', 'D1', 'D2')


def days(net):
    M = C.M; c1 = X.SIDE(2.35) * U; tr = np.abs(np.diff(np.c_[np.zeros(M.n), net], axis=1))
    cum = np.cumsum(net * M.dpx, axis=1) / M.o930[:, None] * U - c1 * np.cumsum(tr, axis=1)
    close = cum[:, -1] - c1 * np.abs(net[:, -1]); worst = np.minimum(np.minimum(cum.min(axis=1), close), 0.0)
    return close, worst


def overlay(rule, ctx, anatomy=False, reenter=False):
    M = C.M; b = M.band; px = M.px; n, nd = b.shape
    pos = np.zeros_like(b); prevb = np.zeros(n); d = np.zeros(n); ent = np.ones(n); best = np.zeros(n); out = np.zeros(n, bool); frac = np.ones(n)
    mfe = np.zeros(n); A0 = np.full(n, np.nan); W0 = np.full(n, np.nan); rec = []; back_n = 0
    for j in range(nd):
        bj = b[:, j]; new = bj != prevb; start = new & (bj != 0)
        if anatomy:
            for i in np.flatnonzero(new & (d != 0)): rec.append((i, d[i], ent[i], mfe[i], d[i] * (px[i, j] - ent[i]), A0[i]))
        d = np.where(new, np.sign(bj), d); ent = np.where(start, px[:, j], ent); best = np.where(start, px[:, j], best)
        out = np.where(new, False, out); frac = np.where(new, 1.0, frac); mfe = np.where(new, 0.0, mfe)
        A0 = np.where(start, ctx.atr[:, j], A0); W0 = np.where(start, ctx.half[:, j], W0)
        pos[:, j] = np.where(out, 0.0, bj * frac); prevb = bj
        p = px[:, j]; live = d != 0
        if reenter:
            back = live & out & (d * (p - best) > 0); out = out & ~back; back_n += int(back.sum())
        intr = live & ~out
        best = np.where(intr & (d > 0), np.maximum(best, p), np.where(intr & (d < 0), np.minimum(best, p), best))
        mfe = np.where(intr, d * (best - ent), mfe); op = d * (p - ent)
        if rule is not None:
            ex, half = rule(NS(mfe=mfe, op=op, A=ctx.atr[:, j], ent=ent, fr=frac, dec=int(M.dec[j]), mdl=ctx.mdl[:, j], vw=ctx.vw[:, j], d=d, p=p, W=W0))
            out = out | (intr & ex); frac = np.where(intr & half, 0.5, frac)
    if anatomy:
        for i in np.flatnonzero(d != 0): rec.append((i, d[i], ent[i], mfe[i], d[i] * (M.exit[i] - ent[i]), A0[i]))
        return pos, pd.DataFrame(rec, columns=['day', 'side', 'entry', 'mfe', 'final', 'atr'])
    return pos


def rules():
    P = 150.0 / 27875.0; Z = lambda S: np.zeros_like(S.mfe, bool)
    gb = lambda a, g: (lambda S: ((S.mfe >= a * S.A) & (S.op <= (1 - g) * S.mfe), Z(S)))
    be = lambda a: (lambda S: ((S.mfe >= a * S.A) & (S.op <= 0), Z(S)))
    tg = lambda k: (lambda S: (S.op >= k * S.A, Z(S)))
    ch = lambda a, c: (lambda S: ((S.mfe >= a * S.A) & (S.op <= S.mfe - c * S.A), Z(S)))
    so = lambda k: (lambda S: (Z(S), (S.fr > 0.75) & (S.op >= k * S.A)))
    dl = lambda usd, g: (lambda S: ((S.mfe >= usd / 300.0 * P * S.ent) & (S.op <= (1 - g) * S.mfe), Z(S)))
    lg = lambda lo, hi, a, g: (lambda S: ((S.mfe >= a * S.A) & (S.op <= (1 - g) * S.mfe) & (lo <= S.dec < hi), Z(S)))
    lc = lambda lo, a, c: (lambda S: ((S.mfe >= a * S.A) & (S.op <= S.mfe - c * S.A) & (S.dec >= lo), Z(S)))
    qg = lambda a, g: (lambda S: ((S.mfe >= a * S.W) & (S.op <= (1 - g) * S.mfe), Z(S)))
    qt = lambda k: (lambda S: (S.op >= k * S.W, Z(S)))
    md = lambda a: (lambda S: ((S.mfe >= a * S.A) & (S.mdl * S.d < 0), Z(S)))
    vx = lambda a: (lambda S: ((S.mfe >= a * S.A) & (S.d * (S.p - S.vw) < 0), Z(S)))
    return {'G1': ('give back 50% after 1 ATR', gb(1, .5), False), 'G2': ('give back 33% after 1 ATR', gb(1, 1 / 3), False),
            'G3': ('give back 50% after 2 ATR', gb(2, .5), False), 'G4': ('give back 33% after 2 ATR', gb(2, 1 / 3), False),
            'G5': ('give back 33% after 3 ATR', gb(3, 1 / 3), False), 'E1': ('breakeven after 1 ATR', be(1), False), 'E2': ('breakeven after 2 ATR', be(2), False),
            'T1': ('target 2 ATR', tg(2), False), 'T2': ('target 3 ATR', tg(3), False), 'T3': ('target 4 ATR', tg(4), False),
            'C1': ('trail 1 ATR after 2 ATR', ch(2, 1), False), 'C2': ('trail 1.5 ATR after 2 ATR', ch(2, 1.5), False), 'C3': ('trail 1 ATR after 3 ATR', ch(3, 1), False),
            'S1': ('half off at 2 ATR', so(2), False), 'S2': ('half off at 3 ATR', so(3), False),
            'D1': ('50% give-back after $300 a micro', dl(300, .5), False), 'D2': ('50% give-back after $500 a micro', dl(500, .5), False),
            'R1': ('re-enter: 50% give-back after 2 ATR', gb(2, .5), True), 'R2': ('re-enter: 33% give-back after 2 ATR', gb(2, 1 / 3), True),
            'R3': ('re-enter: trail 1.5 ATR after 2 ATR', ch(2, 1.5), True), 'R4': ('re-enter: 50% give-back after 1 ATR', gb(1, .5), True),
            'L1': ('11:30-13:30: 33% give-back after 2 ATR', lg(120, 240, 2, 1 / 3), False), 'L2': ('from 12:00: trail 1.5 ATR after 1 ATR', lc(150, 1, 1.5), False),
            'Q1': ('50% give-back after 1 band half-width', qg(1, .5), False), 'Q2': ('target 2 band half-widths', qt(2), False),
            'K1': ('trail 2 ATR from the start', ch(0, 2), False), 'K2': ('trail 2.5 ATR from the start', ch(0, 2.5), False),
            'M1': ('model leg against, after 1 ATR', md(1), False), 'M2': ('model leg against, after 2 ATR', md(2), False),
            'V1': ('back through VWAP after 2 ATR', vx(2), False)}


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR, masterscale as MS
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0)
    gi = np.clip((M.dec - 15) // 5, 0, G.diag['atr'].shape[1] - 1); on = (M.dec >= 15)[None, :]
    slot = lambda k: np.where(on, G.diag[k][:, gi], np.nan)
    ctx = NS(atr=slot('atr'), vw=slot('vw'), half=(slot('ub') - slot('lb')) / 2, mdl=fair)
    keep = np.asarray(M.keep); A_ = keep & np.asarray(G.isA); B_ = keep & ~np.asarray(G.isA)
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    RU = rules()
    say(); say("=" * 160)
    say(f"PROFIT CAPTURE  e5q_capture.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5Q_PLAN.txt + amendment 1) {len(RU)} mechanisms on the band leg; fair model leg 1 + band leg 1 as one net "
        f"position, live cost, $ at today's price; select 2013-19, confirm 2019-26")
    say("=" * 160)
    base_pos, an = overlay(None, ctx, anatomy=True)
    assert np.array_equal(base_pos, M.band), "the overlay without a rule must reproduce the band leg exactly"
    an = an[keep[an.day.values]].copy(); an['mfe_u'] = an.mfe / an.entry * U; an['fin_u'] = an.final / an.entry * U; an['k'] = an.mfe / an.atr
    yrs = keep.sum() / 252
    say(f"ANATOMY (band leg, 1 micro, before costs, 2013-26, {len(an):,} trades = {len(an) / yrs:.0f} a year): keeps ${an.fin_u.sum() / yrs:+,.0f} a year; "
        f"given back from the trades' peaks ${(an.mfe_u - an.fin_u).sum() / yrs:,.0f} a year")
    for k in (1, 2, 3, 5):
        s = an[an.k >= k]
        say(f"   reached >= {k} ATR of open profit: {len(s) / len(an) * 100:4.1f}% of trades ({len(s) / yrs:4.0f}/yr), average peak ${s.mfe_u.mean():,.0f}, average result ${s.fin_u.mean():+,.0f}, "
            f"ended below half the peak {np.mean(s.fin_u < 0.5 * s.mfe_u) * 100:.0f}%, ended in a loss {np.mean(s.fin_u < 0) * 100:.0f}%")
    s = an[an.k < 1]; say(f"   never reached 1 ATR: {len(s) / len(an) * 100:.0f}% of trades, average result ${s.fin_u.mean():+,.0f}")
    fr = an[(an.mfe_u >= 600)]
    say(f"   trades that peaked at +$600 a micro or more (like Friday's): {len(fr) / yrs:.0f} a year, average peak ${fr.mfe_u.mean():,.0f}, average result ${fr.fin_u.mean():+,.0f}; "
        f"ended under +$300: {np.mean(fr.fin_u < 300) * 100:.0f}%, ended above +$600: {np.mean(fr.fin_u >= 600) * 100:.0f}%")
    ub, wb = days(fair + base_pos); vb, _ = days(base_pos)

    def row(lab, u, v, p):
        tr = np.abs(np.diff(np.c_[np.zeros(M.n), p, np.zeros(M.n)], axis=1)).sum(1)[keep].sum() / 2 / yrs
        return (f"   {lab:42s} | book 2013-19 Sharpe {sh(u[A_]):.2f} ${u[A_].mean() * 252:+7,.0f}/yr DD ${dd(u[A_]):6,.0f} | 2019-26 Sharpe {sh(u[B_]):.2f} ${u[B_].mean() * 252:+7,.0f}/yr DD ${dd(u[B_]):6,.0f}"
                f" | band leg alone ${v[A_].mean() * 252:+6,.0f} / ${v[B_].mean() * 252:+6,.0f} a year, {tr:4.0f} round trips/yr")
    say(); say(row('BASELINE (band leg as built)', ub, vb, base_pos))
    res = {}
    for code, (desc, fn, re) in RU.items():
        if code == 'R1': say("   -- amendment 1 (registered after the 17 above were seen) --")
        p = overlay(fn, ctx, reenter=re); u, _ = days(fair + p); v, _ = days(p); res[code] = (p, u)
        say(row(f"{code} {desc}", u, v, p) + ("  SELECTED" if sh(u[A_]) >= sh(ub[A_]) + 0.05 else ""))
    selected = [c for c in RU if sh(res[c][1][A_]) >= sh(ub[A_]) + 0.05]
    bar = 2.0 if len(selected) == 1 else 2.4 if len(selected) <= 5 else 2.6
    say(); say(f"CONFIRMATION 2019-26 for {len(selected)} selected rule(s) (bar: paired t >= {bar}):" if selected else "CONFIRMATION: nothing selected on 2013-19.")
    conf = []
    for c in selected:
        dlt = res[c][1][B_] - ub[B_]; t = dlt.mean() / dlt.std() * np.sqrt(len(dlt)) if dlt.std() > 0 else 0.0
        ok = sh(res[c][1][B_]) > sh(ub[B_]) and t >= bar
        say(f"   {c} {RU[c][0]}: Sharpe {sh(res[c][1][B_]):.2f} vs {sh(ub[B_]):.2f}, ${dlt.mean() * 252:+,.0f}/yr vs baseline, max DD ${dd(res[c][1][B_]):,.0f} vs ${dd(ub[B_]):,.0f}, paired t {t:+.2f} -> "
            f"{'CONFIRMED' if ok else 'not confirmed'}")
        if ok: conf.append(c)
    best = max(RU, key=lambda c: sh(res[c][1][A_])); bestB = max(RU, key=lambda c: sh(res[c][1][B_]))
    show = [('baseline', base_pos)] + [(c, res[c][0]) for c in dict.fromkeys([best, bestB] + conf)]
    say(); say(f"PROJECTION (2019-26 sessions, bootstrap as in e5p_predict.py): baseline, {best} (best on 2013-19), {bestB} (best on 2019-26, hindsight)" + (", confirmed: " + ", ".join(conf) if conf else ""))
    for lab, p in show:
        uE, wE = days(fair + 6 * p); uE, wE = uE[B_], wE[B_]; ntE = (uE != 0).astype(float)
        XE, WE, _ = BL.paths(BL.N_WIN, uE, wE, ntE, 1, BL.SEED_REPORT); ok, nres = BL.evaluation(XE, WE, a['dd'], a['dll'], a['target'], resets=True)
        pe = ok.mean(); fees = (a['qual'] + BL.RESET_FEE * nres.mean()) / pe
        uF, wF = days(fair + p); uF, wF = uF[B_], wF[B_]; ntF = (uF != 0).astype(float); XF, WF, NT = BL.paths(BL.N_FUNDED, uF, wF, ntF, 1, BL.SEED_REPORT)
        out = []
        for nm, rule in (('fixed', lambda cu: np.full(cu.shape, 1.0)), ('cushion max 4', lambda cu: np.clip(np.floor(cu / 2500), 1, 4))):
            pay, ou, mic = MS.master_dyn(XF, WF, NT, a, rule); tot = pay.sum(1)
            out.append(f"funded 1 + 1 {nm}: 2-yr payouts ${tot.mean():,.0f} (median ${np.median(tot):,.0f}), any payout {np.mean(tot > 0) * 100:.0f}%, open after 1 yr {np.mean(ou > 251) * 100:.0f}%")
        say(f"   {lab:9s}: book ${uF.mean() * 252:+,.0f}/yr, max DD ${dd(uF):,.0f}, worst day ${uF.min():,.0f} | evaluation 1 + 6: {pe * 100:.0f}% a try (${fees:,.0f} a pass) | " + " | ".join(out))
    say(); say(f"VERDICT: {'ADOPT ' + ', '.join(conf) if conf else 'no profit-capture mechanism confirmed - the band leg keeps its exits'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
