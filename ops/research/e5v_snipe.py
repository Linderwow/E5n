r"""E5V amendments 1 + 2 (E5V_PLAN.txt): higher-timeframe fair-value-gap zones with a 1-minute entry ("snipe"), 12 registered entries and
the indicator-combination sweep. Discovery NQ 2013-19, holdout MNQ 2019-26.
    python e5v_snipe.py
"""
import os, sys, time, itertools
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5v_struct as ST
from e5b_lab import G, say
X = L.X; U = L.U; TICK = ST.TICK
LBX = {15: 4320, 60: 10080, 240: 28800}
IND = {'trend': ('with', 'against'), 'ema20': ('with', 'against'), 'vwap': ('with', 'against'), 'daily': ('with', 'against'), 'rsi': ('pullback', 'momentum'),
       'adx': ('over 25', 'under 20'), 'vol': ('above', 'below'), 'time': ('09:30-11:00', '11:00-13:30', '13:30-15:30'), 'open': ('discount', 'premium'),
       'model': ('same side', 'opposite', 'flat'), 'band': ('same side', 'opposite', 'flat')}


def indicators(E, v):
    B = E.bars[5]; c = pd.Series(B.c); I = {}
    I['trend'] = np.sign(c.ewm(span=50, adjust=False).mean() - c.ewm(span=200, adjust=False).mean()).values
    I['ema20'] = c.ewm(span=20, adjust=False).mean().values
    dl = c.diff(); up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    I['rsi'] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).values
    h, l = pd.Series(B.h), pd.Series(B.l); um = h.diff(); dm = -l.diff()
    pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0)).ewm(alpha=1 / 14, adjust=False).mean(); ndm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    dx = 100 * (pdm - ndm).abs() / (pdm + ndm).replace(0, np.nan); I['adx'] = dx.ewm(alpha=1 / 14, adjust=False).mean().values
    I['atr'] = B.atr; I['atrmed'] = pd.Series(B.atr).rolling(5520, min_periods=1000).median().values
    rth = (E.md >= 570) & (E.md < 960); tp = (E.h + E.l + E.c) / 3
    df = pd.DataFrame({'d': E.dd, 'pv': np.where(rth, tp * v, 0.0), 'v': np.where(rth, v, 0.0)}); cs = df.groupby('d').cumsum()
    I['vwap'] = np.where(rth & (cs.v.values > 0), cs.pv.values / np.maximum(cs.v.values, 1e-9), np.nan)
    first = pd.DataFrame({'d': E.dd[rth], 'o': E.o[rth], 'c': E.c[rth]}).groupby('d').agg(o=('o', 'first'), c=('c', 'last'))
    I['dayopen'] = first.o.to_dict(); sma = first.c.rolling(20).mean()
    I['daily'] = np.sign(first.c - sma).shift(1).to_dict()
    return I


def opposing(E, tau, d, e, R):
    W = ST.window(E, tau)
    if W is None: return np.nan
    S = E.sw; kl = E.kl.get(int(tau // 1440), np.full(7, np.nan))
    if d > 0:
        i0 = np.searchsorted(S.ht, tau - ST.SWING_LB, 'right'); i1 = np.searchsorted(S.ht, tau, 'right'); p = S.hp[i0:i1]; p = p[ST.after(W, S.hb[i0:i1], 'max') <= p]
        c = np.r_[p, kl]; c = c[np.isfinite(c) & (c >= e + R) & (c <= e + 5 * R)]; return c.min() if len(c) else e + 3 * R
    i0 = np.searchsorted(S.lt, tau - ST.SWING_LB, 'right'); i1 = np.searchsorted(S.lt, tau, 'right'); p = S.lp[i0:i1]; p = p[ST.after(W, S.lb[i0:i1], 'min') >= p]
    c = np.r_[p, kl]; c = c[np.isfinite(c) & (c <= e - R) & (c >= e - 5 * R)]; return c.max() if len(c) else e - 3 * R


def generate(E, htf, trig, t_from):
    F = E.fvg[htf]; F1 = E.fvg[1]; out = []
    for q in np.flatnonzero(F.t >= t_from):
        ft = int(F.t[q]); d = float(F.typ[q]); zlo, zhi = F.lo[q], F.hi[q]
        i0 = np.searchsorted(E.t, ft); i1 = np.searchsorted(E.t, ft + LBX[htf])
        if i1 - i0 < 2: continue
        hit = (E.l[i0:i1] <= zhi - TICK) if d > 0 else (E.h[i0:i1] >= zlo + TICK)
        if not hit.any(): continue
        p = i0 + int(hit.argmax())
        if not (570 <= E.md[p] < 930): continue
        stop = zlo - TICK if d > 0 else zhi + TICK
        if trig == 'L':
            pe = p; e = min(zhi, E.o[p]) if d > 0 else max(zlo, E.o[p])
        else:
            j = np.searchsorted(F1.t, E.t[p], 'right'); pe = -1
            while j < len(F1.t) and F1.t[j] <= E.t[p] + 30:
                if F1.typ[j] == d:
                    k = np.searchsorted(E.t, F1.t[j])
                    if k < len(E.t) and E.t[k] - F1.t[j] <= 2 and E.dd[k] == E.dd[p] and E.md[k] < 930:
                        if (E.l[p:k].min() > zlo) if d > 0 else (E.h[p:k].max() < zhi): pe = k
                    break
                j += 1
            if pe < 0: continue
            e = E.o[pe]
        R = d * (e - stop)
        if not R > 0: continue
        A = ST.atr5(E, int(E.t[pe]))
        if not R >= 0.25 * A: continue
        out.append((pe, d, e, stop, R))
    return out


def finish(E, cands, tgt, trig):
    rows = []
    for pe, d, e, stop, R in cands:
        tau = int(E.t[pe]); tau1 = int(E.dd[pe]) * 1440 + 960
        T = e + 2 * d * R if tgt == 'T2' else opposing(E, tau, d, e, R)
        x = ST.sim_exit(E, tau, tau1, d, T, stop, None, False, entry_bar_target=(trig == 'C'))
        if x is None:
            ie = np.searchsorted(E.t, tau1) - 1; te, Fp = int(E.t[ie]), float(E.c[ie])
        else: te, Fp, _ = x
        rows.append((pe, tau, te, d, e, Fp, R))
    return np.array(rows, float) if rows else np.zeros((0, 7))


def one_at_a_time(T):
    keep = np.zeros(len(T), bool); busy = -1.0
    for i in np.argsort(T[:, 1], kind='stable'):
        if T[i, 1] > busy: keep[i] = True; busy = T[i, 2]
    return keep


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import src2_lib as L2
    ERA = {'A': ST.load(L2, 'nq'), 'B': ST.load(L2, 'mnqx')}; VOL = {'A': np.load(L2.CACHE('nq'))['v'].astype(float), 'B': np.load(L2.CACHE('mnqx'))['v'].astype(float)}
    INDV = {k: indicators(ERA[k], VOL[k]) for k in ERA}
    keep = np.asarray(M.keep); isA = np.asarray(G.isA); A_ = keep & isA; B_ = keep & ~isA
    day0 = M.dates.values.astype('datetime64[m]').astype(np.int64) // 1440; rowof = {int(d): i for i, d in enumerate(day0)}
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); nd = fair.shape[1]
    c1 = X.SIDE(2.35) * U; t13 = int(np.datetime64('2013-01-01', 'm').astype(np.int64))
    tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())

    def table(E, T):
        pe = T[:, 0].astype(int); d = T[:, 3]; e = T[:, 4]; r = np.array([rowof.get(int(x), -1) for x in E.dd[pe]])
        gross = d * (T[:, 5] - e) / (e - E.shift[pe]) * U
        return dict(T=T, pe=pe, d=d, e=e, r=r, gross=gross, net=gross - 2 * c1, Rm=d * (T[:, 5] - e) / T[:, 6])

    def states(E, I, tb):
        pe, d, e, r = tb['pe'], tb['d'], tb['e'], tb['r']; B = E.bars[5]; i5 = np.searchsorted(B.tend, E.t[pe], 'right') - 1; md = E.md[pe]
        j = np.clip(md - 575, 0, nd - 1); rr = np.maximum(r, 0)
        S = {}
        S['trend'] = np.where(d * I['trend'][i5] > 0, 0, 1); S['ema20'] = np.where(d * (e - I['ema20'][i5]) > 0, 0, 1)
        vw = I['vwap'][np.maximum(pe - 1, 0)]; S['vwap'] = np.where(np.isnan(vw), -1, np.where(d * (e - vw) > 0, 0, 1))
        dly = np.array([I['daily'].get(int(x), np.nan) for x in E.dd[pe]]); S['daily'] = np.where(np.isnan(dly) | (dly == 0), -1, np.where(d * dly > 0, 0, 1))
        rsi = I['rsi'][i5]; S['rsi'] = np.where(((d > 0) & (rsi < 45)) | ((d < 0) & (rsi > 55)), 0, np.where(((d > 0) & (rsi > 55)) | ((d < 0) & (rsi < 45)), 1, -1))
        adx = I['adx'][i5]; S['adx'] = np.where(adx > 25, 0, np.where(adx < 20, 1, -1))
        S['vol'] = np.where(np.isnan(I['atrmed'][i5]), -1, np.where(I['atr'][i5] > I['atrmed'][i5], 0, 1))
        S['time'] = np.where(md < 660, 0, np.where(md < 810, 1, 2))
        op = np.array([I['dayopen'].get(int(x), np.nan) for x in E.dd[pe]]); S['open'] = np.where(np.isnan(op), -1, np.where(d * (e - op) < 0, 0, 1))
        ms = fair[rr, j] * d * (md >= 575); S['model'] = np.where(ms > 0, 0, np.where(ms < 0, 1, 2))
        bs = M.band[rr, j] * d * (md >= 575); S['band'] = np.where(bs > 0, 0, np.where(bs < 0, 1, 2))
        return S

    def daily(tb, mask, sel):
        ok = mask & (tb['r'] >= 0)
        return np.bincount(tb['r'][ok], weights=tb['net'][ok], minlength=M.n)[sel]

    def card(tb, mask, sel, zero=False):
        ok = mask & (tb['r'] >= 0) & sel[np.maximum(tb['r'], 0)]
        x = np.bincount(tb['r'][ok], weights=(tb['gross'] if zero else tb['net'])[ok], minlength=M.n)[sel]; n = ok.sum(); yrs = sel.sum() / 252
        s = pd.Series(x, index=M.dates[sel]); yr = s.groupby(s.index.year).sum(); x5 = s.drop(s.nlargest(5).index).values
        return dict(t=tt(x), usd=x.mean() * 252, dd=dd(x), n=n / yrs, win=float((tb['net'][ok] > 0).mean()) if n else 0.0, Rm=float(tb['Rm'][ok].mean()) if n else 0.0, neg=int((yr < 0).sum()), ny=len(yr), t5=tt(x5))

    say(); say("=" * 160)
    say(f"HIGHER-TIMEFRAME GAP, 1-MINUTE ENTRY  e5v_snipe.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5V_PLAN.txt amendments 1 + 2) 1 micro, $ at today's price, live cost on every order; discovery NQ 2013-19, holdout MNQ 2019-26")
    say("=" * 160)
    BASE = {}; names = {}
    for htf in (15, 60, 240):
        for trig in ('L', 'C'):
            cand = {k: generate(ERA[k], htf, trig, t13) for k in ERA}
            for tgt in ('T2', 'TZ'):
                code = f"H{htf}-{trig}-{tgt}"; names[code] = f"{htf}-minute gap, {'limit at the edge' if trig == 'L' else '1-minute gap confirmation'}, target {'2 R' if tgt == 'T2' else 'opposing zone'}"
                BASE[code] = {}
                for k in ERA:
                    tb = table(ERA[k], finish(ERA[k], cand[k], tgt, trig)); tb['S'] = states(ERA[k], INDV[k], tb); BASE[code][k] = tb
        say(f"   built {htf}-minute zones ({time.time() - t0:.0f} s)")
    say(); say("THE 12 REGISTERED ENTRIES, one position at a time - DISCOVERY NQ 2013-19 (kill: t < 3.0, > 1 losing year, drop-5 t < 2.0, < 50 trades a year)")
    say(f"   {'entry':66s} | trades/yr | wins | avg result in R | $/yr live | t live | t zero cost | losing yrs | drop-5 t | max DD")
    surv = []
    for code, eras in BASE.items():
        tb = eras['A']; m1 = one_at_a_time(tb['T']); k = card(tb, m1, A_); z = card(tb, m1, A_, zero=True)
        ok = k['t'] >= 3.0 and k['neg'] <= 1 and k['t5'] >= 2.0 and k['n'] >= 50
        say(f"   {code:11s} {names[code]:54s} | {k['n']:9.0f} | {k['win'] * 100:3.0f}% | {k['Rm']:+15.2f} | ${k['usd']:+8,.0f} | {k['t']:+6.2f} | {z['t']:+11.2f} | {k['neg']:3d} of {k['ny']} | {k['t5']:+8.2f} | ${k['dd']:,.0f}  {'PASS' if ok else 'dead'}")
        if ok: surv.append(code)
    say("   holdout for the 12: " + ("opened for " + ", ".join(surv) if surv else "not opened - none passed discovery"))
    for code in surv:
        tb = BASE[code]['B']; k = card(tb, one_at_a_time(tb['T']), B_)
        say(f"   HOLDOUT {code}: {k['n']:.0f} trades/yr, ${k['usd']:+,.0f}/yr, t {k['t']:+.2f}, losing years {k['neg']} of {k['ny']}, max DD ${k['dd']:,.0f}")
    cells = []
    single = [(ind, s) for ind in IND for s in range(len(IND[ind]))]
    pairs = [(a, b) for a, b in itertools.combinations(single, 2) if a[0] != b[0]]
    yrsA = A_.sum() / 252
    for code, eras in BASE.items():
        tb = eras['A']; inA = (tb['r'] >= 0) & A_[np.maximum(tb['r'], 0)]
        for combo in [()] + [(s,) for s in single] + pairs:
            mask = inA.copy()
            for ind, s in combo: mask &= tb['S'][ind] == s
            n = mask.sum()
            if n / yrsA < 50: cells.append((code, combo, -9.0, n / yrsA)); continue
            x = np.bincount(tb['r'][mask], weights=tb['net'][mask], minlength=M.n)[A_]; cells.append((code, combo, tt(x), n / yrsA))
    ts = np.array([c[2] for c in cells]); live = ts > -9
    say(); say(f"COMBINATION SWEEP - DISCOVERY NQ 2013-19: {len(cells):,} cells (12 entries x (1 + {len(single)} single states + {len(pairs)} pairs)); {int(live.sum()):,} have at least 50 trades a year")
    say(f"   t of daily P&L at live cost across those cells: best {ts[live].max():+.2f}, median {np.median(ts[live]):+.2f}, worst {ts[live].min():+.2f}; above +2: {int((ts[live] > 2).sum())}, above +3: {int((ts[live] > 3).sum())}, "
        f"above the bar +4.2: {int((ts[live] > 4.2).sum())}; positive at all: {int((ts[live] > 0).sum())} ({(ts[live] > 0).mean() * 100:.0f}%)")
    lab = lambda combo: 'no filter' if not combo else ' + '.join(f"{ind} {IND[ind][s]}" for ind, s in combo)
    top = sorted([c for c in cells if c[2] > -9], key=lambda c: -c[2])[:10]
    say("   THE 10 BEST CELLS, re-run one position at a time (discovery | holdout MNQ 2019-26; a candidate needs discovery t >= 4.2, <= 1 losing year, drop-5 t >= 2.5 AND holdout t >= 2.6):")
    cands = []
    for code, combo, t_, n_ in top:
        out = []
        for k, sel in (('A', A_), ('B', B_)):
            tb = BASE[code][k]; mask = np.ones(len(tb['r']), bool)
            for ind, s in combo: mask &= tb['S'][ind] == s
            T = tb['T'][mask]; m1 = np.zeros(len(tb['r']), bool); idx = np.flatnonzero(mask); m1[idx[one_at_a_time(T)]] = True if len(T) else False
            out.append(card(tb, m1, sel))
        a, b = out; ok = a['t'] >= 4.2 and a['neg'] <= 1 and a['t5'] >= 2.5 and a['n'] >= 50 and b['t'] >= 2.6
        say(f"   {code:11s} {lab(combo):48s} | 2013-19: {a['n']:4.0f} trades/yr, ${a['usd']:+7,.0f}/yr, t {a['t']:+5.2f}, DD ${a['dd']:6,.0f}, $/DD {a['usd'] / max(a['dd'], 1):4.2f}, losing yrs {a['neg']} | "
            f"2019-26: {b['n']:4.0f}/yr, ${b['usd']:+7,.0f}/yr, t {b['t']:+5.2f}, DD ${b['dd']:6,.0f}, losing yrs {b['neg']} of {b['ny']}  {'CANDIDATE' if ok else 'not confirmed' if a['t'] >= 4.2 else 'below the bar'}")
        if ok: cands.append((code, combo))
    say(); say(f"VERDICT (gap entries): 12 registered entries: {', '.join(surv) if surv else 'none passes'}; sweep: {len(cands)} candidate(s)" + (': ' + '; '.join(f'{c} {lab(cb)}' for c, cb in cands) if cands else ''))
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
