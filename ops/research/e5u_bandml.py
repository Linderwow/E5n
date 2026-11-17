r"""E5U (E5U_PLAN.txt): a trained filter for the band leg's entries and a one-indicator-at-a-time sweep, in the two-leg book.
    python e5u_bandml.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
from e5b_lab import G, say
X = L.X; U = L.U; ND = L.ND
COMPACT = ['side', 'brk', 'dvw', 'dema', 'half', 'dec', 'df5', 'df15', 'df30', 'df60', 'dr5', 'dr15', 'dr30', 'dr60', 'dropen', 'dgap', 'vol30', 'vratio', 'adx5', 'adx15', 'adx30', 'atrp5', 'prng', 'nprev', 'lastsign']


def trades():
    M = C.M; nd = M.band.shape[1]; c1 = X.SIDE(2.35) * U; FE = G.T[X.F.FEATS].values; rows = []
    for r in range(M.n):
        p = M.band[r]; ch = np.flatnonzero(np.diff(np.r_[0.0, p]) != 0); nprev = 0; last = 0.0
        for a_, j0 in enumerate(ch):
            if p[j0] == 0: continue
            j1 = int(ch[a_ + 1]) if a_ + 1 < len(ch) else nd; d = float(np.sign(p[j0])); gi = (int(j0) - 10) // 5
            if (int(j0) - 10) % 5 or not 0 <= gi < ND: continue
            pe = M.px[r, j1] if j1 < nd else M.exit[r]; usd = d * (pe - M.px[r, j0]) / M.o930[r] * U - 2 * c1
            D = {k: G.diag[k][r, gi] for k in ('close', 'ub', 'lb', 'vw', 'ema', 'atr')}; f = FE[r * ND + gi]
            edge = D['ub'] if d > 0 else D['lb']; atr = D['atr']
            comp = [d, d * (D['close'] - edge) / atr, d * (D['close'] - D['vw']) / atr, d * (D['close'] - D['ema']) / atr, (D['ub'] - D['lb']) / 2 / atr, j0 + 5,
                    d * G.f5[r, gi], d * G.f15[r, gi], d * G.f30[r, gi], d * G.f60[r, gi]]
            rows.append([r, int(j0), j1, d, usd] + comp + [nprev, last] + list(f)); nprev += 1; last = float(np.sign(usd))
    cols = ['r', 'j0', 'j1', 'd', 'usd', 'side', 'brk', 'dvw', 'dema', 'half', 'dec', 'df5', 'df15', 'df30', 'df60', 'nprev', 'lastsign'] + list(X.F.FEATS)
    T = pd.DataFrame(rows, columns=cols); T['date'] = C.M.dates[T.r.values]
    for k, src in (('dr5', 'r5'), ('dr15', 'r15'), ('dr30', 'r30'), ('dr60', 'r60'), ('dropen', 'ropen'), ('dgap', 'gap')): T[k] = T.d * T[src]
    return T


def walk(T, cols, kind, seed=0, shuffle=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    pred = np.full(len(T), np.nan); mean_tr = np.full(len(T), np.nan); yr = T.date.dt.year.values; rng = np.random.default_rng(seed)
    for y in range(2013, int(yr.max()) + 1):
        tr = yr < y; te = yr == y
        if te.sum() == 0 or tr.sum() < 200: continue
        Xtr = T.loc[tr, cols].values.astype(float); ytr = np.clip(T.usd.values[tr], -1500, 1500); Xte = T.loc[te, cols].values.astype(float)
        if shuffle: ytr = rng.permutation(ytr)
        if kind == 'hgb':
            use = np.isfinite(Xtr).sum(0) >= 50; Xtr = Xtr[:, use]; Xte = Xte[:, use]
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.03, min_samples_leaf=40, l2_regularization=1.0, random_state=0, early_stopping=False).fit(Xtr, ytr)
            pred[te] = m.predict(Xte)
        else:
            use = np.isfinite(Xtr).sum(0) >= 50; Xtr = Xtr[:, use]; Xte = Xte[:, use]
            med = np.nanmedian(Xtr, axis=0); Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
            m = Ridge(alpha=10.0).fit((Xtr - mu) / sd, ytr); pred[te] = m.predict((Xte - mu) / sd)
        mean_tr[te] = ytr.mean()
    return pred, mean_tr


def path_with(T, size):
    M = C.M; p = M.band.copy()
    for r, j0, j1, s in zip(T.r.values, T.j0.values, T.j1.values, size):
        if s != 1.0: p[r, j0:j1] *= s
    return p


def main():
    t0 = time.time(); C.load(); M = C.M
    keep = np.asarray(M.keep); isA = np.asarray(G.isA); A_ = keep & isA; B_ = keep & ~isA
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0)
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    T = trades(); ALL = COMPACT + [f for f in X.F.FEATS if f not in COMPACT]
    ub, _ = Q.days(fair + M.band); vb, _ = Q.days(M.band)
    tA = (T.date >= '2013-01-01') & (T.date <= L.JOIN); tB = T.date > L.JOIN; yA = A_.sum() / 252; yB = B_.sum() / 252
    say(); say("=" * 160)
    say(f"BAND-LEG ENTRY FILTER  e5u_bandml.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5U_PLAN.txt) {len(T):,} band trades 2010-26 ({tA.sum():,} in 2013-19, {tB.sum():,} in 2019-26); walk-forward by year; "
        f"fair model leg 1 + band leg 1, live cost, $ at today's price")
    say("=" * 160)
    say(f"   {'BASELINE (every band trade taken)':52s} | book 2013-19 Sharpe {sh(ub[A_]):.2f} ${ub[A_].mean() * 252:+7,.0f}/yr DD ${dd(ub[A_]):6,.0f} | 2019-26 Sharpe {sh(ub[B_]):.2f} ${ub[B_].mean() * 252:+7,.0f}/yr DD ${dd(ub[B_]):6,.0f} | "
        f"average trade ${T.usd[tA].mean():+.0f} / ${T.usd[tB].mean():+.0f}")
    SPEC = {'U1': ('boosted trees, all 107 indicators, skip below $0', ALL, 'hgb', 'skip'), 'U2': ('boosted trees, compact set, skip below $0', COMPACT, 'hgb', 'skip'),
            'U3': ('ridge regression, compact set, skip below $0', COMPACT, 'ridge', 'skip'), 'U4': ('boosted trees, compact set, size 0 / 1 / 2', COMPACT, 'hgb', 'size')}

    def sized(pred, mean_tr, how):
        s = np.where(np.isnan(pred), 1.0, np.where(pred < 0, 0.0, 1.0))
        if how == 'size': s = np.where(~np.isnan(pred) & (pred > 2 * mean_tr), 2.0, s)
        return s
    res = {}
    for code, (desc, cols, kind, how) in SPEC.items():
        pred, mtr = walk(T, cols, kind); s = sized(pred, mtr, how); u, _ = Q.days(fair + path_with(T, s)); res[code] = (u, s, pred)
        sk = s == 0; line = []
        for lab, m_, y_ in (('2013-19', tA, yA), ('2019-26', tB, yB)):
            line.append(f"{lab}: skips {(sk & m_).sum() / y_:3.0f}/yr, skipped avg ${T.usd[sk & m_].mean() if (sk & m_).any() else 0:+5.0f}, kept avg ${T.usd[~sk & m_].mean():+5.0f}")
        c = np.corrcoef(pred[tA.values & ~np.isnan(pred)], T.usd[tA.values & ~np.isnan(pred)])[0, 1], np.corrcoef(pred[tB.values & ~np.isnan(pred)], T.usd[tB.values & ~np.isnan(pred)])[0, 1]
        say(f"   {code} {desc:49s} | book 2013-19 Sharpe {sh(u[A_]):.2f} ${u[A_].mean() * 252:+7,.0f}/yr DD ${dd(u[A_]):6,.0f} | 2019-26 Sharpe {sh(u[B_]):.2f} ${u[B_].mean() * 252:+7,.0f}/yr DD ${dd(u[B_]):6,.0f} | "
            + " | ".join(line) + f" | forecast vs result correlation {c[0]:+.3f} / {c[1]:+.3f}" + ("  SELECTED" if sh(u[A_]) >= sh(ub[A_]) + 0.05 else ""))
    selected = [c for c in SPEC if sh(res[c][0][A_]) >= sh(ub[A_]) + 0.05]; bar = 2.0 if len(selected) == 1 else 2.4
    say(); say(f"CONFIRMATION 2019-26 for {len(selected)} selected filter(s) (bar: paired t >= {bar}):" if selected else "CONFIRMATION: no trained filter selected on 2013-19.")
    conf = []
    for c in selected:
        u = res[c][0]; dlt = u[B_] - ub[B_]; ok = sh(u[B_]) > sh(ub[B_]) and tt(dlt) >= bar
        say(f"   {c}: Sharpe {sh(u[B_]):.2f} vs {sh(ub[B_]):.2f}, ${dlt.mean() * 252:+,.0f}/yr, max DD ${dd(u[B_]):,.0f} vs ${dd(ub[B_]):,.0f}, paired t {tt(dlt):+.2f} -> {'CONFIRMED' if ok else 'not confirmed'}")
        desc, cols, kind, how = SPEC[c]; pl = []
        for sd_ in range(1, 6):
            pp, mm = walk(T, cols, kind, seed=sd_, shuffle=True); up, _ = Q.days(fair + path_with(T, sized(pp, mm, how))); pl.append((sh(up[A_]), sh(up[B_])))
        say(f"      placebos (trained on shuffled results): book Sharpe 2013-19 {min(x[0] for x in pl):.2f} .. {max(x[0] for x in pl):.2f}, 2019-26 {min(x[1] for x in pl):.2f} .. {max(x[1] for x in pl):.2f}")
        beat = sh(u[A_]) > max(x[0] for x in pl) and sh(u[B_]) > max(x[1] for x in pl)
        if ok and beat:
            d = pd.Series((u - ub)[keep], index=M.dates[keep]); yr = d.groupby(d.index.year).sum(); dB = d[d.index > L.JOIN]; yB_ = dB.groupby(dB.index.year).sum()
            k2 = (yr > 0).sum() >= 8 and dB[dB.index.year != yB_.idxmax()].sum() > 0; k3 = dB.drop(dB.nlargest(5).index).sum() >= 0
            say(f"      years positive {(yr > 0).sum()} of {len(yr)}, without the best year ${dB[dB.index.year != yB_.idxmax()].sum():+,.0f} -> {'PASS' if k2 else 'FAIL'}; without the 5 best days ${dB.drop(dB.nlargest(5).index).sum():+,.0f} -> {'PASS' if k3 else 'FAIL'}")
            if k2 and k3: conf.append(c)
        elif ok: say("      does not beat its placebos on both halves -> not adopted")
    say(); say("PART 2 - ONE INDICATOR AT A TIME (skip the band trades in a cell; cuts from the 2013-19 trades):")
    cells = []; dA = T[tA]; dB_ = T[tB]
    for f in X.F.FEATS:
        for side, ms in (('longs', 1.0), ('shorts', -1.0), ('all', 0.0)):
            a = dA if ms == 0 else dA[dA.d == ms]; v = a[f].values; v = v[np.isfinite(v)]
            if len(v) < 60 or len(np.unique(v)) < 3: continue
            q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
            for nm, lo, hi in (('low third', -np.inf, q1), ('middle third', q1, q2), ('high third', q2, np.inf)):
                m = (a[f].values > lo) & (a[f].values <= hi)
                if m.sum() / yA < 10: continue
                x = np.bincount(a.r.values[m], weights=-a.usd.values[m], minlength=M.n)[A_]; cells.append((tt(x), f, side, nm, lo, hi, ms, m.sum() / yA, x.mean() * 252))
    ts = np.array([c[0] for c in cells])
    say(f"   {len(cells)} cells with at least 10 trades a year; t of the P&L saved by skipping: best {ts.max():+.2f}, median {np.median(ts):+.2f}, worst {ts.min():+.2f}; above +2: {(ts > 2).sum()}, above +3: {(ts > 3).sum()}, above the bar +3.8: {(ts > 3.8).sum()}")
    say("   the 10 best cells (2013-19 | the same cell on 2019-26; a candidate needs t >= 3.8 and then t >= 2.6):"); cand = []
    for t_, f, side, nm, lo, hi, ms, n_, usd_ in sorted(cells, key=lambda c: -c[0])[:10]:
        b = dB_ if ms == 0 else dB_[dB_.d == ms]; m = (b[f].values > lo) & (b[f].values <= hi); x = np.bincount(b.r.values[m], weights=-b.usd.values[m], minlength=M.n)[B_]
        ok = t_ >= 3.8 and tt(x) >= 2.6
        say(f"      skip {side:6s} with {f:8s} in its {nm:12s}: 2013-19 {n_:3.0f} trades/yr, saves ${usd_:+6,.0f}/yr, t {t_:+.2f} | 2019-26 {m.sum() / yB:3.0f} trades/yr, saves ${x.mean() * 252:+6,.0f}/yr, t {tt(x):+.2f}  {'CANDIDATE' if ok else ''}")
        if ok: cand.append((f, side, nm))
    say(); say(f"VERDICT (band-leg filters): trained filter: {'ADOPT ' + ', '.join(conf) if conf else 'none confirmed'}; single indicators: {len(cand)} candidate(s)")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
