r"""E5Z (E5Z_PLAN.txt): conviction sizing - bigger size on the most probable trades, fixed tiers and sizes learned from earlier years.
    python e5z_sizing.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5q_capture as Q
import e5u_bandml as UU
from e5b_lab import G, say
X = L.X; U = L.U; ND = L.ND; T0 = L.T0


def sized_path(f5, f15, size, decide):
    n, nm = f5.shape; pos = np.zeros(n); out = np.zeros((n, nm))
    for j in range(nm):
        if decide[j]:
            a = f5[:, j]; b = f15[:, j]; ok = ~np.isnan(a)
            with np.errstate(invalid='ignore'):
                t = np.where(a > T0, 1.0, np.where(a < -T0, -1.0, 0.0)); t = np.where(b * t > 0, t, 0.0)
            pos = np.where(ok, t * size[:, j], pos)
        out[:, j] = pos
    return out


def main():
    t0 = time.time(); C.load(); M = C.M; nm = M.px.shape[1]
    keep = np.asarray(M.keep); isA = np.asarray(G.isA); A_ = keep & isA; B_ = keep & ~isA; yr = M.dates.year.values
    clk0 = (M.dec % 5 == 0) & (M.dec >= 15); cols = np.flatnonzero(clk0); gi = (M.dec[cols] - 15) // 5
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0
    tt = lambda x: x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0
    dd = lambda x: float((np.maximum.accumulate(np.cumsum(x)) - np.cumsum(x)).max())
    one = np.ones((M.n, nm))
    fair = sized_path(M.a5, M.a15, one, clk0); assert np.array_equal(fair, C.path(M.a5, M.a15, 'set', clk0))
    dep = sized_path(M.g5, M.g15, one, clk0)
    f5, f15 = M.a5, M.a15
    with np.errstate(invalid='ignore'):
        sig = (np.abs(f5) > T0) & (f15 * f5 > 0) & clk0[None, :]
    fwd = np.full((M.n, nm), np.nan); pxe = np.c_[M.px, M.exit]
    for j in cols: fwd[:, j] = pxe[:, min(j + 5, nm)] / M.px[:, j] - 1
    res = np.sign(f5) * fwd
    S = {}
    with np.errstate(invalid='ignore'):
        a = np.abs(f5) / T0
        S['Z1'] = np.where(a > 3, 3.0, np.where(a > 2, 2.0, 1.0))
    z2 = one.copy(); z3 = one.copy(); z5 = one.copy(); z6 = one.copy(); FE = G.T[X.F.FEATS].values
    from sklearn.ensemble import HistGradientBoostingRegressor
    for y in range(2014, int(yr.max()) + 1):
        tr = (yr < y) & keep; te = yr == y
        if not te.any() or not tr.any(): continue
        m15 = np.nanmedian(np.abs(f15[tr][sig[tr]]))
        with np.errstate(invalid='ignore'):
            z2[te] = np.where(np.abs(f15[te]) > m15, 2.0, 1.0); z3[te] = np.where((np.abs(f5[te]) > 2 * T0) & (np.abs(f15[te]) > m15), 2.0, 1.0)
        a_tr = np.abs(f5[tr][sig[tr]]); b_tr = np.abs(f15[tr][sig[tr]]); r_tr = res[tr][sig[tr]]; okr = np.isfinite(r_tr)
        qa = np.quantile(a_tr, [.2, .4, .6, .8]); qb = np.quantile(b_tr, [1 / 3, 2 / 3]); ca = np.digitize(a_tr, qa); cb = np.digitize(b_tr, qb); mall = np.nanmean(r_tr)
        tab = np.ones((5, 3))
        for i in range(5):
            for k in range(3):
                m = (ca == i) & (cb == k) & okr
                if m.sum() >= 50:
                    rel = r_tr[m].mean() / mall if mall > 0 else 1.0; tab[i, k] = 0.0 if rel <= 0 else 1.0 if rel < 1.5 else 2.0 if rel < 2.5 else 3.0
        with np.errstate(invalid='ignore'):
            z5[te] = tab[np.digitize(np.nan_to_num(np.abs(f5[te])), qa), np.digitize(np.nan_to_num(np.abs(f15[te])), qb)]
        rr, jj = np.nonzero(sig & tr[:, None]); gg = (M.dec[jj] - 15) // 5; ok = np.isfinite(res[rr, jj])
        Xtr = np.c_[FE[rr[ok] * ND + gg[ok]], f5[rr[ok], jj[ok]], f15[rr[ok], jj[ok]]]; ytr = res[rr[ok], jj[ok]]
        use = np.isfinite(Xtr).sum(0) >= 200
        mdl = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.03, min_samples_leaf=200, l2_regularization=1.0, random_state=0, early_stopping=False).fit(Xtr[:, use], ytr)
        r2, j2 = np.nonzero(sig & te[:, None])
        if len(r2):
            g2 = (M.dec[j2] - 15) // 5; pr = mdl.predict(np.c_[FE[r2 * ND + g2], f5[r2, j2], f15[r2, j2]][:, use]); my = ytr.mean()
            z6[r2, j2] = np.where(pr < 0, 0.0, np.where(pr > 2 * my, 2.0, 1.0))
    S['Z2'] = z2; S['Z3'] = z3; S['Z5'] = z5; S['Z6'] = z6
    z4 = one.copy()
    with np.errstate(invalid='ignore'):
        for c_, g_ in zip(cols, gi):
            s5 = np.sign(M.g5[:, c_]); z4[:, c_] = 1.0 + (G.f30[:, g_] * s5 > 0) + (G.f60[:, g_] * s5 > 0)
    T = UU.trades(); ty = T.date.dt.year.values; b7 = np.ones(len(T)); b8 = np.ones(len(T)); b9 = np.ones(len(T))
    tod = np.where(T.dec.values < 90, 0, np.where(T.dec.values < 240, 1, 2))
    for y in range(2014, int(ty.max()) + 1):
        tr = (ty < y) & (ty >= 2010); te = ty == y
        if not te.any() or tr.sum() < 200: continue
        b7[te] = np.where(T.brk.values[te] > np.nanmedian(T.brk.values[tr]), 2.0, 1.0)
        b8[te] = np.where(T.adx15.values[te] > np.nanquantile(T.adx15.values[tr], 2 / 3), 2.0, 1.0)
        vq = np.nanquantile(T.vol30.values[tr], [1 / 3, 2 / 3]); vc = np.digitize(np.nan_to_num(T.vol30.values), vq); mall = T.usd.values[tr].mean(); sz = np.ones(te.sum()); ii = np.flatnonzero(te)
        for n_, i in enumerate(ii):
            m = tr & (T.d.values == T.d.values[i]) & (tod == tod[i]) & (vc == vc[i])
            if m.sum() >= 30:
                mc = T.usd.values[m].mean(); sz[n_] = 0.0 if mc < 0 else 2.0 if mc > 2 * mall else 1.0
        b9[te] = sz
    ub, _ = Q.days(fair + M.band); ud, _ = Q.days(dep + M.band)
    DESC = {'Z1': 'model: 1 / 2 / 3 micros at 1 / 2 / 3 x the threshold', 'Z2': 'model: 2 micros on a strong 15-minute forecast', 'Z3': 'model: 2 micros when both are strong',
            'Z4': 'model: +1 micro for each of the 30 / 60-minute forecasts agreeing', 'Z5': 'model: size learned from cells of past signals', 'Z6': 'model: size from a trained model of past signals',
            'Z7': 'band: 2 micros on a strong breakout', 'Z8': 'band: 2 micros when ADX is high', 'Z9': 'band: size learned from cells of past trades'}
    say(); say("=" * 160)
    say(f"CONVICTION SIZING  e5z_sizing.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5Z_PLAN.txt) size on the most probable trades; book = model leg + band leg as one net position, live cost on every micro, "
        f"$ at today's price; judged at equal risk")
    say("=" * 160)
    say(f"   {'BASELINE 1 + 1 (fair model leg)':62s} | 2013-19 Sharpe {sh(ub[A_]):.2f} ${ub[A_].mean() * 252:+7,.0f}/yr DD ${dd(ub[A_]):6,.0f} $/DD {ub[A_].mean() * 252 / dd(ub[A_]):.2f} | 2019-26 Sharpe {sh(ub[B_]):.2f} ${ub[B_].mean() * 252:+7,.0f}/yr DD ${dd(ub[B_]):6,.0f} $/DD {ub[B_].mean() * 252 / dd(ub[B_]):.2f}")
    say(f"   {'BASELINE 1 + 1 (deployed model leg, for Z4)':62s} | 2013-19 Sharpe {sh(ud[A_]):.2f} ${ud[A_].mean() * 252:+7,.0f}/yr DD ${dd(ud[A_]):6,.0f} $/DD {ud[A_].mean() * 252 / dd(ud[A_]):.2f} | 2019-26 Sharpe {sh(ud[B_]):.2f} ${ud[B_].mean() * 252:+7,.0f}/yr DD ${dd(ud[B_]):6,.0f} $/DD {ud[B_].mean() * 252 / dd(ud[B_]):.2f}")
    R = {}
    for code in DESC:
        if code == 'Z4': mp = sized_path(M.g5, M.g15, z4, clk0); u, _ = Q.days(mp + M.band); base = ud; avg = np.abs(mp[keep]).sum() / max(np.abs(dep[keep]).sum(), 1)
        elif code in ('Z7', 'Z8', 'Z9'):
            sz = {'Z7': b7, 'Z8': b8, 'Z9': b9}[code]; bp = UU.path_with(T, sz); u, _ = Q.days(fair + bp); base = ub; avg = np.abs(bp[keep]).sum() / np.abs(M.band[keep]).sum()
        else: mp = sized_path(f5, f15, S[code], clk0); u, _ = Q.days(mp + M.band); base = ub; avg = np.abs(mp[keep]).sum() / max(np.abs(fair[keep]).sum(), 1)
        R[code] = (u, base); sel = sh(u[A_]) >= sh(base[A_]) + 0.05
        say(f"   {code} {DESC[code]:59s} | 2013-19 Sharpe {sh(u[A_]):.2f} ${u[A_].mean() * 252:+7,.0f}/yr DD ${dd(u[A_]):6,.0f} $/DD {u[A_].mean() * 252 / dd(u[A_]):.2f} | 2019-26 Sharpe {sh(u[B_]):.2f} ${u[B_].mean() * 252:+7,.0f}/yr DD ${dd(u[B_]):6,.0f} "
            f"$/DD {u[B_].mean() * 252 / dd(u[B_]):.2f} | average size of the leg {avg:.2f}x" + ("  SELECTED" if sel else ""))
    selected = [c for c in DESC if sh(R[c][0][A_]) >= sh(R[c][1][A_]) + 0.05]; bar = 2.0 if len(selected) == 1 else 2.4
    say(); say(f"CONFIRMATION 2019-26 at equal risk for {len(selected)} selected (bar: paired t >= {bar}):" if selected else "CONFIRMATION: no sizing rule selected on 2013-19.")
    conf = []
    for c in selected:
        u, base = R[c]; k = base[B_].std() / u[B_].std(); dlt = u[B_] * k - base[B_]; ok = sh(u[B_]) > sh(base[B_]) and tt(dlt) >= bar
        say(f"   {c} {DESC[c]}: Sharpe {sh(u[B_]):.2f} vs {sh(base[B_]):.2f}; at the baseline's risk ${dlt.mean() * 252:+,.0f}/yr, paired t {tt(dlt):+.2f} -> {'CONFIRMED' if ok else 'not confirmed'}")
        if ok:
            ka = base[keep].std() / u[keep].std(); d = pd.Series((u * ka - base)[keep], index=M.dates[keep]); yy = d.groupby(d.index.year).sum(); dB = d[d.index > L.JOIN]; yB = dB.groupby(dB.index.year).sum()
            k2 = (yy > 0).sum() >= 8 and dB[dB.index.year != yB.idxmax()].sum() > 0; k3 = dB.drop(dB.nlargest(5).index).sum() >= 0
            say(f"      years positive {(yy > 0).sum()} of {len(yy)}; without the best year ${dB[dB.index.year != yB.idxmax()].sum():+,.0f} -> {'PASS' if k2 else 'FAIL'}; without the 5 best days ${dB.drop(dB.nlargest(5).index).sum():+,.0f} -> {'PASS' if k3 else 'FAIL'}")
            if k2 and k3: conf.append(c)
    net = fair + M.band; a_ = np.abs(net[keep][:, clk0]);
    say(); say(f"   what the book already does by itself (decision bars 2013-26): flat {np.mean(a_ == 0) * 100:.1f}%, 1 micro {np.mean(a_ == 1) * 100:.1f}%, 2 micros (both legs agree) {np.mean(a_ == 2) * 100:.1f}%")
    say(f"VERDICT (conviction sizing): {'ADOPT ' + ', '.join(conf) if conf else 'no sizing rule confirmed'}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
