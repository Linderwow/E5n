r"""E5T family SM - smarter day model leg (E5T_PLAN.txt): meta-label filter, regime models, phase models, classifier target.
    python e5t_smart.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import e5b_lab as L
from e5b_lab import G, say
X = L.X; TR = L.TR; F = X.F; ND = L.ND; T0 = L.T0; U = L.U
OUT = os.path.join(HERE, 'E5T_RESULTS.txt'); CACHE = os.path.join(X.CACHE, 'e5t_sm_')
PHASES = ((15, 85), (90, 265), (270, 385))


def clf():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)


def cached(name, fn):
    p = CACHE + name + '.npy'
    if os.path.exists(p): return np.load(p)
    v = fn(); np.save(p, v); return v


def walk_groups(T, y, groups, MR):
    Xm = T[F.FEATS].values; yr = T.date.dt.year.values; pred = np.full(len(T), np.nan)
    for yy in sorted(set(yr)):
        if yy < 2013: continue
        te = yr == yy
        for g in np.unique(groups[np.isfinite(groups)]):
            tr = np.where((yr < yy) & np.isfinite(y) & (groups == g))[0]; tt = te & (groups == g)
            if len(tr) < 5000 or not tt.any(): continue
            m = MR.model(); m.fit(Xm[tr], y[tr]); pred[tt] = m.predict(Xm[tt])
    return pred


def regime_groups(T):
    dv = pd.Series(T.dvol20.values[::ND]); med = dv.expanding(min_periods=60).median().shift(1)
    g = np.where(dv.isna() | med.isna(), np.nan, (dv > med).astype(float)).repeat(ND)
    return g


def phase_groups(T):
    d = T.dec.values; g = np.full(len(T), np.nan)
    for i, (a, b) in enumerate(PHASES): g[(d >= a) & (d <= b)] = i
    return g


def classifier_score(T, side):
    Xm = T[F.FEATS].values; yr = T.date.dt.year.values; y = T.y.values; lab = np.where(y > 2 * side, 2, np.where(y < -2 * side, 0, 1)); lab = np.where(np.isfinite(y), lab, -1)
    out = np.full(len(T), np.nan)
    for yy in sorted(set(yr)):
        if yy < 2013: continue
        tr = np.where((yr < yy) & (lab >= 0))[0]; te = yr == yy
        if len(tr) < 5000 or not te.any(): continue
        m = clf(); m.fit(Xm[tr], lab[tr]); p = m.predict_proba(Xm[te]); cls = list(m.classes_)
        out[te] = p[:, cls.index(2)] - p[:, cls.index(0)]
    return out


def meta_mask(T, f5, f15, f30, f60, side):
    Xm = np.c_[T[F.FEATS].values, f5, f15, f30, f60]; yr = T.date.dt.year.values; y = T.y.values
    with np.errstate(invalid='ignore'):
        t = np.where(f5 > T0, 1.0, np.where(f5 < -T0, -1.0, 0.0)); t = np.where(f15 * t > 0, t, 0.0)
    sig = (t != 0) & np.isfinite(y); win = (t * y - 2 * side > 0).astype(int); keep = np.ones(len(T))
    for yy in sorted(set(yr)):
        tr = np.where(sig & (yr < yy))[0]; te = np.where(sig & (yr == yy))[0]
        if len(tr) < 3000 or not len(te): continue
        m = clf(); m.fit(Xm[tr], win[tr]); keep[te] = (m.predict_proba(Xm[te])[:, 1] > 0.5).astype(float)
    first = min((yy for yy in sorted(set(yr)) if (sig & (yr < yy)).sum() >= 3000), default=2015)
    return keep, first


def variants(book_T, f5, f15, f30, f60, tag, g):
    L2, ML, MR, IL = TR.research(); side = X.SIDE(2.35); m = lambda a: np.asarray(a, float).reshape(g.n, ND)
    keep, first = meta_mask(book_T, f5, f15, f30, f60, side)
    out = {'SM1': (L.model_path(mask=m(keep), g=g), f'meta-label filter (classifier on past signals, keep if P(win) > 0.5; usable from {first})', first)}
    pr = cached(f'{tag}_regime', lambda: walk_groups(book_T, book_T.y.values, regime_groups(book_T), MR))
    out['SM2'] = (L.model_path(f=m(pr), g=g), 'separate models for high / low 20-day volatility', 2013)
    pp = cached(f'{tag}_phase', lambda: walk_groups(book_T, book_T.y.values, phase_groups(book_T), MR))
    out['SM3'] = (L.model_path(f=m(pp), g=g), 'separate models for 09:45-10:55, 11:00-13:55, 14:00-15:55', 2013)
    sc = cached(f'{tag}_clf', lambda: classifier_score(book_T, side))
    out['SM4'] = (L.model_path(f=m(sc), thr=0.10, g=g), 'classifier target (up / down beyond 2 sides\' cost), trade when P(up) - P(down) beyond 0.10', 2013)
    return out


def main():
    t0 = time.time(); L.setup()
    say(); say("=" * 160); say(f"E5T FAMILY SM  e5t_smart.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  smarter day model leg; book = variant + band leg, net orders; $ per micro at today's price, live cost"); say("=" * 160)
    V = variants(X.NQ.T, X.F_NQ, G.f15.reshape(-1), G.f30.reshape(-1), G.f60.reshape(-1), 'nq', G)
    base = L.sim(G.model, G.band); b35 = L.sim(G.model, G.band, ticks=3.5); sel = []; res = {}
    say(L.line('B0 base book', base))
    for nm, (mp, desc, first) in V.items():
        s = L.sim(mp, G.band); cut = pd.Timestamp(f'{first}-01-01'); s_, b_ = s[s.index >= cut], base[base.index >= cut]
        k, kb = L.stats(s_), L.stats(b_); q = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']; res[nm] = (s, mp, cut)
        lg = L.stats(L.sim(mp, G.band, 1, 0)[lambda z: z.index >= cut]); lb = L.stats(L.sim(G.model, G.band, 1, 0)[lambda z: z.index >= cut])
        say(f"   {nm} {desc}")
        say(f"      from {first}: 2013/15-19 Sharpe {k['shA']:.2f} vs {kb['shA']:.2f}, max DD ${k['ddA']:,.0f} vs {kb['ddA']:,.0f} -> {'QUALIFIES' if q else 'does not qualify'} | "
            f"model leg alone Sharpe {lg['shA']:.2f} vs {lb['shA']:.2f} (first half)")
        if q: sel.append(nm)
    say("SELECTED ON THE FIRST HALF: " + (', '.join(sel) if sel else 'none')); bar = 0.05 / max(len(sel), 1)
    ES = None
    for nm in sel:
        s, mp, cut = res[nm]; s_, b_ = s[s.index >= cut], base[base.index >= cut]; k, kb = L.stats(s_), L.stats(b_)
        p = X.boot_p(s_, b_); s35 = L.sim(mp, G.band, ticks=3.5)[lambda z: z.index >= cut]; k35, k350 = L.stats(s35), L.stats(b35[b35.index >= cut])
        dif = (s_ - b_).sort_values(ascending=False); kp = ~s_.index.isin(dif.index[:20])
        if ES is None:
            ES = L.grid(X.ES, X.F_ES, np.load(os.path.join(X.CACHE, 'x5_es.npy')))
            esV = variants(X.ES.T, X.F_ES, ES.f15.reshape(-1), np.load(os.path.join(X.CACHE, 'c5_es.npy')), np.load(os.path.join(X.CACHE, 'c4_es.npy')), 'es', ES)
            ES.base = L.sim(L.model_path(g=ES), 0, 1, 0, g=ES)
        e = L.sim(esV[nm][0], 0, 1, 0, g=ES)[lambda z: z.index >= cut]; eb = ES.base[ES.base.index >= cut]; ea, ebb = L.halves(e); ba, bb = L.halves(eb)
        c = {'C1': k['shB'] > kb['shB'], 'C2': k['ddB'] <= kb['ddB'], 'C3': p < bar, 'C4': k35['shA'] > k350['shA'] and k35['shB'] > k350['shB'],
             'C5': X.sharpe(s_[kp]) > X.sharpe(b_[kp]), 'C6': X.sharpe(ea) > X.sharpe(ba) and X.sharpe(ebb) > X.sharpe(bb)}
        fails = [n_ for n_, v in c.items() if not v]
        say(f"   {nm} 2019-26: Sharpe {k['shB']:.2f} vs {kb['shB']:.2f}, max DD ${k['ddB']:,.0f} vs {kb['ddB']:,.0f}, $/yr {k['yrB']:,.0f} vs {kb['yrB']:,.0f} | P {p:.4f} (bar {bar:.4f}) | "
            f"3.5 ticks {k35['shA']:.2f}/{k35['shB']:.2f} vs {k350['shA']:.2f}/{k350['shB']:.2f} | without its 20 best days {X.sharpe(s_[kp]):.2f} vs {X.sharpe(b_[kp]):.2f} | "
            f"S&P model leg {X.sharpe(ea):.2f}/{X.sharpe(ebb):.2f} vs {X.sharpe(ba):.2f}/{X.sharpe(bb):.2f}")
        say(f"   VERDICT {nm}: " + ('CONFIRMED' if not fails else ('NEAR MISS (only the bootstrap bar)' if fails == ['C3'] else 'not confirmed') + ' - fails ' + ', '.join(fails)))
    say("information only, 2019-26 for every variant: " + " | ".join(f"{nm} Sharpe {L.stats(res[nm][0])['shB']:.2f}, max DD ${L.stats(res[nm][0])['ddB']:,.0f}" for nm in res)
        + f" | B0 {L.stats(base)['shB']:.2f}, ${L.stats(base)['ddB']:,.0f}")
    say(f"run {time.time() - t0:.0f} s")
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(L.LINES) + "\n")


if __name__ == '__main__':
    main()
