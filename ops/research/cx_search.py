r"""CX (CX_PLAN.txt): every indicator / volume state and pair as a filter or booster on both legs, with a measured luck bar.
    python cx_search.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import norm
import e5b_lab as L
import e5c_fast as C
import e5u_bandml as UB
from e5b_lab import G, say

X = L.X; U = L.U; ND = L.ND; TK = 4.5; NPERM = 40; MIN = {'model': 300, 'band': 60}


def states(df, feats, ref):
    cols, names, edges = [], [], {}
    for f in feats:
        v = df[f].values.astype(float); r = ref[f].values.astype(float); r = r[np.isfinite(r)]
        if len(np.unique(r)) <= 3: cuts = np.unique(r)[:-1] + 0.5 * np.diff(np.unique(r)) if len(np.unique(r)) > 1 else np.array([])
        else: cuts = np.unique(np.quantile(r, [1 / 3, 2 / 3]))
        edges[f] = cuts; b = np.searchsorted(cuts, v, side='right'); b[~np.isfinite(v)] = -1
        for k in range(len(cuts) + 1): cols.append(b == k); names.append((f, k))
    return np.array(cols).T.astype(np.float32), names, edges


def apply_states(df, names, edges):
    out = []
    for f, k in names:
        v = df[f].values.astype(float); b = np.searchsorted(edges[f], v, side='right'); b[~np.isfinite(v)] = -1; out.append(b == k)
    return np.array(out).T.astype(np.float32)


def cell_z(S, y, feat_of, min_n):
    n = S.T @ S; s1 = S.T @ (S * y[:, None]); s2 = S.T @ (S * (y ** 2)[:, None])
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s1 / n; var = s2 / n - mean ** 2; z = (mean - y.mean()) / np.sqrt(np.maximum(var, 1e-12) / n)
    same = feat_of[:, None] == feat_of[None, :]; diag = np.eye(len(feat_of), dtype=bool)
    ok = (n >= min_n) & (~same | diag); z = np.where(ok, z, np.nan); mean = np.where(ok, mean, np.nan)
    return z, mean, n


def search(S, y, yr, feat_of, min_n, rng):
    z, mean, n = cell_z(S, y, feat_of, min_n)
    mx = []
    for _ in range(NPERM):
        yp = y.copy()
        for g in np.unique(yr):
            m = np.flatnonzero(yr == g); yp[m] = y[rng.permutation(m)]
        zp, _, _ = cell_z(S, yp, feat_of, min_n); mx.append(np.nanmax(np.abs(np.triu(zp))))
    return z, mean, n, float(np.percentile(mx, 95)), mx


def main():
    t0 = time.time(); C.load(); M = C.M; L.OUT = os.path.join(HERE, 'CX_RESULTS.txt'); rng = np.random.default_rng(1117)
    isA = np.asarray(G.isA); keep = np.asarray(M.keep); FE = G.T[X.F.FEATS].reset_index(drop=True); c45 = X.SIDE(TK) * U; c23 = X.SIDE(2.35) * U
    tg = C.target(M.g5, M.g15); gcols = np.flatnonzero(M.grid); rows = []
    model = C.path(M.g5, M.g15, 'set', M.grid); held = (model != 0).sum(); trips = np.abs(np.diff(np.c_[np.zeros(M.n), model, np.zeros(M.n)], axis=1)).sum()
    avg_cost = trips * c45 / max(held / 5, 1)
    for gi, j in enumerate(gcols):
        t = tg[:, j]; mv = M.dpx[:, j:j + 5].sum(1) / M.o930 * U
        for r in np.flatnonzero((t != 0) & keep):
            rows.append((r, gi, t[r] * mv[r] - avg_cost))
    MU = pd.DataFrame(rows, columns=['r', 'gi', 'usd']); MU = pd.concat([MU, FE.iloc[MU.r.values * ND + MU.gi.values].reset_index(drop=True)], axis=1)
    BT = UB.trades(); BT = BT[keep[BT.r.values]].reset_index(drop=True); BT['usd'] = BT.usd + 2 * c23 - 2 * c45
    bfeats = [c for c in BT.columns if c not in ('r', 'j0', 'j1', 'd', 'usd')]
    say(); say("=" * 150); say(f"CX  INDICATOR / VOLUME COMBINATIONS AS FILTERS AND BOOSTERS  cx_search.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {TK} ticks; luck bar from {NPERM} shuffles"); say("=" * 150)
    surv = []
    for leg, D, feats in (('model', MU, list(X.F.FEATS)), ('band', BT, bfeats)):
        A = isA[D.r.values]; DA, DB = D[A].reset_index(drop=True), D[~A].reset_index(drop=True)
        SA, names, edges = states(DA, feats, DA); SB = apply_states(DB, names, edges)
        feat_of = np.array([feats.index(f) for f, _ in names]); yA = DA.usd.values.astype(float); yrA = pd.DatetimeIndex(M.dates[DA.r.values]).year.values
        z, mean, n, bar, mx = search(SA, yA, yrA, feat_of, MIN[leg], rng)
        iu = np.triu_indices(len(names)); zz = z[iu]; good = np.isfinite(zz)
        say(f"   {leg.upper()} LEG: {len(DA):,} discovery units, {len(names)} states, {int(good.sum()):,} cells (singles + pairs) with >= {MIN[leg]} units; "
            f"largest |z| by chance (40 shuffles): median {np.median(mx):.2f}, 95% = LUCK BAR {bar:.2f}; largest real |z| {np.nanmax(np.abs(zz)):.2f}")
        hits = [(iu[0][k], iu[1][k]) for k in np.flatnonzero(good & (np.abs(zz) >= bar))]
        cand = []
        for a, b in hits:
            za = z[a, b]; kind = 'BOOST' if za > 0 else 'FILTER'
            if kind == 'FILTER' and not mean[a, b] < 0: continue
            cand.append((a, b, kind, za, mean[a, b], n[a, b]))
        say(f"      cells beyond the luck bar: {len(cand)}")
        if cand:
            yB = DB.usd.values.astype(float); zbar = norm.ppf(1 - 0.05 / (2 * len(cand)))
            for a, b, kind, za, ma, na in sorted(cand, key=lambda c: -abs(c[3]))[:30]:
                m_ = (SB[:, a] * SB[:, b]) > 0; yb = yB[m_]
                zb = (yb.mean() - yB.mean()) / (yb.std() / np.sqrt(len(yb))) if len(yb) > 2 and yb.std() > 0 else np.nan
                ok = np.isfinite(zb) and np.sign(zb) == np.sign(za) and abs(zb) >= zbar
                lab = names[a] if a == b else (names[a], names[b])
                say(f"      {kind:6s} {str(lab):70s} 2013-19 z {za:+.2f} mean ${ma:+.1f} n {int(na)} | 2019-26 z {zb:+.2f} n {int(m_.sum())} {'CONFIRMED' if ok else ''}")
                if ok: surv.append((leg, kind, lab, a, b, names, edges))
    say(f"   confirmed by both periods: {len(surv)}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()
    return surv


if __name__ == '__main__':
    main()
