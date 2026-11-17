r"""E5B amendment G: indicators rebuilt on bars ending at every decision minute, models retrained on every-minute rows.
    python e5j_roll.py build | run
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
from e5b_lab import G, say
X = L.X; TR = L.TR; F = X.F; NM = C.NM; P = os.path.join(X.CACHE, 'e5j_')


def phase_bars(S, tf, off):
    n = S.C.shape[0]; nb = (390 - off) // tf
    r = lambda a: a[:, off:off + nb * tf].reshape(n, nb, tf)
    return nb, r(S.O)[:, :, 0].ravel(), r(S.H).max(2).ravel(), r(S.L).min(2).ravel(), r(S.C)[:, :, -1].ravel(), r(S.V).sum(2).ravel()


def build():
    t0 = time.time(); L2, ML, MR, IL = TR.research(); F.STEP = 1; F.FIRST_DEC = 5; parts = []; worst = 0.0
    for nm, sel in (('nq', lambda d: d <= X.JOIN), ('mnqx', lambda d: d > X.JOIN)):
        R = TR.Raw(L2, nm); T = F.features(R); k = T.k.values; p_ = T.dec.values
        T['y15'] = np.log(R.C[k, np.minimum(p_ + 14, 389)] / R.O[k, p_])
        for tf in F.TFS:
            cols = {i: np.full(len(T), np.nan) for i in F.IND}
            for off in range(tf):
                nb, o, h, l, c, v = phase_bars(R, tf, off); ind = F.indicators(o, h, l, c, v)
                rows = np.flatnonzero(p_ % tf == off); j = k[rows] * nb + (p_[rows] - off) // tf - 1; ok = j >= 0; j = np.clip(j, 0, None)
                for i in F.IND: cols[i][rows] = np.where(ok, ind[i][j], np.nan)
            for i in F.IND:
                on = p_ % tf == 0; a, b = cols[i][on], T[f'{i}{tf}'].values[on]; both = np.isfinite(a) & np.isfinite(b)
                if both.any(): worst = max(worst, float(np.abs(a[both] - b[both]).max()))
                T[f'{i}{tf}'] = cols[i]
        T = T[sel(T.date)].reset_index(drop=True); parts.append(T); print(f"{nm}: {len(T):,} rows with fresh indicators, {time.time() - t0:.0f} s", flush=True)
    print(f"audit: on-clock rows equal the deployed features, max |diff| {worst:.1e}", flush=True)
    T = pd.concat(parts, ignore_index=True); del parts
    n = len(T) // NM; Xf = T[F.FEATS].values; yr = T.date.dt.year.values; dec = T.dec.values; rows = dec >= 15
    for name, y in (('r5', T.y.values), ('r15', T.y15.values)):
        pred = np.full(len(T), np.nan)
        for yy in range(2013, 2027):
            tr = np.where(rows & (yr < yy) & np.isfinite(y))[0]; te = yr == yy
            m = MR.model(); m.fit(Xf[tr], y[tr]); pred[te] = m.predict(Xf[te])
        np.save(P + name + '.npy', pred.reshape(n, NM)); print(f"{name}: done, {time.time() - t0:.0f} s", flush=True)
    print('done', flush=True)


def run():
    t0 = time.time(); C.load(); M = C.M; r5 = np.load(P + 'r5.npy'); r15 = np.load(P + 'r15.npy')
    say(); say("=" * 160); say(f"E5B AMENDMENT G  e5j_roll.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  fresh indicators at every minute; $ per micro at today's price, live cost on net orders"); say("=" * 160)
    clk = lambda o: (M.dec % 5 == o) & (M.dec >= 15)
    say("model leg alone, one copy per clock offset - fresh indicators (new) vs stale indicators (same retraining, old features) vs deployed models:")
    for o in range(5):
        a, _ = C.simm(C.path(r5, r15, 'set', clk(o))); b, _ = C.simm(C.path(M.a5, M.a15, 'set', clk(o))); c, _ = C.simm(C.path(M.g5, M.g15, 'set', clk(o)))
        ka, kb, kc = L.stats(a), L.stats(b), L.stats(c)
        say(f"   offset {o}: fresh Sharpe {ka['shA']:.2f} / {ka['shB']:.2f} / {ka['sh']:.2f}, ${ka['yr']:6,.0f}/yr, max DD ${ka['dd']:6,.0f} | stale {kb['shA']:.2f} / {kb['shB']:.2f} / {kb['sh']:.2f}, ${kb['yr']:6,.0f}/yr, max DD ${kb['dd']:6,.0f} | "
            f"deployed {kc['shA']:.2f} / {kc['shB']:.2f} / {kc['sh']:.2f}, ${kc['yr']:6,.0f}/yr, max DD ${kc['dd']:6,.0f}")
    ph = lambda f5, f15: sum(C.path(f5, f15, 'set', clk(o)) for o in range(5)) / 5
    V = {'RF1': ('one copy on the deployed clock', C.path(r5, r15, 'set', clk(0)), C.path(M.a5, M.a15, 'set', clk(0))),
         'RF2': ('average of 5 copies on clocks offset by 0-4 minutes', ph(r5, r15), ph(M.a5, M.a15)),
         'RF3': ('every minute, the latest signal holds 5 minutes', C.path(r5, r15, 'hold5', M.every), C.path(M.a5, M.a15, 'hold5', M.every)),
         'RF4': ('every minute, 5-minute blocks from the first signal', C.path(r5, r15, 'block5', M.every), C.path(M.a5, M.a15, 'block5', M.every))}
    base, _ = C.simm(C.path(M.g5, M.g15, 'set', M.grid) + M.band); kb = L.stats(base); b35, _ = C.simm(C.path(M.g5, M.g15, 'set', M.grid) + M.band, 3.5); k350 = L.stats(b35)
    say(); say(L.line('B0 base book (deployed models and clock)', base))
    say("BOOK = model-leg variant + band leg. (a) against B0, 2013-19 selection; (b) against the fair baseline (same rule, stale indicators)")
    res = {}
    for nm, (lab, mp, fair) in V.items():
        s, tr = C.simm(mp + M.band); f, _ = C.simm(fair + M.band); k = L.stats(s); kf = L.stats(f); lg = L.stats(C.simm(mp)[0])
        q = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']; res[nm] = (s, k, mp, q)
        say(f"   {nm} {lab:52s} Sharpe {k['shA']:.2f} / {k['shB']:.2f} / {k['sh']:.2f} | ${k['yrA']:6,.0f} / {k['yrB']:6,.0f} / {k['yr']:6,.0f} a year | max DD ${k['ddA']:5,.0f} / {k['ddB']:5,.0f} / {k['dd']:5,.0f} | "
            f"{tr / (len(s) / 252):,.0f} round trips/yr | leg alone Sharpe {lg['sh']:.2f} | vs B0 on 2013-19: {'qualifies' if q else 'does not qualify'} | "
            f"fair baseline book Sharpe {kf['shA']:.2f} / {kf['shB']:.2f} / {kf['sh']:.2f}, max DD ${kf['dd']:5,.0f} -> {'better than fair in both halves' if k['shA'] > kf['shA'] and k['shB'] > kf['shB'] else 'not better than fair in both halves'}")
    sel = [nm for nm in res if res[nm][3]]
    if sel:
        nm = max(sel, key=lambda z: res[z][1]['shA']); s, k, mp, _ = res[nm]; p = X.boot_p(s, base); s35, _ = C.simm(mp + M.band, 3.5); k35 = L.stats(s35)
        dif = (s - base).sort_values(ascending=False); kp = ~s.index.isin(dif.index[:20])
        c = {'C1': k['shB'] > kb['shB'], 'C2': k['ddB'] <= kb['ddB'], 'C3': p < 0.05, 'C4': k35['shA'] > k350['shA'] and k35['shB'] > k350['shB'], 'C5': X.sharpe(s[kp]) > X.sharpe(base[kp])}
        fails = [n_ for n_, okk in c.items() if not okk]
        say(f"   SELECTED {nm}: 2019-26 Sharpe {k['shB']:.2f} vs {kb['shB']:.2f}, max DD ${k['ddB']:,.0f} vs {kb['ddB']:,.0f}, P(Sharpe <= B0) {p:.4f}, 3.5 ticks {k35['shA']:.2f}/{k35['shB']:.2f} vs {k350['shA']:.2f}/{k350['shB']:.2f}, "
            f"without its 20 best days {X.sharpe(s[kp]):.2f} vs {X.sharpe(base[kp]):.2f} -> " + ('PASSES C1-C5' if not fails else 'not confirmed - fails ' + ', '.join(fails)))
    else:
        say("   no RF variant qualifies against B0 on 2013-19")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build': build()
    elif a == 'run': run()
    else: print(__doc__)
