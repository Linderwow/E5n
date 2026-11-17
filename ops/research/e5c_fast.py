r"""E5B amendment A: faster decisions (F), earlier start (Y), to-the-close forecast as a band filter (H).
    python e5c_fast.py build     every-minute feature tables and walk-forward forecasts -> cache\e5c_*.npy (about 10 minutes)
    python e5c_fast.py run       audit, stage 1 on 2013-19, stage 2 on 2019-26 -> E5B_RESULTS.txt
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import G, say
X = L.X; TR = L.TR; T0 = L.T0; U = L.U; F = X.F
NM = 381; C = os.path.join(X.CACHE, 'e5c_')


def build():
    t0 = time.time(); L2, ML, MR, IL = TR.research(); F.STEP = 1; F.FIRST_DEC = 5; parts = []
    for nm, sel in (('nq', lambda d: d <= X.JOIN), ('mnqx', lambda d: d > X.JOIN)):
        R = TR.Raw(L2, nm); T = F.features(R); k = T.k.values; p_ = T.dec.values
        T['y15'] = np.log(R.C[k, np.minimum(p_ + 14, 389)] / R.O[k, p_]); T['yc'] = np.log(R.C[k, 389] / R.O[k, p_])
        T = T[sel(T.date)].reset_index(drop=True); parts.append(T); print(f"{nm}: {len(T):,} every-minute rows, {time.time() - t0:.0f} s", flush=True)
    T = pd.concat(parts, ignore_index=True); del parts
    n = len(T) // NM; assert len(T) == n * NM and (T.dec.values[:NM] == np.arange(5, 386)).all()
    Xf = T[F.FEATS].values; yr = T.date.dt.year.values; dec = T.dec.values
    grid = (dec % 5 == 0) & (dec >= 15); early = (dec % 5 == 0); every = dec >= 15
    np.save(C + 'px.npy', T.entry.values.reshape(n, NM)); np.save(C + 'dates.npy', T.date.values[::NM].astype('datetime64[ns]'))
    np.save(C + 'exit.npy', T.exitp.values.reshape(n, NM)[:, -1]); np.save(C + 'o930.npy', T.open930.values[::NM])
    targets = {'y': T.y.values, 'y15': T.y15.values, 'yc': T.yc.values}
    jobs = [('g5', 'y', grid), ('g15', 'y15', grid), ('e5', 'y', early), ('e15', 'y15', early), ('gc', 'yc', grid), ('a5', 'y', every), ('a15', 'y15', every)]
    for name, tgt, rows in jobs:
        y = targets[tgt]; pred = np.full(len(T), np.nan, np.float64)
        for yy in range(2013, 2027):
            tr = np.where(rows & (yr < yy) & np.isfinite(y))[0]; te = yr == yy
            m = MR.model(); m.fit(Xf[tr], y[tr]); pred[te] = m.predict(Xf[te])
        np.save(C + name + '.npy', pred.reshape(n, NM)); print(f"{name}: done, {time.time() - t0:.0f} s", flush=True)


def target(f5, f15, thr=T0):
    with np.errstate(invalid='ignore'):
        t = np.where(f5 > thr, 1.0, np.where(f5 < -thr, -1.0, 0.0)); return np.where(f15 * t > 0, t, 0.0)


def path(f5, f15, rule, decide, thr=T0):
    n = f5.shape[0]; pos = np.zeros(n); ttl = np.zeros(n); out = np.zeros((n, NM))
    for j in range(NM):
        if not decide[j] and rule == 'set':
            out[:, j] = pos; continue
        ok = ~np.isnan(f5[:, j]) & decide[j]; t = target(f5[:, j], f15[:, j], thr)
        if rule == 'set': pos = np.where(ok, t, pos)
        elif rule == 'hold5':
            sig = ok & (t != 0); ttl = np.where(sig, 5.0, ttl - 1); pos = np.where(sig, t, np.where(ttl <= 0, 0.0, pos))
        else:
            ttl = ttl - 1; free = ttl <= 0; new = np.where(ok, t, 0.0)
            pos = np.where(free, new, pos); ttl = np.where(free & (new != 0), 5.0, ttl)
        out[:, j] = pos
    return out


class M:
    pass


def load():
    L.setup(); ld = lambda nm: np.load(C + nm + '.npy')
    M.px = ld('px'); M.exit = ld('exit'); M.o930 = ld('o930'); M.dates = pd.DatetimeIndex(ld('dates')); M.n = len(M.dates)
    assert M.n == G.n and (M.dates == G.dates).all()
    M.dpx = np.diff(np.c_[M.px, M.exit], axis=1); M.dec = np.arange(5, 386); M.keep = G.keep
    for nm in ('g5', 'g15', 'e5', 'e15', 'gc', 'a5', 'a15'): setattr(M, nm, ld(nm))
    gi = (M.dec - 15) // 5; M.band = np.where((M.dec >= 15)[None, :], G.band[:, np.clip(gi, 0, 74)], 0.0)
    M.grid = (M.dec % 5 == 0) & (M.dec >= 15); M.every = M.dec >= 15


def simm(net, ticks=2.35):
    pnl = (net * M.dpx).sum(1); sides = np.abs(np.diff(np.c_[np.zeros(M.n), net, np.zeros(M.n)], axis=1)).sum(1)
    r = (pnl / M.o930 - sides * X.SIDE(ticks)) * U
    return pd.Series(r[M.keep], index=M.dates[M.keep]), float(sides[M.keep].sum() / 2)


def run():
    t0 = time.time(); load()
    say(); say("=" * 160); say(f"E5B AMENDMENT A  e5c_fast.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  faster decisions, earlier start, to-the-close filter; $ per micro at today's price, live cost on net orders"); say("=" * 160)
    base5 = L.sim(G.model, G.band); mgrid = path(M.g5, M.g15, 'set', M.grid); bm, _ = simm(mgrid + M.band)
    d1 = float(np.abs(bm - base5).max()); wp = pd.read_parquet(os.path.join(X.OPS, 'parity', 'walk_pred.parquet')).pred.values.reshape(G.n, L.ND)
    gsel = M.g5[:, M.grid]; okf = np.isfinite(wp) & np.isfinite(gsel); d2 = float(np.abs(gsel[okf] - wp[okf]).max())
    say(f"audit: every-minute simulator on the 5-minute decisions vs the 5-minute book: max |daily diff| ${d1:.2e} | grid-trained forecasts vs the deployed ones: max |diff| {d2:.1e} "
        f"-> {'PASS' if d1 < 1e-6 and d2 < 1e-12 else 'FAIL'}")
    if not (d1 < 1e-6 and d2 < 1e-12): say("audit FAILED - nothing graded"); L.flush(); return
    base = bm; kb = L.stats(base); say(L.line('B0 base book', base)); V = {}

    def reg(name, desc, net, fam):
        s, trips = simm(net); V[name] = dict(desc=desc, net=net, s=s, k=L.stats(s), fam=fam, trips=trips)

    for nm, rule, lab in (('F1', 'set', 're-decide every minute'), ('F2', 'hold5', 'latest signal holds 5 minutes'), ('F3', 'block5', '5-minute blocks from the first signal minute')):
        reg(nm, f'frozen models, {lab}', path(M.g5, M.g15, rule, M.every) + M.band, 'F')
    bestF = max(('F1', 'F2', 'F3'), key=lambda v: V[v]['k']['shA']); rule = {'F1': 'set', 'F2': 'hold5', 'F3': 'block5'}[bestF]
    reg('F4', f'models retrained on every-minute rows, rule of {bestF}', path(M.a5, M.a15, rule, M.every) + M.band, 'F')
    for nm, mult in (('F5', 1.5), ('F6', 2.0)):
        reg(nm, f'frozen models, latest signal holds 5 minutes, threshold {mult} x T', path(M.g5, M.g15, 'hold5', M.every, mult * T0) + M.band, 'F')
    ph = lambda f5, f15, offs: sum(path(f5, f15, 'set', (M.dec % 5 == o) & (M.dec >= 15)) for o in offs) / len(offs)
    reg('F7', 'frozen models, average of 5 copies on clocks offset by 0-4 minutes', ph(M.g5, M.g15, range(5)) + M.band, 'F')
    reg('F8', 'retrained models, average of 5 copies on clocks offset by 0-4 minutes', ph(M.a5, M.a15, range(5)) + M.band, 'F')
    reg('F9', 'frozen models, two half-size copies on clocks offset by 0 and 3 minutes', ph(M.g5, M.g15, (0, 3)) + M.band, 'F')
    for nm, first in (('Y1', 5), ('Y2', 10)):
        reg(nm, f'earlier start: decisions from {(570 + first) // 60}:{(570 + first) % 60:02d}', path(M.e5, M.e15, 'set', (M.dec % 5 == 0) & (M.dec >= first)) + M.band, 'Y')
    fc = M.gc[:, M.grid]; a = np.abs(fc[G.keep]); med = pd.Series(np.nanmedian(a, axis=1)).expanding(min_periods=20).median().shift(1).values
    medfull = np.full(G.n, np.nan); medfull[np.flatnonzero(G.keep)] = med

    def gate_close(strong):
        def gate(date, Mm, new, regular):
            i = G.row.get(date)
            if i is None or Mm < 585 or Mm > 955: return True
            v = fc[i, (Mm - 585) // 5]
            if v != v: return True
            if strong: return not (v * new < 0 and medfull[i] == medfull[i] and abs(v) > medfull[i])
            return v * new > 0
        return gate

    for nm, strong, lab in (('H1', False, 'band entry needs the to-the-close forecast on its side'), ('H2', True, 'band entry blocked only by a strong to-the-close forecast against it')):
        bp = L.band_path(gate=gate_close(strong)); s = L.sim(G.model, bp); V[nm] = dict(desc=lab, net=None, s=s, k=L.stats(s), fam='H', trips=0, bp=bp)
    say(); say("STAGE 1 - 2013-19 ONLY (selection; q = Sharpe above B0 and max drawdown not above B0's)")
    for nm, v in V.items():
        k = v['k']; v['q'] = k['shA'] > kb['shA'] and k['ddA'] <= kb['ddA']; a_, _ = L.halves(v['s'])
        say(f"   {nm:3s} {v['desc']:66s} Sharpe {k['shA']:5.2f} ({k['shA'] - kb['shA']:+.2f}) | ${k['yrA']:7,.0f}/yr ({k['yrA'] - kb['yrA']:+6,.0f}) | max DD ${k['ddA']:6,.0f} ({k['ddA'] - kb['ddA']:+6,.0f}) | "
            f"worst day ${a_.min():7,.0f} {'q' if v['q'] else ' '}" + (f" | {v['trips'] / (len(v['s']) / 252):,.0f} net round trips/yr" if v['trips'] else ''))
    sel = []
    for fam in ('F', 'Y', 'H'):
        qs = [nm for nm, v in V.items() if v['fam'] == fam and v['q']]
        if qs: sel.append(max(qs, key=lambda nm: V[nm]['k']['shA']))
    say(); say("SELECTED ON 2013-19: " + (", ".join(f"{nm} ({V[nm]['desc']})" for nm in sel) if sel else "none")); bar = 0.05 / max(len(sel), 1)
    say(); say(f"STAGE 2 - CONFIRMATION ON 2019-26 (bar for the bootstrap P: {bar:.4f})"); conf = []
    for nm in sel:
        v = V[nm]; s = v['s']; k = v['k']; p = X.boot_p(s, base)
        if v['net'] is not None: s35, _ = simm(v['net'], 3.5); b35, _ = simm(mgrid + M.band, 3.5)
        else: s35 = L.sim(G.model, v['bp'], ticks=3.5); b35 = L.sim(G.model, G.band, ticks=3.5)
        k35 = L.stats(s35); k350 = L.stats(b35); dif = (s - base).sort_values(ascending=False); kp = ~s.index.isin(dif.index[:20])
        c = {'C1': k['shB'] > kb['shB'], 'C2': k['ddB'] <= kb['ddB'], 'C3': p < bar, 'C4': k35['shA'] > k350['shA'] and k35['shB'] > k350['shB'], 'C5': X.sharpe(s[kp]) > X.sharpe(base[kp])}
        fails = [n_ for n_, okk in c.items() if not okk]
        say(f"   {nm} {v['desc']}")
        say(f"      2019-26: Sharpe {k['shB']:.2f} vs {kb['shB']:.2f} | ${k['yrB']:,.0f}/yr vs {kb['yrB']:,.0f} | max DD ${k['ddB']:,.0f} vs {kb['ddB']:,.0f} | all years Sharpe {k['sh']:.2f} vs {kb['sh']:.2f}, "
            f"max DD ${k['dd']:,.0f} vs {kb['dd']:,.0f} | P(Sharpe <= B0) {p:.4f} | 3.5 ticks {k35['shA']:.2f}/{k35['shB']:.2f} vs {k350['shA']:.2f}/{k350['shB']:.2f} | "
            f"without its 20 best days {X.sharpe(s[kp]):.2f} vs {X.sharpe(base[kp]):.2f}")
        say(f"      VERDICT {nm}: " + ('PASSES C1-C5 (S&P check C6 still to run)' if not fails else ('NEAR MISS (only the bootstrap bar)' if fails == ['C3'] else 'not confirmed') + ' - fails ' + ', '.join(fails)))
        if not fails: conf.append(nm)
    say(); say("PASSING C1-C5: " + (", ".join(conf) if conf else "none"))
    say(); say("INFORMATION ONLY - every variant on 2019-26 and all years (not used for selection):")
    for nm, v in V.items():
        k = v['k']
        say(f"   {nm:3s} {v['desc']:66s} 2019-26 Sharpe {k['shB']:5.2f} ({k['shB'] - kb['shB']:+.2f}) ${k['yrB']:7,.0f}/yr max DD ${k['ddB']:6,.0f} ({k['ddB'] - kb['ddB']:+6,.0f}) | "
            f"all Sharpe {k['sh']:5.2f} ${k['yr']:7,.0f}/yr max DD ${k['dd']:6,.0f}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build': build()
    elif a == 'run': run()
    else: print(__doc__)
