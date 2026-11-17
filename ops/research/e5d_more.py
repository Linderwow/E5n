r"""E5B amendment B: add to band-leg winners (P), S&P model leg (X), switching legs by their own recent record (Q).
    python e5d_more.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import G, say
X = L.X; T0 = L.T0; U = L.U


def tstat(s):
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std() > 0 else 0.0


def judge(fam, items, base_fn, lab_base):
    base = base_fn(2.35); b35 = base_fn(3.5); kb = L.stats(base); k350 = L.stats(b35)
    say(); say(f"FAMILY {fam} - base for this family: {lab_base}"); say(L.line('   base', base))
    say("   STAGE 1 - 2013-19 only (q = Sharpe above base and profit per $ of max drawdown not below base):")
    marA0 = kb['yrA'] / kb['ddA']
    for it in items:
        s = it['fn'](2.35); k = L.stats(s); it['s'] = s; it['k'] = k; a, _ = L.halves(s)
        q = k['shA'] > kb['shA'] and k['yrA'] / k['ddA'] >= marA0; ex = ''
        if it.get('leg') is not None:
            la, lb = L.halves(it['leg']); it['legB'] = lb.mean() * 252; q = q and tstat(la) > 2.8
            ex = f" | leg alone 2013-19 ${la.mean() * 252:,.0f}/yr t {tstat(la):.2f}"
        it['q'] = q
        say(f"      {it['name']:3s} {it['desc']:62s} Sharpe {k['shA']:5.2f} ({k['shA'] - kb['shA']:+.2f}) | ${k['yrA']:7,.0f}/yr ({k['yrA'] - kb['yrA']:+6,.0f}) | max DD ${k['ddA']:6,.0f} ({k['ddA'] - kb['ddA']:+6,.0f}) | "
            f"$ per $ of DD {k['yrA'] / k['ddA']:.2f} vs {marA0:.2f} | worst day ${a.min():7,.0f} {'q' if q else ' '}{ex}")
    qs = [it for it in items if it['q']]
    if not qs:
        say(f"   no qualifier on 2013-19 -> family {fam} dead without opening 2019-26")
    else:
        it = max(qs, key=lambda z: z['k']['shA']); s = it['s']; k = it['k']; p = X.boot_p(s, base); s35 = it['fn'](3.5); k35 = L.stats(s35)
        dif = (s - base).sort_values(ascending=False); kp = ~s.index.isin(dif.index[:20])
        c = {'C1': k['shB'] > kb['shB'], 'C2': k['yrB'] / k['ddB'] >= kb['yrB'] / kb['ddB'], 'C3': p < 0.05 / 3, 'C4': k35['shA'] > k350['shA'] and k35['shB'] > k350['shB'],
             'C5': X.sharpe(s[kp]) > X.sharpe(base[kp])}
        if it.get('leg') is not None: c['leg B > 0'] = it['legB'] > 0
        fails = [n for n, ok in c.items() if not ok]
        say(f"   SELECTED {it['name']} ({it['desc']}) - STAGE 2 on 2019-26 (bootstrap bar {0.05 / 3:.4f}):")
        say(f"      2019-26: Sharpe {k['shB']:.2f} vs {kb['shB']:.2f} | ${k['yrB']:,.0f}/yr vs {kb['yrB']:,.0f} | max DD ${k['ddB']:,.0f} vs {kb['ddB']:,.0f} | $ per $ of DD {k['yrB'] / k['ddB']:.2f} vs {kb['yrB'] / kb['ddB']:.2f} | "
            f"all years Sharpe {k['sh']:.2f} vs {kb['sh']:.2f}, max DD ${k['dd']:,.0f} vs {kb['dd']:,.0f} | P(Sharpe <= base) {p:.4f} | 3.5 ticks {k35['shA']:.2f}/{k35['shB']:.2f} vs {k350['shA']:.2f}/{k350['shB']:.2f} | "
            f"without its 20 best days {X.sharpe(s[kp]):.2f} vs {X.sharpe(base[kp]):.2f}")
        say(f"      VERDICT {it['name']}: " + ('CONFIRMED' if not fails else ('NEAR MISS (only the bootstrap bar)' if fails == ['C3'] else 'not confirmed') + ' - fails ' + ', '.join(fails)))
    say("   information only - 2019-26 and all years:")
    for it in items:
        k = it['k']
        say(f"      {it['name']:3s} 2019-26 Sharpe {k['shB']:5.2f} ({k['shB'] - kb['shB']:+.2f}) ${k['yrB']:7,.0f}/yr max DD ${k['ddB']:6,.0f} | all Sharpe {k['sh']:5.2f} ${k['yr']:7,.0f}/yr max DD ${k['dd']:6,.0f}")


def main():
    t0 = time.time(); L.setup(); m, b = G.model, G.band
    say(); say("=" * 160); say(f"E5B AMENDMENT B  e5d_more.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  $ per micro at today's price, live cost on net orders"); say("=" * 160)
    book = lambda t: L.sim(m, b, ticks=t)
    P = []
    for nm, pyr, lab in (('P1', ('profit', 1.0), 'add 1 micro to a band trade ahead by 1 entry width'), ('P2', ('profit', 2.0), 'add 1 micro to a band trade ahead by 2 entry widths'),
                         ('P3', ('profit', 0.5), 'add 1 micro to a band trade ahead by half an entry width'), ('P4', ('check',), 'add 1 micro at the next half hour if still beyond the band and ahead')):
        bp = L.band_path(pyramid=pyr); P.append(dict(name=nm, desc=lab, fn=(lambda bp: lambda t: L.sim(m, bp, ticks=t))(bp)))
    judge('P', P, book, 'the netted book B0')
    ES = L.grid(X.ES, X.F_ES, np.load(os.path.join(X.CACHE, 'x5_es.npy'))); w = 0.62; tick_es = 4.8; thr_es = 4 * X.SIDE(tick_es)
    Xs = []
    for nm, thr, lab in (('X1', thr_es, 'add 1 MES micro of the S&P model leg, threshold 4 x its own cost'), ('X2', T0, 'add 1 MES micro of the S&P model leg, threshold as on MNQ')):
        mp = L.model_path(thr=thr, g=ES)
        leg = lambda t, mp=mp: w * L.sim(mp, 0, 1, 0, ticks=tick_es + (t - 2.35) * 2, g=ES)
        Xs.append(dict(name=nm, desc=lab, fn=(lambda leg: lambda t: (book(t) + leg(t).reindex(book(t).index).fillna(0.0)))(leg), leg=leg(2.35)))
    judge('X', Xs, book, 'the netted book B0')
    mo = lambda t: L.sim(m, b, 1, 0, ticks=t); ba = lambda t: L.sim(m, b, 0, 1, ticks=t); sep = lambda t: mo(t) + ba(t)

    def switch(t, legs, win=60):
        a, c = mo(t), ba(t); on = lambda s: (s.rolling(win, min_periods=win).sum().shift(1) > 0) | s.rolling(win, min_periods=win).sum().shift(1).isna()
        return (a * on(a) if 'm' in legs else a) + (c * on(c) if 'b' in legs else c)

    def weights(t, win=120):
        a, c = mo(t), ba(t); sh = lambda s: (s.rolling(win, min_periods=win).mean() / s.rolling(win, min_periods=win).std()).shift(1).clip(lower=0)
        sa, sc = sh(a), sh(c); tot = sa + sc; wa = (2 * sa / tot).where(tot > 0, 1.0).fillna(1.0); wc = (2 * sc / tot).where(tot > 0, 1.0).fillna(1.0)
        return a * wa + c * wc

    Q = [dict(name='Q1', desc='model leg only while its last 60 sessions are positive', fn=lambda t: switch(t, 'm')),
         dict(name='Q2', desc='band leg only while its last 60 sessions are positive', fn=lambda t: switch(t, 'b')),
         dict(name='Q3', desc='each leg only while its own last 60 sessions are positive', fn=lambda t: switch(t, 'mb')),
         dict(name='Q4', desc='leg weights by trailing 120-session Sharpe', fn=lambda t: weights(t))]
    judge('Q', Q, sep, 'the two legs costed separately')
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
