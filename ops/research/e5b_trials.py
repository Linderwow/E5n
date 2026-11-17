r"""E5B trials (E5B_PLAN.txt): every registered variant, selection on 2013-19, confirmation on 2019-26.
    python e5b_trials.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
from e5b_lab import G, say
X = L.X; ND = L.ND; T0 = L.T0; U = L.U
V = []


def jm(M):
    return (M - 585) // 5


def agree_fn(kind):
    def ok(i, j, new):
        v = [G.f15[i, j], G.f30[i, j], G.f60[i, j]]
        if any(x != x for x in v): return True
        s = [x * new > 0 for x in v]
        if kind == 'f15': return s[0]
        if kind == 'f30': return s[1]
        if kind == 'f60': return s[2]
        if kind == 'two': return sum(s) >= 2
        if kind == 'all': return all(s)
        return sum(v) * new > 0
    return ok


def gate_of(kind=None, last=None, no_oppose=False, extra_kind=None):
    ok = agree_fn(kind) if kind else None; okx = agree_fn(extra_kind) if extra_kind else None

    def gate(date, M, new, regular):
        i = G.row.get(date)
        if i is None or M < 585 or M > 955: return True
        j = jm(M)
        if last is not None and M > last: return False
        if ok is not None and not ok(i, j, new): return False
        if not regular and okx is not None and not okx(i, j, new): return False
        if no_oppose and G.model[i, j] == -new: return False
        return True
    return gate


def force_oppose(date, M, d):
    i = G.row.get(date)
    return i is not None and 585 <= M <= 955 and G.model[i, jm(M)] == -d


def day_risk(net, ticks, stop=None, lock=None):
    c1 = X.SIDE(ticks) * U; out = np.zeros_like(net); live = np.ones(G.n, bool); cum = np.zeros(G.n); peak = np.zeros(G.n); prev = np.zeros(G.n)
    for j in range(ND):
        if j > 0: cum += out[:, j - 1] * G.dpx[:, j - 1] / G.o930 * U
        peak = np.maximum(peak, cum)
        if stop is not None: live &= ~(cum <= -stop)
        if lock is not None: live &= ~((peak >= lock) & (cum <= 0.5 * peak))
        out[:, j] = np.where(live, net[:, j], 0.0); cum -= np.abs(out[:, j] - prev) * c1; prev = out[:, j]
    return out


def add(fam, name, desc, netfn, scale=None, sized=False, leg=None):
    V.append(dict(fam=fam, name=name, desc=desc, netfn=netfn, scale=scale, sized=sized, leg=leg))


def series(v, ticks=2.35):
    return L.sim(net=v['netfn'](ticks), ticks=ticks, scale=v['scale'])


def build():
    m, b = G.model, G.band; vr = G.m(X.NQ.volratio()); base = m + b
    for n in (1, 2, 3, 4):
        add('M', f'M{n}', f'model leg flat once a signal is older than {n * 5} minutes', (lambda mp: lambda t: mp + b)(L.model_path(maxrun=n)))
    add('M', 'M5', 'as M2, only in high 30-minute volatility', (lambda mp: lambda t: mp + b)(L.model_path(maxrun=2, runmask=(vr > 1).astype(float))))
    G.bandG = {}
    for k, (nm, lab) in enumerate((('f15', '15-minute forecast'), ('f30', '30-minute forecast'), ('f60', '60-minute forecast'), ('two', 'at least 2 of the 3 forecasts'),
                                   ('all', 'all 3 forecasts'), ('sum', 'sum of the 3 forecasts')), 1):
        bp = L.band_path(gate=gate_of(nm)); G.bandG[nm] = bp
        add('G', f'G{k}', f'band entry needs the {lab} on its side', (lambda bp: lambda t: m + bp)(bp))
    add('D', 'D1', 'legs oppose: the model leg decides (band suspended)', lambda t: np.where(m * b < 0, m, m + b))
    add('D', 'D2', 'band leg exits when the model leg holds the opposite side', (lambda bp: lambda t: m + bp)(L.band_path(force=force_oppose)))
    add('D', 'D3', 'band leg may not enter against the model leg', (lambda bp: lambda t: m + bp)(L.band_path(gate=gate_of(no_oppose=True))))
    add('D', 'D4', 'model leg may not oppose an open band position', lambda t: np.where(m * b < 0, b, m + b))
    add('S', 'S1', 'model leg x2 while both legs agree', lambda t: np.where(m * b > 0, 2 * m + b, m + b), sized=True)
    add('S', 'S2', 'band leg x2 while both legs agree', lambda t: np.where(m * b > 0, m + 2 * b, m + b), sized=True)
    for k, lim in enumerate((870, 840, 810), 1):
        add('T', f'T{k}', f'last band entry check {lim // 60}:{lim % 60:02d}', (lambda bp: lambda t: m + bp)(L.band_path(gate=gate_of(last=lim))))
    add('E', 'E1', 'extra band entries at any 5-minute mark with the 15-minute forecast', (lambda bp: lambda t: m + bp)(L.band_path(gate=gate_of(extra_kind='f15'), extra=lambda d, M: True)))
    for k, s in enumerate((300, 500, 800, 1100), 1): add('R', f'R{k}', f'book flat for the day at -${s:,}', (lambda s: lambda t: day_risk(base, t, stop=s))(s))
    for k, s in enumerate((400, 700, 1000), 5): add('R', f'R{k}', f'profit lock: ahead ${s:,}, flat after giving back half', (lambda s: lambda t: day_risk(base, t, lock=s))(s))
    for k, (cm, cb) in enumerate(((1, 2), (2, 1), (1, 1.5), (1.5, 1)), 1): add('W', f'W{k}', f'weights model {cm} : band {cb}', (lambda cm, cb: lambda t: cm * m + cb * b)(cm, cb), sized=True)
    dv = pd.Series(G.T.dvol20.values[::ND]); med = dv.expanding(min_periods=60).median().shift(1)
    for k, (lo, hi) in enumerate(((0.5, 2.0), (0.67, 1.5)), 1):
        sc = (med / dv).clip(lo, hi).fillna(1.0).values
        add('V', f'V{k}', f'book size = usual volatility / current, {lo}-{hi}', lambda t: base, scale=sc, sized=True)
    for k, (h, mult) in enumerate((('f15', 2), ('f15', 3), ('f15', 4), ('f30', 3), ('f30', 5), ('f60', 4), ('f60', 6)), 1):
        lp = L.model_path(f=getattr(G, h), thr=mult * T0, check=None)
        add('N', f'N{k}', f'new leg: {h[1:]}-minute forecast beyond {mult} x T', (lambda lp: lambda t: base + lp)(lp), sized=True, leg=lp)


def e2(kind):
    m = G.model
    add('E', 'E2', f'extra band entries at any 5-minute mark, every entry needs "{kind}"', (lambda bp: lambda t: m + bp)(L.band_path(gate=gate_of(kind, extra_kind=kind), extra=lambda d, M: True)))


def tstat(s):
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std() > 0 else 0.0


def main():
    t0 = time.time(); L.setup(); build()
    say(); say("=" * 160); say(f"E5B TRIALS  e5b_trials.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5B_PLAN.txt)  $ per micro at today's price, live cost on net orders"); say("=" * 160)
    base = L.sim(G.model, G.band); b35 = L.sim(G.model, G.band, ticks=3.5); kb = L.stats(base)
    say(L.line('B0 base book', base))
    say(); say("STAGE 1 - 2013-19 ONLY (selection). q = qualifies (Sharpe above B0 and max drawdown not above B0's; for size-changing families profit per $ of max drawdown not below B0's)")
    res = {}

    def stage1(v):
        s = series(v); k = L.stats(s); res[v['name']] = (s, k)
        a, _ = L.halves(s); marA = k['yrA'] / k['ddA'] if k['ddA'] > 0 else 0; marA0 = kb['yrA'] / kb['ddA']
        q = k['shA'] > kb['shA'] and ((marA >= marA0) if v['sized'] else (k['ddA'] <= kb['ddA']))
        ex = ''
        if v['leg'] is not None:
            lg = L.sim(net=v['leg']); la, lb = L.halves(lg); v['legt'] = tstat(la); v['legB'] = lb.mean() * 252
            ex = f" | leg alone 2013-19 ${la.mean() * 252:,.0f}/yr t {v['legt']:.2f}"; q = q and v['legt'] > 2.8
        v['q'] = q
        say(f"   {v['name']:3s} {v['desc']:62s} Sharpe {k['shA']:5.2f} ({k['shA'] - kb['shA']:+.2f}) | ${k['yrA']:7,.0f}/yr ({k['yrA'] - kb['yrA']:+6,.0f}) | max DD ${k['ddA']:6,.0f} ({k['ddA'] - kb['ddA']:+6,.0f}) | "
            f"worst day ${a.min():7,.0f} {'q' if q else ' '}{ex}")

    for v in V: stage1(v)
    gq = [v for v in V if v['fam'] == 'G' and v['q']]
    kind = {'G1': 'f15', 'G2': 'f30', 'G3': 'f60', 'G4': 'two', 'G5': 'all', 'G6': 'sum'}[max(gq, key=lambda v: res[v['name']][1]['shA'])['name']] if gq else 'two'
    e2(kind); stage1(V[-1])
    sel = []
    for fam in dict.fromkeys(v['fam'] for v in V):
        qs = [v for v in V if v['fam'] == fam and v['q']]
        if qs: sel.append(max(qs, key=lambda v: res[v['name']][1]['shA']))
    say(); say("SELECTED ON 2013-19: " + (", ".join(f"{v['name']} ({v['desc']})" for v in sel) if sel else "none"))
    say(f"families with no qualifier (dead without opening 2019-26): " + (", ".join(f for f in dict.fromkeys(v['fam'] for v in V) if not any(v['q'] for v in V if v['fam'] == f)) or 'none'))
    bar = 0.05 / max(len(sel), 1)
    say(); say(f"STAGE 2 - CONFIRMATION ON 2019-26 (bar for the bootstrap P: {bar:.4f})")
    ES = None; conf = []
    for v in sel:
        s, k = res[v['name']]; p = X.boot_p(s, base); s35 = series(v, 3.5); k35 = L.stats(s35); k350 = L.stats(b35)
        dif = (s - base).sort_values(ascending=False); kp = ~s.index.isin(dif.index[:20])
        marB = k['yrB'] / k['ddB'] if k['ddB'] > 0 else 0; marB0 = kb['yrB'] / kb['ddB']
        c = {'C1': k['shB'] > kb['shB'], 'C2': (marB >= marB0) if v['sized'] else (k['ddB'] <= kb['ddB']), 'C3': p < bar,
             'C4': k35['shA'] > k350['shA'] and k35['shB'] > k350['shB'], 'C5': X.sharpe(s[kp]) > X.sharpe(base[kp])}
        if v['leg'] is not None: c['leg B > 0'] = v['legB'] > 0
        es = ''
        if v['fam'] == 'M':
            if ES is None:
                ES = L.grid(X.ES, X.F_ES, np.load(os.path.join(X.CACHE, 'x5_es.npy'))); ES.base = L.sim(L.model_path(g=ES), 0, 1, 0, g=ES)
            n = int(v['name'][1]) if v['name'] != 'M5' else 2
            e = L.sim(L.model_path(maxrun=n, g=ES), 0, 1, 0, g=ES); ea, eb = L.halves(e); ba, bb = L.halves(ES.base)
            c['C6'] = X.sharpe(ea) > X.sharpe(ba) and X.sharpe(eb) > X.sharpe(bb)
            es = f" | S&P model leg: ES {X.sharpe(ea):.2f} vs {X.sharpe(ba):.2f}, MES {X.sharpe(eb):.2f} vs {X.sharpe(bb):.2f}"
        fails = [n_ for n_, okk in c.items() if not okk]
        say(f"   {v['name']} {v['desc']}")
        say(f"      2019-26: Sharpe {k['shB']:.2f} vs {kb['shB']:.2f} | ${k['yrB']:,.0f}/yr vs {kb['yrB']:,.0f} | max DD ${k['ddB']:,.0f} vs {kb['ddB']:,.0f} | all years Sharpe {k['sh']:.2f} vs {kb['sh']:.2f}, "
            f"max DD ${k['dd']:,.0f} vs {kb['dd']:,.0f} | P(Sharpe <= B0) {p:.4f} | 3.5 ticks {k35['shA']:.2f}/{k35['shB']:.2f} vs {k350['shA']:.2f}/{k350['shB']:.2f} | "
            f"without its 20 best days {X.sharpe(s[kp]):.2f} vs {X.sharpe(base[kp]):.2f}{es}")
        say(f"      VERDICT {v['name']}: " + ('CONFIRMED' if not fails else ('NEAR MISS (only the bootstrap bar)' if fails == ['C3'] else 'not confirmed') + ' - fails ' + ', '.join(fails)))
        if not fails: conf.append(v)
        v['fails'] = fails
    say(); say("CONFIRMED: " + (", ".join(v['name'] for v in conf) if conf else "none"))
    say(); say("INFORMATION ONLY - every variant on 2019-26 and all years (not used for selection):")
    for v in V:
        s, k = res[v['name']]
        say(f"   {v['name']:3s} {v['desc']:62s} 2019-26 Sharpe {k['shB']:5.2f} ({k['shB'] - kb['shB']:+.2f}) ${k['yrB']:7,.0f}/yr max DD ${k['ddB']:6,.0f} ({k['ddB'] - kb['ddB']:+6,.0f}) | "
            f"all Sharpe {k['sh']:5.2f} ${k['yr']:7,.0f}/yr max DD ${k['dd']:6,.0f}")
    G.res = res; G.sel = sel; G.conf = conf; G.base = base
    say(f"run {time.time() - t0:.0f} s"); L.flush()
    return res, sel, conf


if __name__ == '__main__':
    main()
