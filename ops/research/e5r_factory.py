r"""E5R (E5R_PLAN.txt): account-side policies for E5n on Bulenox 50K as stored - funded sizing policies and several accounts with replacement.
    python e5r_factory.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5k_fair as K
from e5b_lab import G, say
NP_, ND_ = 10000, 504; EVAL_FEE, RESET, MONTH = 175.0, 78.0, 21


def rule(per, mx): return lambda cu: np.clip(np.floor(cu / per), 1, mx)


POL = {'A0': ('fixed 1 + 1', 'book', rule(1e12, 1)), 'A1': ('1 + 1 per $2,500 of cushion, max 4 (build 4)', 'book', rule(2500, 4)),
       'B1': ('band only until the threshold locks, then A1', 'lock', rule(2500, 4)), 'B2': ('band only until the first payout, then A1', 'pay', rule(2500, 4)),
       'C1': ('max 6 units', 'book', rule(2500, 6)), 'C2': ('max 8 units', 'book', rule(2500, 8)), 'C3': ('$2,000 a unit, max 4', 'book', rule(2000, 4)),
       'C4': ('$2,000 a unit, max 8', 'book', rule(2000, 8)), 'C5': ('$3,500 a unit, max 8', 'book', rule(3500, 8)),
       'D1': ('band only until the lock, then max 6', 'lock', rule(2500, 6)), 'D2': ('band only until the lock, then max 8', 'lock', rule(2500, 8)),
       'D3': ('band only until the first payout, then max 8', 'pay', rule(2500, 8)), 'D4': ('band only until the lock, then $2,000 a unit, max 8', 'lock', rule(2000, 8)),
       'E1': ('unit = model 1 + band 2 per $3,500, max 4', 'book12', rule(3500, 4))}


class Slot:
    def __init__(s, n, a):
        s.n = n; s.a = a; s.phase = np.full(n, -1); s.bal = np.zeros(n); s.hi = np.zeros(n); s.thr = np.full(n, -float(a['dd'])); s.locked = np.zeros(n, bool)
        s.npay = np.zeros(n, int); s.days = np.zeros(n, int); s.ptot = np.zeros(n); s.pbest = np.zeros(n); s.wait = np.zeros(n, int); s.ev = np.zeros(n, int)
        s.lost = np.zeros(n, int); s.passed = np.zeros(n, int); s.units = np.zeros(n)

    def start_eval(s, m):
        s.phase[m] = 0; s.bal[m] = 0.0; s.hi[m] = 0.0; s.ev[m] = 0

    def start_funded(s, m):
        s.phase[m] = 1; s.bal[m] = 0.0; s.hi[m] = 0.0; s.thr[m] = -float(s.a['dd']); s.locked[m] = False
        s.npay[m] = 0; s.days[m] = 0; s.ptot[m] = 0.0; s.pbest[m] = 0.0; s.wait[m] = 0


def step_funded(BL, s, S, t, mode, rl, replace):
    a = s.a; f = s.phase == 1; cash = np.zeros(s.n)
    if not f.any(): return cash
    band = (~s.locked) if mode == 'lock' else (s.npay == 0) if mode == 'pay' else np.zeros(s.n, bool)
    key = 'book12' if mode == 'book12' else 'book'
    X = np.where(band, S['band'][0][:, t], S[key][0][:, t]); W = np.where(band, S['band'][1][:, t], S[key][1][:, t]); NT = np.where(band, S['band'][2][:, t], S[key][2][:, t])
    q = np.where(band, 1.0, rl(s.bal - s.thr)); s.units = np.where(f, q, 0.0)
    x, w = BL.day(X * q, W * q, np.where(s.locked, np.inf, float(a['dll'])))
    x = np.where(f, x, 0.0); w = np.where(f, w, 0.0)
    br = f & (s.bal + w < s.thr); alive = f & ~br; s.bal = s.bal + x
    s.hi = np.where(alive, np.maximum(s.hi, s.bal), s.hi)
    up = alive & ~s.locked; s.thr = np.where(up, np.maximum(s.thr, s.hi - a['dd']), s.thr)
    lk = up & (s.thr >= 100); s.thr[lk] = 100.0; s.locked |= lk
    ww = alive & (s.wait > 0); s.wait[ww] -= 1; act = alive & ~ww
    s.days += (act & (NT > 0)).astype(int); s.ptot = np.where(act, s.ptot + x, s.ptot); s.pbest = np.where(act, np.maximum(s.pbest, x), s.pbest)
    amt = np.minimum(np.where(s.npay < 3, float(a['cap']), 0.5 * s.bal - 1), s.bal - a['reserve'])
    el = act & (s.days >= 10) & (s.ptot > 0) & (s.pbest <= 0.4 * s.ptot) & (amt >= 1000)
    s.bal[el] -= amt[el]; cash[el] += 0.9 * amt[el]; s.npay[el] += 1; s.days[el] = 0; s.ptot[el] = 0.0; s.pbest[el] = 0.0; s.wait[el] = 3
    s.lost += br.astype(int)
    if replace: s.phase[br] = 2
    else: s.phase[br] = 3
    return cash


def step_eval(BL, s, S, t):
    a = s.a; e = s.phase == 0; cash = np.zeros(s.n)
    if not e.any(): return cash
    fee = e & (s.ev % MONTH == 0); cash[fee] -= EVAL_FEE; s.ev += e.astype(int)
    x, w = BL.day(S['eval'][0][:, t], S['eval'][1][:, t], float(a['dll']))
    br = e & (s.bal + w < s.hi - a['dd']); ok = e & ~br
    s.bal = np.where(ok, s.bal + x, s.bal)
    cash[br] -= RESET; s.bal[br] = 0.0; s.hi[br] = 0.0
    s.hi = np.where(ok, np.maximum(s.hi, s.bal), s.hi)
    hit = ok & (s.bal >= a['target']); cash[hit] -= float(a['act']); s.passed += hit.astype(int); s.phase[hit] = 4
    return cash


def factory(BL, S, a, k, stagger, mode, rl, with_eval=True, replace=True):
    slots = [Slot(NP_, a) for _ in range(k)]; CF = np.zeros((NP_, ND_)); alive = np.zeros((NP_, ND_)); units = np.zeros((NP_, ND_))
    pay = np.zeros((NP_, ND_)); first_lost = np.full(NP_, ND_)
    for t in range(ND_):
        for i, s in enumerate(slots):
            if t == i * stagger:
                m = np.ones(NP_, bool)
                if with_eval: s.start_eval(m)
                else: s.start_funded(m)
            nf = s.phase == 4; s.start_funded(nf)
            ne = s.phase == 2; s.start_eval(ne)
            c1 = step_eval(BL, s, S, t) if with_eval else 0.0
            c2 = step_funded(BL, s, S, t, mode, rl, replace and with_eval)
            if not with_eval: first_lost = np.where((s.phase == 3) & (first_lost == ND_), t, first_lost)
            CF[:, t] += c1 + c2; pay[:, t] += c2; alive[:, t] += (s.phase == 1); units[:, t] += np.where(s.phase == 1, s.units, 0.0)
    return NS_(CF=CF, pay=pay, alive=alive, units=units, first_lost=first_lost, passed=sum(s.passed for s in slots), lost=sum(s.lost for s in slots))


class NS_:
    def __init__(s, **k): s.__dict__.update(k)


def streams(BR, legs, seed):
    S = {}
    for nm, (u, w) in legs.items():
        nt = (u != 0).astype(float); X, NT = BR.boot_paths(ND_, NP_, u, nt, seed=seed); W, _ = BR.boot_paths(ND_, NP_, w, nt, seed=seed); S[nm] = (X, W, NT)
    return S


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR, masterscale as MS
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0); dep = C.path(M.g5, M.g15, 'set', clk0)
    mu = K.day_arrays(fair)[0].mean()

    def legs(mp, none=False):
        out = {}
        for nm, cm, cb in (('book', 1, 1), ('book12', 1, 2), ('band', 0, 1), ('eval', 1, 6)):
            u, w = K.day_arrays(cm * mp + cb * M.band)
            if none and cm: u = u - cm * mu; w = np.minimum(np.minimum(w - cm * mu * (w < 0), u), 0.0)
            out[nm] = (u, w)
        return out
    LEG = {'FAIR': legs(fair), 'AS BACKTESTED': legs(dep), 'MODEL LEG NO EDGE': legs(fair, none=True)}
    say(); say("=" * 160)
    say(f"ACCOUNT SIDE  e5r_factory.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5R_PLAN.txt) Bulenox 50K as stored; bootstrap of the 2019-26 sessions, {NP_:,} paths x {ND_} sessions; $ at today's price, live cost")
    say("=" * 160)
    S = streams(BR, LEG['FAIR'], BL.SEED_REPORT)
    for code in ('A0', 'A1'):
        r = factory(BL, S, a, 1, 0, POL[code][1], POL[code][2], with_eval=False)
        ref_pay, ref_ou, _ = MS.master_dyn(S['book'][0], S['book'][1], S['book'][2], a, POL[code][2])
        assert np.allclose(r.pay, ref_pay) and np.array_equal(r.first_lost, ref_ou), f"simulator differs from the stored funded simulation for {code}"
    say("   check: the simulator reproduces the stored funded-account simulation exactly for fixed 1 + 1 and for cushion sizing (payouts and closing days on every path)")

    def one(Sx, code):
        r = factory(BL, Sx, a, 1, 0, POL[code][1], POL[code][2], with_eval=False); tot = r.pay.sum(1); on = r.units > 0
        return dict(pay=tot.mean(), med=np.median(tot), p25=np.percentile(tot, 25), p75=np.percentile(tot, 75), any=np.mean(tot > 0), open1=np.mean(r.first_lost > 251), open2=np.mean(r.first_lost > 503),
                    units=float(r.units[on].mean()), top=float(r.units.max()))
    say(); say("PART 1 - ONE FUNDED ACCOUNT, 2 years (pick seed -> reporting seed shown; BETTER = more payouts than A1 with no lower survival and no lower chance of a payout at FAIR, and not worse with no model edge)")
    say(f"   {'policy':58s} | FAIR: payouts avg / median | any payout | open 1 yr / 2 yrs | avg units | NO EDGE: payouts, open 1 yr | AS BACKTESTED: payouts")
    Sp = streams(BR, LEG['FAIR'], BL.SEED_PICK); Sn = streams(BR, LEG['MODEL LEG NO EDGE'], BL.SEED_REPORT); Sd = streams(BR, LEG['AS BACKTESTED'], BL.SEED_REPORT)
    RES = {}; better = []
    for code in POL:
        f = one(S, code); n = one(Sn, code); d = one(Sd, code); p = one(Sp, code); RES[code] = (f, n, d, p)
    ref = RES['A1']
    for code, (f, n, d, p) in RES.items():
        ok = code not in ('A0', 'A1') and all(x['pay'] > r_['pay'] and x['open1'] >= r_['open1'] - 1e-9 and x['any'] >= r_['any'] - 1e-9 for x, r_ in ((f, ref[0]), (p, ref[3]))) and n['pay'] >= ref[1]['pay']
        say(f"   {code:3s} {POL[code][0]:54s} | ${f['pay']:7,.0f} / ${f['med']:7,.0f}        | {f['any'] * 100:9.0f}% | {f['open1'] * 100:5.0f}% / {f['open2'] * 100:3.0f}%    | {f['units']:9.2f} | "
            f"${n['pay']:7,.0f}, {n['open1'] * 100:3.0f}%            | ${d['pay']:7,.0f}" + ("  BETTER" if ok else ""))
        if ok: better.append(code)
    best = max(better, key=lambda c: RES[c][0]['pay']) if better else 'A1'
    say(f"   -> funded policy used in part 2: {best} ({POL[best][0]})" + ("" if better else " - nothing beats the reference on all conditions"))
    say(); say(f"PART 2 - SEVERAL ACCOUNTS WITH REPLACEMENT, 2 years: evaluation at model 1 + band 6 -> funded with {best}; a lost funded account is replaced by a new evaluation; every slot takes the same trades")
    say(f"   {'strength':18s} | slots, start      | out of pocket at the worst point: median / bad case (5%) | net after 2 years: avg / median / 5% case | net loss | payouts a month in year 2: avg / median | "
        f"funded accounts alive at the end (avg) | evaluations passed / funded accounts lost (avg)")
    for lab, Sx in (('FAIR', S), ('MODEL LEG NO EDGE', Sn), ('AS BACKTESTED', Sd)):
        for k, stg in ((1, 0), (3, 0), (3, 21), (5, 0), (5, 21), (10, 0), (10, 21)):
            if lab != 'FAIR' and (k, stg) not in ((1, 0), (5, 21), (10, 21)): continue
            r = factory(BL, Sx, a, k, stg, POL[best][1], POL[best][2]); cum = np.cumsum(r.CF, axis=1); trough = np.minimum(cum.min(1), 0.0); net = cum[:, -1]; y2 = r.pay[:, 252:].sum(1) / 12
            say(f"   {lab:18s} | {k:2d}, {'together' if stg == 0 else 'one a month':11s} | ${-np.median(trough):6,.0f} / ${-np.percentile(trough, 5):6,.0f}{'':32s} | ${net.mean():8,.0f} / ${np.median(net):8,.0f} / ${np.percentile(net, 5):8,.0f} | "
                f"{np.mean(net < 0) * 100:7.0f}% | ${y2.mean():7,.0f} / ${np.median(y2):7,.0f}{'':16s} | {r.alive[:, -1].mean():5.1f} of {k:2d}{'':28s} | {r.passed.mean():.1f} / {r.lost.mean():.1f}")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
