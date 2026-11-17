r"""UT T3B (UT_PLAN.txt amendment 2): E5n's model leg decided on the bar D seconds before its close vs today (final bar, filled 0.4 s
after the close), E5n's own Python model rerun on tick-built decision bars, NinjaTrader's MNQ tick store.
    python ut_timing2.py
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, pandas as pd
import e5n_daily as DLY
import e5n_features as F
from e5n_model import ModelSet
import ut_tests as U

DS = (2.0, 5.0); FILL_LAG = 0.4; COMM = 0.61; START = '2025-10-28'


def one_day(files):
    import ncd_reader as RD, ncd_export as E
    ts, px, vo = [], [], []
    for p in files:
        try: d, meta = RD.decode_tick_bytes(E._read_shared(p))
        except Exception: continue
        lt = d['local_ticks']
        if len(lt): ts.append((lt - U.EPOCH) // 10_000); px.append(d['price_ticks'] * meta['tick_size']); vo.append(d['volume'].astype(float))
    if not ts: return None
    ms = np.concatenate(ts); o = np.argsort(ms, kind='stable'); return ms[o], np.concatenate(px)[o], np.concatenate(vo)[o]


def tick_days():
    jobs = {}
    for inst in sorted(os.listdir(U.TICKS)):
        if not inst.startswith('MNQ '): continue
        for fn in os.listdir(os.path.join(U.TICKS, inst)):
            if fn.endswith('.Last.ncd') and fn[:8] >= START.replace('-', ''): jobs.setdefault((inst, fn[:8]), []).append(os.path.join(U.TICKS, inst, fn))
    keys = sorted(jobs)
    with ProcessPoolExecutor(8) as ex: data = dict(zip(keys, ex.map(one_day, [sorted(jobs[k]) for k in keys])))
    best = {}
    for (inst, day), v in data.items():
        if v is not None and (day not in best or len(v[0]) > len(best[day][0])): best[day] = v
    return {pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}"): v for d, v in best.items()}


def sub(S, idx):
    T = DLY.Sessions.__new__(DLY.Sessions); T.D = S.D.iloc[idx].reset_index(drop=True)
    for k in ('O', 'H', 'L', 'C', 'V', 'present'): setattr(T, k, getattr(S, k)[idx].copy())
    return T


def main():
    t0 = time.time(); M = ModelSet(DLY.MODELS); M15 = ModelSet(DLY.MODELS15)
    a, keep = DLY.all_bars(); S0 = DLY.Sessions(DLY.minute_bars(keep))
    first = np.searchsorted(S0.D.date.values, np.datetime64(START)); S = sub(S0, np.arange(max(0, first - 40), len(S0.D)))
    TK = tick_days(); print(f"sessions {len(S.D)}, tick days {len(TK)}, {time.time() - t0:.0f} s", flush=True)
    decs = np.arange(15, 390, 5); epoch = pd.Timestamp('1970-01-01')
    T0 = {k: int((pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30) - epoch).total_seconds() * 1000) for k, d in enumerate(S.D.date)}
    def px_at(k, t):
        v = TK.get(pd.Timestamp(S.D.date.iat[k]))
        if v is None: return np.nan
        i = np.searchsorted(v[0], t, side='right') - 1; return v[1][i] if i >= 0 else np.nan
    def tick_bar(k, lo, hi):
        v = TK.get(pd.Timestamp(S.D.date.iat[k]))
        if v is None: return None
        a_, b_ = np.searchsorted(v[0], lo, side='left'), np.searchsorted(v[0], hi, side='left')
        if b_ <= a_: return None
        p = v[1][a_:b_]; return p[0], p.max(), p.min(), p[-1], v[2][a_:b_].sum()
    ks = [k for k in range(len(S.D)) if pd.Timestamp(S.D.date.iat[k]) in TK and k >= DLY.WARMUP]
    out = {lab: {} for lab in ('final', *[f'D{d:g}' for d in DS])}
    for dec in decs:
        for lab, cut in (('final', 0.0), *[(f'D{d:g}', d) for d in DS]):
            Sx = sub(S, np.arange(len(S.D))); col = dec - 1
            for k in ks:
                bar_start = T0[k] + col * 60_000; tb = tick_bar(k, bar_start, bar_start + 60_000 - int(cut * 1000))
                if tb is None: continue
                Sx.O[k, col], Sx.H[k, col], Sx.L[k, col], Sx.C[k, col], Sx.V[k, col] = tb
            T = F.features(Sx); T['mend'] = np.array([DLY.CAL.month_end_flag(d.date()) for d in Sx.D.date], float)[T.k.values]
            T = T[(T.dec == dec) & T.k.isin(ks)]
            if not len(T): continue
            X_ = T[F.FEATS].values; p5, _ = M.predict(T.date.values, X_); p15, _ = M15.predict(T.date.values, X_)
            for k, f5, f15 in zip(T.k.values, p5, p15):
                base = 1 if f5 > M.threshold else (-1 if f5 < -M.threshold else 0); out[lab][(k, dec)] = base if base != 0 and f15 * base > 0 else 0
        print(f"decision minute {dec} done, {time.time() - t0:.0f} s", flush=True)
    U.LINES.clear(); say = U.say
    say(); say("=" * 150); say(f"UT T3B  DECISION TIMING WITH E5n's OWN MODEL  ut_timing2.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  model leg + 15-minute check, NinjaTrader MNQ tick store"); say("=" * 150)
    keys = sorted(out['final'])
    def leg(lab, lag):
        rows = []
        for k in ks:
            pos = 0; last = np.nan; usd = 0.0; trades = 0
            for dec in decs:
                tgt = out[lab].get((k, dec))
                if tgt is None: continue
                t = T0[k] + dec * 60_000 + int(lag * 1000); p = px_at(k, t)
                if p != p: continue
                if pos != 0: usd += pos * (p - last) * 2.0
                if tgt != pos: usd -= abs(tgt - pos) * COMM; trades += abs(tgt - pos); pos = tgt
                last = p
            if pos != 0:
                p = px_at(k, T0[k] + 390 * 60_000 + 500)
                if p == p: usd += pos * (p - last) * 2.0 - COMM; trades += 1
            rows.append((pd.Timestamp(S.D.date.iat[k]), usd, trades))
        return pd.DataFrame(rows, columns=['date', 'usd', 'sides']).set_index('date')
    base = leg('final', FILL_LAG)
    sv = []
    for k in ks:
        prev = 0
        for dec in decs:
            tgt = out['final'].get((k, dec))
            if tgt is None: continue
            if tgt != prev:
                T = T0[k] + dec * 60_000; sv.append((tgt - prev) * (px_at(k, T + 400) - px_at(k, T - 2000)) / 0.25); prev = tgt
    sv = np.array(sv); sv = sv[np.isfinite(sv)]
    say(f"   check: on these orders, price move in the order's direction from -2 s to +0.4 s = {sv.mean() / np.mean(np.abs(np.sign(sv)) if len(sv) else 1):+.2f} ticks per contract-side over {len(sv)} orders")
    for d in DS:
        pt = leg('final', -d); say(f"   timing only (same final decisions, filled {d:g}s before the close): ${pt.usd.sum():+,.0f} vs ${base.usd.sum():+,.0f}")
    say(f"   {len(ks)} sessions {pd.Timestamp(S.D.date.iat[ks[0]]):%Y-%m-%d} .. {pd.Timestamp(S.D.date.iat[ks[-1]]):%Y-%m-%d}; today's leg (final bar, filled +{FILL_LAG}s): ${base.usd.sum():+,.0f}, {base.sides.sum():,.0f} sides")
    for d in DS:
        lab = f'D{d:g}'; agree = np.mean([out[lab].get(x) == out['final'][x] for x in keys]); e = leg(lab, -d)
        diff = e.usd - base.usd; t = diff.mean() / diff.std() * np.sqrt(len(diff)) if diff.std() > 0 else np.nan
        mon = diff.groupby(diff.index.to_period('M')).sum(); worse = (mon < 0).mean()
        ok = t >= 3.0 and agree >= 0.90 and worse <= 1 / 3
        say(f"   decide {d:g}s early: decisions agree {agree:.1%}; leg ${e.usd.sum():+,.0f} ({e.sides.sum():,.0f} sides) vs ${base.usd.sum():+,.0f}: difference ${diff.sum():+,.0f} = ${diff.mean():+.2f} a session, "
            f"paired t {t:+.2f}, months worse {worse:.0%} -> {'ADOPT CANDIDATE (forward shadow next)' if ok else 'no'}")
    say(f"run {time.time() - t0:.0f} s"); U.flush()


if __name__ == '__main__':
    main()
