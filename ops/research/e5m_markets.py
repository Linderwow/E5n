r"""E5M - E5n's model leg on other markets (E5M_PLAN.txt).
    python e5m_markets.py build    one-minute caches, features and walk-forward forecasts per market -> cache\e5m_*
    python e5m_markets.py run      trials -> E5M_RESULTS.txt
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_features as F
import e5n_train as TR

C = os.path.join(X.CACHE, 'e5m_'); OUT = os.path.join(HERE, 'E5M_RESULTS.txt')
MK = {'mym': ('MYM_ohlcv-1m_2019-2026.dbn.zst', 0.5, 1.0, 'Dow (MYM)'), 'm2k': ('M2K_ohlcv-1m_2019-2026.dbn.zst', 5.0, 0.1, 'Russell 2000 (M2K)'),
      'mgc': ('MGC_ohlcv-1m_2019-2026.dbn.zst', 10.0, 0.1, 'gold (MGC)'), 'mcl': ('MCL_ohlcv-1m_2021-2026.dbn.zst', 100.0, 0.01, 'crude oil (MCL)'),
      'm6e': ('M6E_ohlcv-1m_2019-2026.dbn.zst', 12500.0, 0.0001, 'euro (M6E)'), 'mbt': ('MBT_ohlcv-1m_2021-2026.dbn.zst', 0.1, 5.0, 'bitcoin (MBT)'),
      'zn': ('ZN_ohlcv-1m_2019-2026.dbn.zst', 1000.0, 1 / 64, '10-year note (ZN)')}
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def setup_l2():
    L2, ML, MR, IL = TR.research()
    for k, v in MK.items(): L2.SRC[k] = [v[0]]
    L2.CACHE = lambda name: (C + name + '_1m.npz') if name in MK else X.PATHS.bars_1m(name)
    return L2, MR


def build():
    t0 = time.time(); L2, MR = setup_l2()
    for k in MK:
        if not os.path.exists(L2.CACHE(k)): L2.build(k)
        R = TR.Raw(L2, k); T = F.features(R); kk = T.k.values; p_ = T.dec.values
        T['exit_dt'] = 0; T.to_parquet(C + k + '.parquet')
        first = int(T.date.dt.year.min()) + 2
        pred = X.walk(T, T.y.values, MR.model, first=first); np.save(C + k + '_f5.npy', pred)
        print(f"{k}: {T.k.nunique():,} sessions {T.date.min():%Y-%m-%d} .. {T.date.max():%Y-%m-%d}, out of sample from {first}, {time.time() - t0:.0f} s", flush=True)


def tstat(s):
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std() > 0 else 0.0


def run_market(T, f, mult, tick, ticks, thr_ticks=2.0):
    k = T.k.values; st = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]; en = np.r_[st[1:], len(k)]
    entry = T.entry.values; exitp = T.exitp.values; o930 = T.open930.values; dates = pd.DatetimeIndex(T.date.values[st])
    p_now = float(T.exitp.values[-1]); notional = p_now * mult
    c_now = (tick * mult * ticks + 0.61) / notional; thr = 4 * (tick * mult * thr_ticks + 0.61) / notional
    out = np.full(len(st), np.nan); trips = 0
    for d in range(len(st)):
        a, b = st[d], en[d]; pos = 0; pnl = 0.0; sides = 0; last = 0.0; seen = False
        for i in range(a, b):
            fi = f[i]
            if fi != fi: continue
            seen = True; new = 1 if fi > thr else (-1 if fi < -thr else 0)
            if pos != 0: pnl += pos * (entry[i] - last); last = entry[i]
            if new != pos: sides += abs(new - pos); pos = new; last = entry[i]
        if pos != 0: pnl += pos * (exitp[b - 1] - last); sides += 1
        if seen: out[d] = (pnl / o930[a] - sides * c_now) * notional; trips += sides / 2
    s = pd.Series(out, index=dates).dropna()
    return s, trips


def run():
    t0 = time.time(); import e5b_lab as L
    L.setup(); book = L.sim(L.G.model, L.G.band)
    say(); say("=" * 160); say(f"E5M  e5m_markets.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  E5n's model leg trained on other markets; $ per contract at today's price; costs on every side"); say("=" * 160)
    cand = []; legs = {}
    for k, (_, mult, tick, name) in MK.items():
        T = pd.read_parquet(C + k + '.parquet'); f = np.load(C + k + '_f5.npy')
        s, n = run_market(T, f, mult, tick, 2.0); s3, _ = run_market(T, f, mult, tick, 3.0); z, _ = run_market(T, f, mult, tick, 0.0)
        y0 = int(s.index.year.min()); cut = pd.Timestamp(f'{y0 + (3 if y0 <= 2021 else 2)}-01-01'); A, B = s[s.index < cut], s[s.index >= cut]
        yrs = s.groupby(s.index.year).sum(); d5 = s.drop(s.sort_values(ascending=False).index[:5])
        j = pd.concat([book.rename('book'), s.rename('leg')], axis=1, join='inner').fillna(0.0); jA, jB = j[j.index < cut], j[j.index >= cut]
        ok = {'t >= 2.7': tstat(s) >= 2.7, 'both halves positive': A.sum() > 0 and B.sum() > 0, 'at most 1 losing year': int((yrs < 0).sum()) <= 1,
              'positive without its 5 best days': d5.sum() > 0, 'positive at 3 ticks': s3.sum() > 0,
              'book Sharpe up in both halves': X.sharpe(jA.book + jA.leg) > X.sharpe(jA.book) and X.sharpe(jB.book + jB.leg) > X.sharpe(jB.book)}
        fails = [x for x, v in ok.items() if not v]; legs[k] = s
        say(); say(f"{name}: {len(s):,} out-of-sample sessions from {s.index.min():%Y-%m-%d}, {n / (len(s) / 252):,.0f} round trips a year")
        say(f"   ${s.mean() * 252:+,.0f}/yr per contract, t {tstat(s):+.2f}, Sharpe {X.sharpe(s):.2f}, max DD ${L.mdd(s.values):,.0f} | halves ${A.mean() * 252:+,.0f} / ${B.mean() * 252:+,.0f} a year | "
            f"commission only t {tstat(z):+.2f} | 3 ticks ${s3.mean() * 252:+,.0f}/yr | without its 5 best days ${d5.mean() * 252:+,.0f}/yr")
        say("   by year: " + ", ".join(f"{y} {v:+,.0f}" for y, v in yrs.items()) + f" | correlation with the E5n book {j.book.corr(j.leg):+.2f} | "
            f"book Sharpe {X.sharpe(jA.book + jA.leg):.2f} / {X.sharpe(jB.book + jB.leg):.2f} vs {X.sharpe(jA.book):.2f} / {X.sharpe(jB.book):.2f}")
        say(f"   VERDICT {k.upper()}: " + ('FORWARD-TEST CANDIDATE' if not fails else 'dead - fails: ' + '; '.join(fails)))
        if not fails: cand.append(k)
    say(); say("CANDIDATES: " + (', '.join(c.upper() for c in cand) if cand else 'none'))
    if len(cand) >= 1:
        j = pd.concat([book.rename('book')] + [legs[c].rename(c) for c in cand], axis=1, join='inner').fillna(0.0); tot = j.sum(axis=1)
        say(f"   E5n book + 1 contract of each candidate, common days: ${tot.mean() * 252:,.0f}/yr, Sharpe {X.sharpe(tot):.2f}, max DD ${L.mdd(tot.values):,.0f} vs book alone ${j.book.mean() * 252:,.0f}/yr, "
            f"Sharpe {X.sharpe(j.book):.2f}, max DD ${L.mdd(j.book.values):,.0f}")
    say(f"run {time.time() - t0:.0f} s")
    with open(OUT, 'a', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'build': build()
    elif a == 'run': run()
    else: print(__doc__)
