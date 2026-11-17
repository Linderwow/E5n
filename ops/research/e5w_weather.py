r"""E5W evaluation weather (E5W_PLAN.txt): does the trailing volatility at the start change a Bulenox 50K evaluation's pass rate?
    python e5w_weather.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import e5k_fair as K
from e5b_lab import G, say
X = L.X; U = L.U


def days(net):
    M = C.M; c1 = X.SIDE(2.35) * U; tr = np.abs(np.diff(np.c_[np.zeros(M.n), net], axis=1))
    cum = np.cumsum(net * M.dpx, axis=1) / M.o930[:, None] * U - c1 * np.cumsum(tr, axis=1)
    close = cum[:, -1] - c1 * np.abs(net[:, -1]); worst = np.minimum(np.minimum(cum.min(axis=1), close), 0.0)
    return close, worst


def main():
    t0 = time.time(); C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0)
    lp = np.log(np.c_[M.px, M.exit])
    r5 = np.diff(lp[:, ::5], axis=1); rv = np.sqrt(np.nansum(r5 ** 2, axis=1))
    keep = np.asarray(M.keep); dates = M.dates
    idx = np.where(keep)[0]; d = dates[idx]; rvk = rv[idx]
    trail = pd.Series(rvk).rolling(20).mean().shift(1).values
    disc = (d < L.TR.JOIN) if hasattr(L.TR, 'JOIN') else (d < pd.Timestamp('2019-05-05'))
    say(); say("=" * 160)
    say(f"EVALUATION WEATHER  e5w_weather.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5W_PLAN.txt) Bulenox 50K as stored; every session a start, 21 sessions a try; FAIR book at today's price, live cost")
    say("=" * 160)
    for code, cm, cb in (('W1', 0, 6), ('W2', 1, 6)):
        u, w = days(cm * fair + cb * M.band); u, w = u[idx], w[idx]
        st = np.arange(20, len(idx) - 21); st = st[np.isfinite(trail[st])]
        Xw = np.stack([u[s:s + 21] for s in st]); Ww = np.stack([w[s:s + 21] for s in st])
        ok, nres = BL.evaluation(Xw, Ww, a['dd'], a['dll'], a['target'], resets=True)
        sd = d[st]; tv = trail[st]; isd = disc[st]
        cuts = np.quantile(tv[isd], [1 / 3, 2 / 3]); terc = np.digitize(tv, cuts)
        say(f"{code} model {cm} + band {cb}: tercile cuts of trailing volatility {cuts[0] * 100:.2f}% / {cuts[1] * 100:.2f}% a day (set on 2013-19)")
        for lab, sel in (('discovery 2013-01 .. 2019-05', isd), ('confirmation 2019-05 .. 2026-10', ~isd)):
            rates = [ok[sel & (terc == k)].mean() for k in range(3)]; ns = [int((sel & (terc == k)).sum()) for k in range(3)]
            fees = [(a['qual'] + BL.RESET_FEE * nres[sel & (terc == k)].mean()) / max(rates[k], 1e-9) for k in range(3)]
            mon = pd.PeriodIndex(sd[sel], freq='M'); um = np.unique(mon); rng = np.random.default_rng(20261115); diffs = []
            ok_s, t_s = ok[sel], terc[sel]; mi = {m: np.where(mon == m)[0] for m in um}
            for _ in range(2000):
                pick = np.concatenate([mi[m] for m in rng.choice(um, len(um))]); oo, tt = ok_s[pick], t_s[pick]
                if (tt == 2).any() and (tt == 0).any(): diffs.append(oo[tt == 2].mean() - oo[tt == 0].mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            say(f"   {lab}: pass a try  quiet {rates[0] * 100:4.0f}% ({ns[0]:,} starts, ${fees[0]:,.0f} a pass) | middle {rates[1] * 100:4.0f}% ({ns[1]:,}, ${fees[1]:,.0f}) | "
                f"busy {rates[2] * 100:4.0f}% ({ns[2]:,}, ${fees[2]:,.0f}) | all {ok[sel].mean() * 100:4.0f}% | busy - quiet {(rates[2] - rates[0]) * 100:+5.1f} points (95% {lo * 100:+.1f} .. {hi * 100:+.1f})")
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
