r"""MOC amendment 1 (MOC_PLAN.txt): the 3:50 pm NYSE imbalance used inside E5n's book (no extra trades).
    python moc_book.py
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5c_fast as C
import moc_test as MT


def main():
    C.load(); M = C.M; MT.LINES.clear(); say = MT.say
    m = pd.read_csv(MT.SRC, encoding='utf-8-sig'); m['Date'] = pd.to_datetime(m['Date']); N = m.set_index('Date')['Net Notional'].astype(float)
    sg = np.sign(N.reindex(pd.DatetimeIndex(M.dates).normalize()).values); sg = np.nan_to_num(sg)
    col = lambda dec: int(np.flatnonzero(M.dec == dec)[0])
    base_model = C.path(M.g5, M.g15, 'set', M.grid); band = M.band.copy()
    f5 = M.g5.copy(); j = col(385); t = C.target(f5[:, j], M.g15[:, j]); f5[:, j] = np.where(t * sg < 0, 0.0, f5[:, j]); x1 = C.path(f5, M.g15, 'set', M.grid)
    b2 = band.copy(); j0 = col(380); flag = band[:, j0] * sg < 0; b2[np.ix_(flag, np.arange(col(381), C.NM))] = 0.0
    keep = np.asarray(M.keep); d = pd.DatetimeIndex(M.dates)
    A = keep & (d >= MT.DISC[0]) & (d <= MT.DISC[1]); B = keep & (d >= MT.HOLD[0])
    def series(net): s, _ = C.simm(net); return s
    base = series(base_model + band); full = pd.Series(np.nan, index=d); full[keep] = base.values; base = full
    sh = lambda x: x.mean() / x.std() * np.sqrt(252); dd = lambda x: float((x.cumsum().cummax() - x.cumsum()).max())
    say(); say("=" * 150); say(f"MOC AMENDMENT 1  THE 3:50 PM IMBALANCE INSIDE E5n's BOOK  moc_book.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  1 + 1, 2.35 ticks"); say("=" * 150)
    bA, bB = base[A], base[B]
    say(f"   B0 baseline  2015-10..2021 Sharpe {sh(bA):.2f} DD ${dd(bA):,.0f} | 2022..2026-10 Sharpe {sh(bB):.2f} DD ${dd(bB):,.0f} ${bB.mean() * 252:,.0f}/yr")
    for lab, net, n_ch in (('X1 model leg: 15:55 signal against the imbalance -> flat', x1 + band, int((t * sg < 0).sum())),
                           ('X2 band leg: held against the imbalance at 15:50 -> out at 15:51', base_model + b2, int(flag.sum()))):
        s = series(net); f = pd.Series(np.nan, index=d); f[keep] = s.values; sA, sB = f[A], f[B]
        sel = sh(sA) > sh(bA) and dd(sA) <= dd(bA); df = (sB - bB).values; tt = df.mean() / df.std() * np.sqrt(len(df)) if df.std() > 0 else 0.0
        ok = sel and sh(sB) > sh(bB) and dd(sB) <= dd(bB) and tt >= 2.0
        say(f"   {lab:62s} ({n_ch} cases) 2015-21 Sharpe {sh(sA):.2f} ({sh(sA) - sh(bA):+.3f}) DD ${dd(sA):,.0f} {'SEL' if sel else '   '}| 2022-26 Sharpe {sh(sB):.2f} "
            f"({sh(sB) - sh(bB):+.3f}) DD ${dd(sB):,.0f} paired t {tt:+.2f}" + (' CONFIRMED' if ok else ''))
    say("   trial count 10,524")
    with open(MT.OUT, 'a', encoding='utf-8') as fh: fh.write('\n'.join(MT.LINES) + '\n')


if __name__ == '__main__':
    main()
