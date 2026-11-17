r"""E5B amendment H: model consensus (MC), thresholds with the check (TH), stricter check (CK), long-only band leg (LO).
    python e5m_final.py
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5b_lab as L
import e5d_more as D
from e5b_lab import G, say
X = L.X; ND = L.ND; T0 = L.T0


def main():
    t0 = time.time(); L.setup(); m, b = G.model, G.band
    say(); say("=" * 160); say(f"E5B AMENDMENT H  e5m_final.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  model consensus, thresholds with the check, stricter check, long-only band; $ per micro at today's price, live cost on net orders"); say("=" * 160)
    book = lambda t: L.sim(m, b, ticks=t)
    dec_m = np.arange(5, 386); on = (dec_m % 5 == 0) & (dec_m >= 15)
    a5 = np.load(os.path.join(X.CACHE, 'e5c_a5.npy'))[:, on]; a15 = np.load(os.path.join(X.CACHE, 'e5c_a15.npy'))[:, on]
    pa = L.model_path(f=a5, check=a15); both = np.where((m == pa), m, 0.0); union = np.where(m != 0, m, pa); avg = L.model_path(f=(G.f5 + a5) / 2, check=(G.f15 + a15) / 2)
    MC = [dict(name='MC1', desc='model leg only when the deployed and the retrained models agree', fn=lambda t: L.sim(net=both + b, ticks=t)),
          dict(name='MC2', desc='model leg from the average of the two models', fn=lambda t: L.sim(net=avg + b, ticks=t)),
          dict(name='MC3', desc='model leg when either model signals (deployed first)', fn=lambda t: L.sim(net=union + b, ticks=t))]
    D.judge('MC', MC, book, 'the netted book B0')
    TH = [dict(name=f'TH{k}', desc=f'model-leg threshold {mult} x T with the check', fn=(lambda mp: lambda t: L.sim(net=mp + b, ticks=t))(L.model_path(thr=mult * T0)))
          for k, mult in ((1, 0.5), (2, 0.75), (3, 1.25))]
    D.judge('TH', TH, book, 'the netted book B0')
    CK = []
    for k, mult in ((1, 1.0), (2, 2.0)):
        with np.errstate(invalid='ignore'): strong = np.where(np.abs(G.f15) > mult * T0, G.f15, 0.0)
        CK.append(dict(name=f'CK{k}', desc=f'check needs the 15-minute forecast beyond {mult} x T', fn=(lambda mp: lambda t: L.sim(net=mp + b, ticks=t))(L.model_path(check=strong))))
    D.judge('CK', CK, book, 'the netted book B0')
    LO = [dict(name='LO1', desc='band leg long only', fn=lambda t: L.sim(net=m + np.maximum(b, 0), ticks=t)),
          dict(name='LO2', desc='band leg shorts at half size', fn=lambda t: L.sim(net=m + np.where(b < 0, 0.5 * b, b), ticks=t))]
    D.judge('LO', LO, book, 'the netted book B0')
    say(f"run {time.time() - t0:.0f} s"); L.flush()


if __name__ == '__main__':
    main()
