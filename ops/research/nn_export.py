r"""NN (NN_PLAN.txt) step 1: export the research grid for the network trainer (numpy only on the other side).
    python nn_export.py   -> data\research\nn_grid.npz
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X
TR = X.TR; F = X.F; P = X.PATHS; SEQ = 60


def seqs(R, kk, dec):
    C = R.C.astype(np.float64); H = R.H; L = R.L; V = np.log1p(R.V)
    ret = np.zeros_like(C); ret[:, 1:] = np.log(C[:, 1:] / C[:, :-1]); rng = (H - L) / C
    med = np.zeros((C.shape[0], 391))
    for d in range(1, 391): med[:, d] = np.median(V[:, :d], axis=1)
    out = np.zeros((len(kk), 3, SEQ), np.float32)
    for j in range(SEQ):
        p = dec - SEQ + j; ok = p >= 0; pp = np.clip(p, 0, 389)
        out[:, 0, j] = np.where(ok, ret[kk, pp], 0.0); out[:, 1, j] = np.where(ok, rng[kk, pp], 0.0); out[:, 2, j] = np.where(ok, V[kk, pp] - med[kk, dec], 0.0)
    return out


def main():
    t0 = time.time(); X.setup(); T = X.NQ.T; L2, ML, MR, IL = TR.research()
    Rn = TR.Raw(L2, 'nq'); Rm = TR.Raw(L2, 'mnqx'); k = T.k.values; dec = T.dec.values.astype(int); isB = k >= 10 ** 6
    S = np.zeros((len(T), 3, SEQ), np.float32)
    S[~isB] = seqs(Rn, k[~isB], dec[~isB]); S[isB] = seqs(Rm, k[isB] - 10 ** 6, dec[isB])
    np.savez(os.path.join(P.CACHE, 'nn_grid.npz'), X=T[F.FEATS].values.astype(np.float32), y=T.y.values.astype(np.float32), yr=T.date.dt.year.values.astype(np.int16),
             f5=np.asarray(X.F_NQ, np.float32), S=S)
    print(f"{len(T):,} rows, {len(F.FEATS)} features, sequences {S.shape}, NaN in sequences {np.isnan(S).sum()}, {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
