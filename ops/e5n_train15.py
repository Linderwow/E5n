r"""E5n 15-minute check models (run after e5n_train.py export, and again each January): refits the walk-forward yearly models of the
15-minute return on the same features and writes the second model file the strategy loads, plus parity data for the offline C# test.
    python e5n_train15.py export      -> ..\nt8\e5n\e5n_models_15min.txt and ops\parity\*_check.bin
"""
import os, sys, struct, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_paths as PATHS
import e5n_features as F
import e5n_train as TR
from e5n_model import predict_nodes, ModelSet

MODELS15 = os.path.join(os.path.dirname(HERE), 'nt8', 'e5n', 'e5n_models_15min.txt')
CACHE = PATHS.CACHE


def table(raw):
    tabs = {}
    for nm in raw:
        B = F.features(raw[nm]); k = B.k.values; p_ = B.dec.values
        B['y15'] = np.log(raw[nm].C[k, np.minimum(p_ + 14, 389)] / raw[nm].O[k, p_]); tabs[nm] = B
    Ta = tabs['nq'][tabs['nq'].date <= TR.JOIN].assign(mkt=0); Tb = tabs['mnqx'][tabs['mnqx'].date > TR.JOIN].assign(mkt=1)
    return pd.concat([Ta, Tb], ignore_index=True)


def export():
    t0 = time.time(); L2, ML, MR, IL = TR.research()
    raw = {'nq': TR.Raw(L2, 'nq'), 'mnqx': TR.Raw(L2, 'mnqx')}
    T = table(raw); yr = T.date.dt.year.values; X = T[F.FEATS].values; y = T.y15.values
    base = pd.read_parquet(os.path.join(TR.PARITY, 'walk_pred.parquet'))
    assert len(base) == len(T) and np.array_equal(base.date.values, T.date.values) and np.array_equal(base.dec.values, T.dec.values), 'rows differ from the base export'
    pred = np.full(len(T), np.nan); models = []
    for yy in range(2013, 2027):
        tr = np.where((yr < yy) & np.isfinite(y))[0]; te = yr == yy
        m = MR.model(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        b, trees = TR.nodes_of(m); d = float(np.max(np.abs(predict_nodes(b, trees, X[te]) - pred[te]))); assert d < 1e-15, d
        models.append((f"chk{yy}", f"{yy}-01-01", f"{yy}-12-31" if yy < 2026 else str(TR.TRAIN_END.date()), b, trees))
        print(f"   {yy}: trained on {len(tr):,} rows, {len(trees)} trees, evaluator max |diff| {d:.1e}; {time.time() - t0:.0f} s", flush=True)
    ok = (T.date <= TR.TRAIN_END).values & np.isfinite(y)
    m = MR.model(); m.fit(X[ok], y[ok]); b, trees = TR.nodes_of(m)
    models.append(('chkfinal', str((TR.TRAIN_END + pd.Timedelta(days=1)).date()), '9999-12-31', b, trees))
    old = os.path.join(HERE, 'data', 'e5n_model_15min.txt')
    if os.path.exists(old):
        a, _ = ModelSet(old).predict(np.array(['2026-11-02'] * 5000, dtype='datetime64[ns]'), X[-5000:])
        print(f"final model vs the shadow ledger's frozen model: max |diff| {np.max(np.abs(a - predict_nodes(b, trees, X[-5000:]))):.1e}", flush=True)
    TR.write_models(MODELS15, models)
    print(f"wrote {MODELS15} ({os.path.getsize(MODELS15) / 1e6:.1f} MB, {len(models)} models)", flush=True)
    back, _ = ModelSet(MODELS15).predict(T.date.values, X); fin = np.isfinite(pred)
    assert np.array_equal(back[fin], pred[fin]), 'model file read back differs'
    cp = os.path.join(CACHE, 'x5_nq.npy'); tp = os.path.join(CACHE, 'nq.parquet'); mp = os.path.join(CACHE, 'mnqx.parquet')
    if os.path.exists(cp) and os.path.exists(tp) and os.path.exists(mp):
        a = pd.read_parquet(tp, columns=['date', 'dec', 'k']); a = a[a.date <= TR.JOIN]
        c = pd.read_parquet(mp, columns=['date', 'dec', 'k']); c = c[c.date > TR.JOIN].assign(k=lambda d: d.k + 10 ** 6)
        R = pd.concat([a, c], ignore_index=True).sort_values(['k', 'dec']).reset_index(drop=True); R['f15'] = np.load(cp)
        j = T[['date', 'dec']].assign(p=pred).merge(R[['date', 'dec', 'f15']], on=['date', 'dec'], how='left')
        both = np.isfinite(j.p.values) & np.isfinite(j.f15.values)
        print(f"walk-forward 15-minute forecasts vs the research cache: {int(both.sum()):,} rows, max |diff| {np.max(np.abs(j.p.values[both] - j.f15.values[both])):.1e}, "
              f"sign differs on {int((np.sign(j.p.values[both]) != np.sign(j.f15.values[both])).sum())}", flush=True)
    for nm, mk in (('nq', 0), ('mnqx', 1)):
        sel = (T.mkt == mk).values
        with open(os.path.join(TR.PARITY, f'{nm}_check.bin'), 'wb') as fh:
            fh.write(struct.pack('<i', int(sel.sum()))); fh.write(pred[sel].astype('<f8').tobytes())
    T['pred15'] = pred; T[['date', 'dec', 'mkt', 'pred15']].to_parquet(os.path.join(TR.PARITY, 'walk_pred15.parquet'))
    print(f"parity data in {TR.PARITY}; {time.time() - t0:.0f} s")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export': export()
    else: print(__doc__)
