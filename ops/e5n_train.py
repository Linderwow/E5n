r"""E5n training and export (run once, and again each January): refits the walk-forward yearly models and writes the model file the
NinjaTrader strategy loads, plus the parity data for the offline C# test.
    python e5n_train.py export      -> ..\nt8\e5n\e5n_models.txt and ops\parity\*.bin
"""
import os, sys, pickle, struct, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESEARCH = os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8', 'agent_exports', 'review', 'research')
import numpy as np, pandas as pd
import e5n_paths as PATHS
import e5n_features as F
import e5n_calendar as CAL
from e5n_model import predict_nodes, ModelSet

MODELS = os.path.join(os.path.dirname(HERE), 'nt8', 'e5n', 'e5n_models.txt')
PARITY = os.path.join(HERE, 'parity')
JOIN = pd.Timestamp('2019-05-05')
TRAIN_END = pd.Timestamp('2026-10-15')


def research():
    sys.path.insert(0, RESEARCH)
    import src2_lib as L2, ml_lab as ML, ml_run as MR, ind_lab as IL
    L2.CACHE = PATHS.bars_1m; L2.DB = PATHS.DATABENTO
    return L2, ML, MR, IL


class Raw:
    def __init__(self, L2, name):
        mk = L2.Mins(name); rows = []; O = []; H = []; Lo = []; C = []; V = []; M = []; R = []
        for d in mk.dates:
            i0, i1 = mk.days[d]
            if mk.m[i1] < 959: continue
            pos = mk.m[i0:i1 + 1] - 570; ok = (pos >= 0) & (pos < 390)
            a = {k: np.full(390, np.nan) for k in 'ohlcv'}; mask = np.zeros(390, np.uint8); mask[pos[ok]] = 1
            for k, src in (('o', mk.o), ('h', mk.h), ('l', mk.l), ('c', mk.c), ('v', mk.v)): a[k][pos[ok]] = src[i0:i1 + 1][ok]
            R.append(np.stack([a[k] for k in 'ohlcv'], 1))
            c = pd.Series(a['c']).ffill().bfill().values; o = np.where(np.isnan(a['o']), c, a['o'])
            h = np.where(np.isnan(a['h']), c, a['h']); l = np.where(np.isnan(a['l']), c, a['l']); v = np.nan_to_num(a['v'])
            rows.append(dict(date=pd.Timestamp(d), iid=int(mk.iid[i0]))); O.append(o); H.append(h); Lo.append(l); C.append(c); V.append(v); M.append(mask)
        self.D = pd.DataFrame(rows); self.O, self.H, self.L, self.C, self.V = (np.array(x) for x in (O, H, Lo, C, V))
        self.mask = np.array(M); self.raw = np.array(R)


def nodes_of(model):
    trees = []
    for it in model._predictors:
        nd = it[0].nodes
        trees.append([(int(x['feature_idx']), float(x['num_threshold']), int(x['left']), int(x['right']), int(x['missing_go_to_left']),
                       int(x['is_leaf']), float(x['value'])) for x in nd])
    return float(np.asarray(model._baseline_prediction).ravel()[0]), trees


def write_models(path, models):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        fh.write(f"E5N_MODELS 1\nfeatures {len(F.FEATS)}\n" + "\n".join(F.FEATS) + f"\nthreshold {F.THRESHOLD!r}\nmodels {len(models)}\n")
        for name, d0, d1, base, trees in models:
            fh.write(f"model {name} {d0} {d1} {base!r} {len(trees)}\n")
            for tr in trees:
                fh.write(f"tree {len(tr)}\n")
                for f, th, le, ri, ml, leaf, val in tr:
                    fh.write(f"L {val!r}\n" if leaf else f"N {f} {th!r} {le} {ri} {ml}\n")


def export():
    t0 = time.time(); L2, ML, MR, IL = research()
    CROSS = ['x_r15', 'x_r60', 'x_ropen', 'rel15', 'rel_open']
    IL.FEATS = [f for f in IL.FEATS if f not in CROSS]
    assert IL.FEATS == F.FEATS, "feature order differs from the research definition"
    assert abs(F.THRESHOLD - 4 * MR.SIDE(2.35)) < 1e-18 and F.N_USD == L2.N_USD
    raw = {'nq': Raw(L2, 'nq'), 'mnqx': Raw(L2, 'mnqx')}
    tabs = {}
    for nm in raw:
        S = ML.Sessions(nm); assert np.array_equal(S.C, raw[nm].C) and np.array_equal(S.V, raw[nm].V)
        A = IL.features(S, None, 5, 5); A = A[A.dec >= 15].reset_index(drop=True)
        B = F.features(raw[nm])
        same = all(np.array_equal(A[c].values, B[c].values, equal_nan=True) for c in F.FEATS + ['y', 'entry', 'exitp', 'dec', 'k'])
        cal = np.array([CAL.month_end_flag(d.date()) for d in raw[nm].D.date]); ds = F.mend_flags(pd.DatetimeIndex(raw[nm].D.date.values))
        print(f"{nm}: standalone features == research features: {same}; calendar month-end flag agrees with the dataset flag on "
              f"{np.mean(cal[:-2] == ds[:-2]):.2%} of sessions", flush=True)
        assert same
        tabs[nm] = B
    Ta = tabs['nq'][tabs['nq'].date <= JOIN].assign(mkt=0); Tb = tabs['mnqx'][tabs['mnqx'].date > JOIN].assign(mkt=1)
    T = pd.concat([Ta, Tb], ignore_index=True); yr = T.date.dt.year.values; X = T[F.FEATS].values; y = T.y.values
    pred = np.full(len(T), np.nan); models = []
    for yy in range(2013, 2027):
        tr = (yr < yy) & np.isfinite(y); te = yr == yy
        m = MR.model(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        base, trees = nodes_of(m); mine = predict_nodes(base, trees, X[te]); d = float(np.max(np.abs(mine - pred[te])))
        assert d < 1e-15, d
        models.append((f"wf{yy}", f"{yy}-01-01", f"{yy}-12-31" if yy < 2026 else str(TRAIN_END.date()), base, trees))
        print(f"   {yy}: trained on {int(tr.sum()):,} rows, {len(trees)} trees, evaluator max |diff| {d:.1e}; {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(PATHS.CACHE, 'e5n_model.pkl'), 'rb') as fh: Mf = pickle.load(fh)
    assert Mf['feats'] == F.FEATS
    base, trees = nodes_of(Mf['model']); chk = float(np.max(np.abs(predict_nodes(base, trees, X[-5000:]) - Mf['model'].predict(X[-5000:])))); assert chk < 1e-15
    models.append(('final', str((TRAIN_END + pd.Timedelta(days=1)).date()), '9999-12-31', base, trees))
    write_models(MODELS, models)
    print(f"wrote {MODELS} ({os.path.getsize(MODELS) / 1e6:.1f} MB, {len(models)} models)", flush=True)
    back, _ = ModelSet(MODELS).predict(T.date.values, X); ok = np.isfinite(pred)
    assert np.array_equal(back[ok], pred[ok]), 'model file read back differs'
    print('model file read back by date reproduces every walk-forward forecast exactly', flush=True)
    os.makedirs(PARITY, exist_ok=True)
    for nm, mk in (('nq', 0), ('mnqx', 1)):
        R = raw[nm]; n = len(R.D)
        with open(os.path.join(PARITY, f'{nm}_minutes.bin'), 'wb') as fh:
            fh.write(struct.pack('<i', n))
            for k in range(n):
                d = R.D.date.iat[k]; fh.write(struct.pack('<iq', d.year * 10000 + d.month * 100 + d.day, int(R.D.iid.iat[k])))
                fh.write(R.mask[k].tobytes()); fh.write(np.nan_to_num(R.raw[k]).astype('<f8').tobytes())
        sub = T[T.mkt == mk]; P = pred[(T.mkt == mk).values]
        with open(os.path.join(PARITY, f'{nm}_expected.bin'), 'wb') as fh:
            fh.write(struct.pack('<ii', len(sub), len(F.FEATS)))
            dd = sub.date.dt.year.values * 10000 + sub.date.dt.month.values * 100 + sub.date.dt.day.values
            fh.write(dd.astype('<i4').tobytes()); fh.write(sub.dec.values.astype('<i4').tobytes())
            fh.write(sub[F.FEATS].values.astype('<f8').tobytes()); fh.write(P.astype('<f8').tobytes())
            fh.write(sub.entry.values.astype('<f8').tobytes()); fh.write(sub.exitp.values.astype('<f8').tobytes()); fh.write(sub.open930.values.astype('<f8').tobytes())
    T['pred'] = pred; T[['date', 'dec', 'mkt', 'pred']].to_parquet(os.path.join(PARITY, 'walk_pred.parquet'))
    print(f"parity data in {PARITY}; {time.time() - t0:.0f} s")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export': export()
    else: print(__doc__)
