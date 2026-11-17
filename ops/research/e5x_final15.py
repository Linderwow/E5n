import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5x_lab as X
import e5n_train as TR
import e5n_features as F
from e5n_model import ModelSet

OUT = os.path.join(X.OPS, 'data', 'e5n_model_15min.txt')
L2, ML, MR, IL = TR.research()
T = X.table('nq', 'mnqx'); T = T[T.date <= TR.TRAIN_END]; y = T.y15.values; ok = np.isfinite(y)
m = MR.model(); m.fit(T[F.FEATS].values[ok], y[ok]); base, trees = TR.nodes_of(m)
TR.write_models(OUT, [('final15', str((TR.TRAIN_END + pd.Timedelta(days=1)).date()), '9999-12-31', base, trees)])
chk, _ = ModelSet(OUT).predict(np.array(['2026-11-02'] * 2000, dtype='datetime64[ns]'), T[F.FEATS].values[-2000:])
print(f"wrote {OUT}: trained on {int(ok.sum()):,} rows to {TR.TRAIN_END.date()}; read-back max |diff| {np.max(np.abs(chk - m.predict(T[F.FEATS].values[-2000:]))):.1e}")
