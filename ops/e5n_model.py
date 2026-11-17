import numpy as np, pandas as pd


def predict_nodes(base, trees, X):
    out = np.full(len(X), base)
    for tr in trees:
        f = np.array([t[0] for t in tr]); th = np.array([t[1] for t in tr]); le = np.array([t[2] for t in tr]); ri = np.array([t[3] for t in tr])
        ml = np.array([t[4] for t in tr], bool); leaf = np.array([t[5] for t in tr], bool); val = np.array([t[6] for t in tr])
        idx = np.zeros(len(X), np.int64); live = ~leaf[idx]
        while live.any():
            i = idx[live]; x = X[live, f[i]]
            go = np.where(np.isnan(x), ml[i], x <= th[i]); idx[live] = np.where(go, le[i], ri[i]); live = ~leaf[idx]
        out += val[idx]
    return out


class ModelSet:
    def __init__(self, path):
        with open(path, encoding='ascii') as fh: L = fh.read().split('\n')
        assert L[0].startswith('E5N_MODELS'), path
        nf = int(L[1].split()[1]); self.features = L[2:2 + nf]; i = 2 + nf
        self.threshold = float(L[i].split()[1]); nm = int(L[i + 1].split()[1]); i += 2; self.models = []
        for _ in range(nm):
            h = L[i].split(); i += 1; trees = []
            for _t in range(int(h[5])):
                nn = int(L[i].split()[1]); i += 1; tr = []
                for ln in L[i:i + nn]:
                    p = ln.split()
                    tr.append((0, 0.0, 0, 0, 0, 1, float(p[1])) if p[0] == 'L' else (int(p[1]), float(p[2]), int(p[3]), int(p[4]), int(p[5]), 0, 0.0))
                trees.append(tr); i += nn
            self.models.append(dict(name=h[1], d0=pd.Timestamp(h[2]), d1=pd.Timestamp(h[3]) if h[3] < '9999' else pd.Timestamp.max, base=float(h[4]), trees=trees))

    def predict(self, dates, X):
        dates = pd.DatetimeIndex(dates); out = np.full(len(X), np.nan); names = np.full(len(X), '', dtype=object)
        for m in self.models:
            sel = np.asarray((dates >= m['d0']) & (dates <= m['d1']))
            if sel.any(): out[sel] = predict_nodes(m['base'], m['trees'], X[sel]); names[sel] = m['name']
        return out, names
