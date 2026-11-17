r"""NN (NN_PLAN.txt) step 2: walk-forward neural networks on the GPU (numpy + torch only).
    <python with torch> nn_train.py <nn_grid.npz> <out.npz>
"""
import sys, time
import numpy as np, torch, torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    def __init__(self, nf):
        super().__init__(); self.f = nn.Sequential(nn.Linear(nf, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1))
        self.h = nn.Linear(64, 1)

    def forward(self, x, s=None):
        return self.h(self.f(x)).squeeze(-1)


class Hybrid(nn.Module):
    def __init__(self, nf):
        super().__init__(); self.f = nn.Sequential(nn.Linear(nf, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1))
        self.c = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU())
        self.h = nn.Sequential(nn.Linear(64 + 64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, s):
        c = self.c(s); z = torch.cat([self.f(x), c.mean(-1), c.amax(-1)], -1); return self.h(z).squeeze(-1)


def fit_predict(kind, Xtr, Str, ytr, Xte, Ste, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = (MLP if kind == 'mlp' else Hybrid)(Xtr.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lossf = nn.HuberLoss(delta=1.0)
    Xt = torch.tensor(Xtr, device=DEV); St = torch.tensor(Str, device=DEV); yt = torch.tensor(ytr, device=DEV); n = len(ytr)
    for ep in range(6):
        m.train(); perm = torch.randperm(n, device=DEV)
        for a in range(0, n, 4096):
            b = perm[a:a + 4096]; opt.zero_grad(); loss = lossf(m(Xt[b], St[b]), yt[b]); loss.backward(); opt.step()
    m.eval(); out = []
    with torch.no_grad():
        for A, B in ((Xtr, Str), (Xte, Ste)):
            r = []
            for a in range(0, len(A), 16384): r.append(m(torch.tensor(A[a:a + 16384], device=DEV), torch.tensor(B[a:a + 16384], device=DEV)).cpu().numpy())
            out.append(np.concatenate(r))
    return out


def main():
    t0 = time.time(); z = np.load(sys.argv[1]); X = z['X']; y = z['y']; yr = z['yr']; f5 = z['f5']; S = z['S']; preds = {}
    for kind in ('mlp', 'hybrid'):
        pred = np.full(len(y), np.nan, np.float32)
        for yy in range(2013, int(yr.max()) + 1):
            tr = np.flatnonzero((yr < yy) & np.isfinite(y)); te = np.flatnonzero(yr == yy)
            if not len(te): continue
            mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd[~(sd > 0)] = 1.0
            Xtr = np.nan_to_num((X[tr] - mu) / sd).astype(np.float32); Xte = np.nan_to_num((X[te] - mu) / sd).astype(np.float32)
            smu = S[tr].mean((0, 2), keepdims=True); ssd = S[tr].std((0, 2), keepdims=True) + 1e-9
            Str = ((S[tr] - smu) / ssd).astype(np.float32); Ste = ((S[te] - smu) / ssd).astype(np.float32)
            ys = y[tr].std(); ytr = (y[tr] / ys).astype(np.float32)
            ins, out = np.zeros(len(tr)), np.zeros(len(te))
            for seed in (0, 1, 2):
                a, b = fit_predict(kind, Xtr, Str, ytr, Xte, Ste, seed); ins += a / 3; out += b / 3
            okf = np.isfinite(f5[tr]); k = np.nanstd(f5[tr][okf]) / ins[okf].std() if ins[okf].std() > 0 else 0.0
            pred[te] = out * k
            print(f"{kind} {yy}: trained on {len(tr):,} rows, {time.time() - t0:.0f} s", flush=True)
        preds[kind] = pred
    np.savez(sys.argv[2], mlp=preds['mlp'], hybrid=preds['hybrid'])
    print(f"done on {DEV}, {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
