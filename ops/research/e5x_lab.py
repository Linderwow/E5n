r"""E5X - improving E5n (E5X_PLAN.txt).
    python e5x_lab.py data      build the cached tables (features, targets, base forecasts for NQ/MNQ and ES/MES)
    python e5x_lab.py rules     audit the base, then trials X1-X4 and the book X9 -> E5X_RESULTS.txt
"""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); OPS = os.path.dirname(HERE); sys.path.insert(0, OPS)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS
import e5n_features as F
import e5n_train as TR

CACHE = PATHS.CACHE; OUT = os.path.join(HERE, 'E5X_RESULTS.txt')
JOIN = pd.Timestamp('2019-05-05'); U = F.N_USD; T0 = F.THRESHOLD
SIDE = lambda ticks: (0.5 * ticks + 0.61) / U
P_BAR = 0.05 / 9
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def build():
    os.makedirs(CACHE, exist_ok=True); t0 = time.time()
    L2, ML, MR, IL = TR.research()
    for nm in ('nq', 'mnqx', 'es', 'mes'):
        R = TR.Raw(L2, nm); T = F.features(R)
        k = T.k.values; p_ = T.dec.values
        T['y15'] = np.log(R.C[k, np.minimum(p_ + 14, 389)] / R.O[k, p_])
        T.to_parquet(os.path.join(CACHE, f'{nm}.parquet')); print(f"{nm}: {len(T):,} rows, {time.time() - t0:.0f} s", flush=True)
    T = table('es', 'mes'); p = walk(T, T.y.values, MR.model)
    pd.DataFrame({'date': T.date.values, 'dec': T.dec.values, 'pred': p}).to_parquet(os.path.join(CACHE, 'es_base_pred.parquet'))
    print(f"ES/MES base forecasts done, {time.time() - t0:.0f} s")


def table(a, b):
    Ta = pd.read_parquet(os.path.join(CACHE, f'{a}.parquet')); Ta = Ta[Ta.date <= JOIN]
    Tb = pd.read_parquet(os.path.join(CACHE, f'{b}.parquet')); Tb = Tb[Tb.date > JOIN].assign(k=lambda d: d.k + 10 ** 6)
    return pd.concat([Ta, Tb], ignore_index=True).sort_values(['k', 'dec']).reset_index(drop=True)


def walk(T, y, model_fn, X=None, first=2013, sample=None):
    X = T[F.FEATS].values if X is None else X; yr = T.date.dt.year.values; pred = np.full(len(T), np.nan); kk = T.k.values
    for yy in sorted(set(yr)):
        if yy < first: continue
        tr = np.where((yr < yy) & np.isfinite(y))[0]; te = yr == yy
        if not te.any() or len(tr) < 5000: continue
        if sample is not None:
            rng = np.random.default_rng(1000 * sample + yy); ks = np.unique(kk[tr]); pick = rng.choice(ks, len(ks), replace=True)
            order = np.argsort(kk[tr], kind='stable'); ksort = kk[tr][order]
            lo = np.searchsorted(ksort, pick, 'left'); hi = np.searchsorted(ksort, pick, 'right')
            tr = tr[order][np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])]
        m = model_fn(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def base_pred(T, which='nq'):
    if which == 'nq':
        w = pd.read_parquet(os.path.join(OPS, 'parity', 'walk_pred.parquet'))[['date', 'dec', 'pred']]
    else:
        w = pd.read_parquet(os.path.join(CACHE, 'es_base_pred.parquet'))
    return T[['date', 'dec']].merge(w, on=['date', 'dec'], how='left').pred.values


class Book:
    def __init__(self, T):
        self.T = T; k = T.k.values
        self.st = np.r_[0, np.where(k[1:] != k[:-1])[0] + 1]; self.en = np.r_[self.st[1:], len(k)]
        self.entry = T.entry.values; self.exitp = T.exitp.values; self.o930 = T.open930.values
        self.dates = pd.DatetimeIndex(T.date.values[self.st]); self.keep = self.dates.year >= 2013
        self.vr = None

    def volratio(self):
        if self.vr is None:
            piv = self.T.pivot_table(index='k', columns='dec', values='vol30')
            med = piv.rolling(60, min_periods=30).median().shift(1).stack().rename('med').reset_index()
            j = self.T[['k', 'dec', 'vol30']].merge(med, on=['k', 'dec'], how='left')
            with np.errstate(invalid='ignore', divide='ignore'): self.vr = j.vol30.values / j.med.values
        return self.vr

    def run(self, f, rule, ticks=2.35):
        c1 = SIDE(ticks); n = len(self.st); net = np.zeros(n); trd = np.zeros(n); entry = self.entry
        for d in range(n):
            a, b = self.st[d], self.en[d]; pos = 0; pnl = 0.0; sides = 0; last = 0.0
            for i in range(a, b):
                fi = f[i]
                if fi != fi: continue
                new = rule(pos, fi, i)
                if pos != 0: pnl += pos * (entry[i] - last); last = entry[i]
                if new != pos: sides += abs(new - pos); pos = new; last = entry[i]
            if pos != 0: pnl += pos * (self.exitp[b - 1] - last); sides += abs(pos)
            net[d] = pnl / self.o930[a] - sides * c1; trd[d] = sides / 2
        return pd.Series(net[self.keep] * U, index=self.dates[self.keep]), float(trd[self.keep].sum())


def r_base(thr=T0):
    return lambda pos, f, i: 1 if f > thr else (-1 if f < -thr else 0)


def r_hold(stay=0.0, thr=T0):
    def rule(pos, f, i):
        if f > thr: return 1
        if f < -thr: return -1
        if pos > 0: return 1 if f >= stay else 0
        if pos < 0: return -1 if f <= -stay else 0
        return 0
    return rule


def r_gate(vr, g, thr=T0):
    return lambda pos, f, i: (1 if f > thr else (-1 if f < -thr else 0)) if vr[i] > g else 0


def r_size(big, thr=T0):
    def rule(pos, f, i):
        a = abs(f)
        if a <= thr: return 0
        return (2 if a > big * thr else 1) * (1 if f > 0 else -1)
    return rule


def r_agree(f2, thr=T0):
    def rule(pos, f, i):
        t = 1 if f > thr else (-1 if f < -thr else 0)
        return t if t != 0 and f2[i] == f2[i] and f2[i] * t > 0 else 0
    return rule


def sharpe(x):
    s = x.std(ddof=1); return float(x.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def boot_p(a, b, reps=10000, seed=20261115):
    if a.index.min() <= JOIN < a.index.max(): a, b = a[a.index > JOIN], b[b.index > JOIN]
    g =pd.DataFrame({'a': a.values, 'b': b.values, 'a2': a.values ** 2, 'b2': b.values ** 2, 'c': 1.0}, index=a.index).groupby(a.index.to_period('M')).sum().to_numpy()
    rng = np.random.default_rng(seed); S = np.zeros((reps, 5))
    for lo in range(0, reps, 1000):
        idx = rng.integers(0, len(g), (min(1000, reps - lo), len(g))); S[lo:lo + len(idx)] = g[idx].sum(axis=1)
    sh = lambda s1, s2, c: (s1 / c) / np.sqrt(np.maximum(s2 / c - (s1 / c) ** 2, 1e-30))
    return float((sh(S[:, 0], S[:, 2], S[:, 4]) <= sh(S[:, 1], S[:, 3], S[:, 4])).mean())


def describe(lab, x, trades):
    eq = x.cumsum(); dd = float((eq.cummax() - eq).max()); top = x.sort_values(ascending=False).head(20).sum(); tot = x.sum()
    A, B = x[x.index <= JOIN], x[x.index > JOIN]; eqb = B.cumsum(); ddb = float((eqb.cummax() - eqb).max())
    yrs = len(x) / 252
    say(f"{lab}: ${tot / yrs:,.0f}/yr | Sharpe {sharpe(x):.2f} (A {sharpe(A):.2f}, B {sharpe(B):.2f}) | {trades / yrs:,.0f} trades/yr, ${tot / max(trades, 1):.2f} net a trade | "
        f"best 20 days = {top / tot if tot else float('nan'):.0%} of the profit, without them ${tot - top:,.0f} | max DD ${dd:,.0f} | period B max DD ${ddb:,.0f}, worst day ${B.min():,.0f} "
        f"-> 50K fit {'yes' if ddb < 2500 and B.min() > -1100 else 'no'}")


def judge(name, desc, mk, es_mk, neighbours=()):
    """mk(book, ticks) -> (series, trades) for the trial; es_mk the same on the ES/MES book (or None = computed by the caller later)."""
    x, n = mk(NQ, 2.35); b, nb = BASE[2.35]; x35, _ = mk(NQ, 3.5); b35, _ = BASE[3.5]
    say(); say(f"{name} {desc}"); describe('   trial', x, n)
    A = lambda s: s[s.index <= JOIN]; B_ = lambda s: s[s.index > JOIN]
    p = boot_p(x, b); dif = (x - b).sort_values(ascending=False); kp = ~x.index.isin(dif.index[:20])
    own = x.sum() - x.sort_values(ascending=False).head(20).sum()
    g = {'G1': sharpe(A(x)) > sharpe(A(b)) and sharpe(B_(x)) > sharpe(B_(b)), 'G2': p < P_BAR,
         'G3': sharpe(A(x35)) > sharpe(A(b35)) and sharpe(B_(x35)) > sharpe(B_(b35)), 'G4': sharpe(x[kp]) > sharpe(b[kp]) and own > 0}
    es_txt = 'not run'
    if es_mk is not None:
        e, _ = es_mk(ES, 2.35); eb, _ = ES_BASE
        g['G5'] = sharpe(A(e)) >= sharpe(A(eb)) and sharpe(B_(e)) >= sharpe(B_(eb))
        es_txt = f"ES {sharpe(A(e)):.2f} vs {sharpe(A(eb)):.2f}, MES {sharpe(B_(e)):.2f} vs {sharpe(B_(eb)):.2f}"
    nb_txt = ''
    if neighbours:
        res = [(lab, sharpe(f(NQ, 2.35)[0])) for lab, f in neighbours]
        g['G6'] = all(s > sharpe(b) for _, s in res); nb_txt = " | neighbours: " + ", ".join(f"{lab} {s:.2f}" for lab, s in res)
    say(f"   vs base: P(Sharpe <= base) {p:.4f} (bar {P_BAR:.4f}) | at 3.5 ticks Sharpe A {sharpe(A(x35)):.2f} vs {sharpe(A(b35)):.2f}, B {sharpe(B_(x35)):.2f} vs {sharpe(B_(b35)):.2f} | "
        f"without the 20 days that help it most {sharpe(x[kp]):.2f} vs {sharpe(b[kp]):.2f} | second instrument: {es_txt}{nb_txt}")
    fails = [k for k, v in g.items() if not v]
    say(f"   VERDICT {name}: {'IMPROVEMENT (forward-test candidate)' if not fails else 'not an improvement (' + ', '.join(fails) + ')'}")
    return not fails, x


def setup():
    global NQ, ES, BASE, ES_BASE, F_NQ, F_ES
    NQ = Book(table('nq', 'mnqx')); ES = Book(table('es', 'mes'))
    F_NQ = base_pred(NQ.T, 'nq'); F_ES = base_pred(ES.T, 'es')
    BASE = {t: NQ.run(F_NQ, r_base(), t) for t in (2.35, 3.5)}; ES_BASE = ES.run(F_ES, r_base(), 2.35)


def rules():
    t0 = time.time(); setup()
    say(f"e5x_lab.py rules  {pd.Timestamp.now():%Y-%m-%d %H:%M}  (E5X_PLAN.txt)  $ per micro at today's price, 2.35 ticks + $0.61 a side, every session counted")
    b, nb = BASE[2.35]; yrs = len(b) / 252
    ok = abs(b.sum() / yrs - 7723) < 40 and abs(sharpe(b) - 1.51) < 0.01 and abs(nb - 10977) < 1
    say(f"audit base: ${b.sum() / yrs:,.0f}/yr, Sharpe {sharpe(b):.2f}, {nb:,.0f} trades (expected $7,723 / 1.51 / 10,977) -> {'PASS' if ok else 'FAIL'}")
    if not ok: say("audit FAILED - nothing graded"); return
    describe('BASE E5n', b, nb); eb, en = ES_BASE; describe('BASE on ES/MES', eb, en)
    mk = lambda rule_nq, rule_es: (lambda bk, t: bk.run(F_NQ, rule_nq, t), lambda bk, t: bk.run(F_ES, rule_es, t))
    surv = []
    a, e = mk(r_base(1.5 * T0), r_base(1.5 * T0))
    if judge('X1', 'stronger forecasts only: threshold 1.5 x T', a, e, [(f"{m} x T", (lambda m: lambda bk, t: bk.run(F_NQ, r_base(m * T0), t))(m)) for m in (1.25, 2.0)])[0]: surv.append('X1')
    a, e = mk(r_hold(0.0), r_hold(0.0))
    if judge('X2', 'hold until the forecast turns (stay while forecast >= 0)', a, e,
             [(f"stay >= {s:+.1f} T", (lambda s: lambda bk, t: bk.run(F_NQ, r_hold(s * T0), t))(s)) for s in (-0.5, 0.5)])[0]: surv.append('X2')
    vn, ve = NQ.volratio(), ES.volratio()
    a, e = mk(r_gate(vn, 1.0), r_gate(ve, 1.0))
    if judge('X3', 'high-volatility gate (30-min volatility above its 60-session median)', a, e,
             [(f"ratio > {gg}", (lambda gg: lambda bk, t: bk.run(F_NQ, r_gate(vn, gg), t))(gg)) for gg in (0.8, 1.25)])[0]: surv.append('X3')
    a, e = mk(r_size(2.0), r_size(2.0))
    if judge('X4', 'size by strength (2 units above 2 x T)', a, e,
             [(f"2 units > {m} T", (lambda m: lambda bk, t: bk.run(F_NQ, r_size(m), t))(m)) for m in (1.5, 3.0)])[0]: surv.append('X4')
    book(b)
    say(); say("RULE TRIALS THAT IMPROVE: " + (", ".join(surv) if surv else "none")); say(f"run {time.time() - t0:.0f} s")


def book(b):
    say(); say("X9 BOOK: E5n + band leg, 1 micro each (information)")
    x3 = pd.read_csv(os.path.join(CACHE, 'x3_daily.csv'), index_col=0, parse_dates=True).r_close * U
    c = pd.concat([b.rename('e'), x3.rename('n')], axis=1, join='inner'); c['book'] = c.e + c.n
    for lab, s in (('E5n alone      ', c.e), ('band leg alone', c.n), ('book 1 + 1     ', c.book)):
        eq = s.cumsum(); dd = float((eq.cummax() - eq).max()); B = s[s.index > JOIN]; eqb = B.cumsum(); ddb = float((eqb.cummax() - eqb).max())
        say(f"   {lab}: ${s.mean() * 252:,.0f}/yr | Sharpe {sharpe(s):.2f} | max DD ${dd:,.0f} | 2019-26: Sharpe {sharpe(B):.2f}, max DD ${ddb:,.0f}, worst day ${B.min():,.0f}")
    say(f"   daily correlation {c.e.corr(c.n):+.2f}; months both lose {((c.e.groupby(c.index.to_period('M')).sum() < 0) & (c.n.groupby(c.index.to_period('M')).sum() < 0)).mean():.0%}")
    try:
        sys.path.insert(0, TR.RESEARCH)
        import biglever as BL, bulenox_routes as BR
        a = BR.SIZES['50K']; Bm = c[c.index > JOIN]
        for lab, s in (('E5n alone', Bm.e), ('band leg alone', Bm.n), ('book 1 + 1', Bm.book)):
            u = s.values; nt = (u != 0).astype(float)
            e = BL.eval_stats(u, u, nt, 1, BL.SEED_REPORT, a['dd'], a['dll'], a['target'], a['qual'], resets=True)
            f = BL.funded_stats(BL.master_qm, u, u, nt, a, 1, BL.SEED_REPORT)
            say(f"   Bulenox 50K, {lab}, close marks (optimistic): evaluation pass {e['p'] * 100:.1f}% a try | funded 2-yr payouts ${f['pay']:,.0f}, open after 1 yr {f['open1'] * 100:.0f}%")
    except Exception as ex:
        say(f"   Bulenox model not run: {type(ex).__name__}: {ex}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'data': build()
    elif cmd == 'rules':
        rules()
        with open(OUT, 'w', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")
    else: print(__doc__)
