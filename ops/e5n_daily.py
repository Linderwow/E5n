r"""E5n nightly job - its own Windows task ("E5n nightly"); it never touches another strategy's files.
Scores the frozen models and the band leg on MNQ one-minute bars (the strategy's own bar log, the recorder's files, the minute store),
runs the C# engine on the same bars as a cross-check, reads the strategy's own logs (decisions, status, Strategy Analyzer run) and
appends the forward ledger.
    python e5n_daily.py          one run; writes ops\data\e5n_ledger.csv, e5n_daily_status.json, e5n_daily_log.txt
"""
import os, sys, glob, json, hashlib, subprocess, datetime as dt
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_paths as PATHS
import e5n_features as F
import e5n_calendar as CAL
import e5n_band as B
import e5n_map as MAP
import e5n_health as H
import e5n_stocks as STK
import e5n_checkpoints as CK
import e5n_flow as FL
import e5n_refit as RF
import e5n_target as TGT
from e5n_model import ModelSet

ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, 'nt8', 'e5n', 'e5n_models.txt'); MODELS15 = os.path.join(ROOT, 'nt8', 'e5n', 'e5n_models_15min.txt')
EXE = os.path.join(HERE, 'parity_test', 'E5nParity.exe')
DATA = os.path.join(HERE, 'data')
LEDGER = os.path.join(DATA, 'e5n_ledger.csv'); TRADE_MAP = os.path.join(DATA, 'e5n_trade_map.html'); STATUS = os.path.join(DATA, 'e5n_daily_status.json'); LOG = os.path.join(DATA, 'e5n_daily_log.txt')
NT = os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8')
NT_E5N = os.path.join(NT, 'e5n'); OWN_BARS = os.path.join(NT_E5N, 'bars')
NT_MIN = os.path.join(PATHS.NT_MINUTE, 'MNQ')
REC = 'D:/nt_recorder'
TRAIN_END = pd.Timestamp('2026-10-15')
WARMUP = 20
COLS = ['date', 'base_pts', 'base_trips', 'base_usd', 'model_pts', 'model_trips', 'model_usd', 'band_pts', 'band_trips', 'band_usd', 'book_usd', 'model', 'csharp_same', 'recorded']
NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def usd(pts, trips):
    return round(pts * 2.0 - trips * 2 * F.SIDE_USD, 2)


def all_bars():
    parts = []
    for d in sorted(glob.glob(os.path.join(OWN_BARS, 'MNQ_*'))):
        code = os.path.basename(d)[4:]
        for f in sorted(glob.glob(os.path.join(d, '*.csv'))):
            if os.path.getsize(f) == 0: continue
            x = pd.read_csv(f)
            if x.empty: continue
            parts.append(pd.DataFrame({'end': pd.to_datetime(x.time_et), 'o': x.open, 'h': x.high, 'l': x.low, 'c': x.close, 'v': x.volume.astype(float), 'code': code}))
    for f in sorted(glob.glob(os.path.join(NT_MIN, '*.parquet'))):
        code = os.path.basename(f)[:-8]; x = pd.read_parquet(f)
        t = pd.DatetimeIndex(x.time).tz_convert('America/New_York').tz_localize(None)
        parts.append(pd.DataFrame({'end': t, 'o': x.open.values, 'h': x.high.values, 'l': x.low.values, 'c': x.close.values, 'v': x.volume.values.astype(float), 'code': code}))
    for d in sorted(glob.glob(os.path.join(REC, 'MNQ_*'))):
        code = os.path.basename(d)[4:]
        for f in sorted(glob.glob(os.path.join(d, '*_bars1minute.csv'))):
            if os.path.getsize(f) == 0: continue
            x = pd.read_csv(f)
            if x.empty: continue
            parts.append(pd.DataFrame({'end': pd.to_datetime(x.time_et), 'o': x.open, 'h': x.high, 'l': x.low, 'c': x.close, 'v': x.volume.astype(float), 'code': code}))
    cols = ['end', 'o', 'h', 'l', 'c', 'v', 'code', 'date', 'pos', 'tday']
    if not parts: return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    a = pd.concat(parts).drop_duplicates(['end', 'code']).sort_values(['end', 'code']).reset_index(drop=True)
    start = a.end - pd.Timedelta(minutes=1); a['date'] = start.dt.normalize(); a['pos'] = start.dt.hour * 60 + start.dt.minute - 570
    a['tday'] = (start + pd.Timedelta(hours=6)).dt.normalize()
    r = a[(a.pos >= 0) & (a.pos < 390)]
    front = r.groupby(['tday', 'code']).v.sum().reset_index().sort_values(['tday', 'v']).drop_duplicates('tday', keep='last').set_index('tday').code
    front = front.reindex(sorted(a.tday.unique())).bfill().ffill()
    return a, a[a.code.values == a.tday.map(front).values].sort_values('end').reset_index(drop=True)


def minute_bars(keep=None):
    if keep is None: keep = all_bars()[1]
    return keep[(keep.pos >= 0) & (keep.pos < 390)].reset_index(drop=True)


def continuous(a, keep):
    codes = keep.groupby('tday').code.first(); days = list(codes.index); offs = []
    nt = None; p = os.path.join(OWN_BARS, 'rollovers_MNQ.csv')
    if os.path.exists(p):
        try: nt = pd.read_csv(p, parse_dates=['date'])
        except Exception: nt = None
    for i in range(1, len(days)):
        if codes.iat[i] == codes.iat[i - 1]: continue
        old, new = codes.iat[i - 1], codes.iat[i]; x = a[(a.tday == days[i - 1]) & a.code.isin([old, new])]
        pv = x.pivot_table(index='end', columns='code', values='c')
        off = 0.0
        if old in pv.columns and new in pv.columns:
            pv = pv[[old, new]].dropna()
            if len(pv): off = float(pv[new].iloc[-1] - pv[old].iloc[-1])
        if nt is not None and len(nt):
            near = nt[(nt.date - days[i]).abs() <= pd.Timedelta(days=7)]
            if len(near): off = float(near.offset.iloc[0])
        offs.append((days[i], off))
    adj = np.zeros(len(keep)); td = keep.tday.values
    for day, off in offs: adj[td < np.datetime64(day)] += off
    out = keep[['end', 'v']].copy()
    for k in 'ohlc': out[k] = keep[k].values + adj
    return out, offs


class Sessions:
    def __init__(self, bars):
        rows = []; O = []; H = []; L = []; C = []; V = []; P = []
        for d, g in bars.groupby('date', sort=True):
            if not CAL.full_session(d.date()) or g.pos.iat[0] != 0 or g.pos.iat[-1] < 385 or len(g) < 201: continue
            arr = {k: np.full(390, np.nan) for k in 'ohlcv'}
            for k in 'ohlcv': arr[k][g.pos.values] = g[k].values
            c = pd.Series(arr['c']).ffill().values; o = np.where(np.isnan(arr['o']), c, arr['o'])
            h = np.where(np.isnan(arr['h']), c, arr['h']); l = np.where(np.isnan(arr['l']), c, arr['l']); v = np.nan_to_num(arr['v'])
            rows.append(dict(date=pd.Timestamp(d), iid=int(g.code.iat[0].replace('-', '')))); O.append(o); H.append(h); L.append(l); C.append(c); V.append(v)
            P.append(~np.isnan(arr['c']))
        self.D = pd.DataFrame(rows); self.O, self.H, self.L, self.C, self.V = (np.array(x) for x in (O, H, L, C, V)); self.present = np.array(P)


def python_days(S, M, M15=None):
    T = F.features(S)
    T['mend'] = np.array([CAL.month_end_flag(d.date()) for d in S.D.date], float)[T.k.values]
    T = T[(T.k >= WARMUP) & S.present[T.k.values, T.dec.values - 1]].reset_index(drop=True)
    X_ = T[F.FEATS].values; pred, names = M.predict(T.date.values, X_); T['pred'] = pred; T['model'] = names
    T['chk'] = M15.predict(T.date.values, X_)[0] if M15 is not None else np.nan
    out = {}; dec = []
    for k, g in T.groupby('k', sort=True):
        pos = [0, 0]; pts = [0.0, 0.0]; sides = [0, 0]; last = [None, None]; trail = ([], []); n = 0
        for r in g.itertuples():
            if not np.isfinite(r.pred): continue
            n += 1; base = 1 if r.pred > M.threshold else (-1 if r.pred < -M.threshold else 0)
            fin = base if base != 0 and np.isfinite(r.chk) and r.chk * base > 0 else 0
            dec.append((r.date, int(r.dec), base, fin, float(r.pred), float(r.chk), float(S.C[int(r.k), int(r.dec) - 1])))
            for j, new in enumerate((base, fin)):
                if pos[j] != 0: pts[j] += pos[j] * (r.entry - last[j]); last[j] = r.entry
                if new != pos[j]:
                    sides[j] += abs(new - pos[j]); pos[j] = new; last[j] = r.entry
                    trail[j].append(f"{int(r.dec)}:{new}")
        for j in (0, 1):
            if pos[j] != 0: pts[j] += pos[j] * (g.exitp.iat[-1] - last[j]); sides[j] += 1
        if n: out[pd.Timestamp(g.date.iat[0]).date()] = dict(gross_pts=round(pts[0], 2), round_trips=sides[0] / 2, chk_pts=round(pts[1], 2), chk_trips=sides[1] / 2,
                                                              decisions=n, model=g.model.iat[-1], trail=' '.join(trail[0]), chk_trail=' '.join(trail[1]))
    return out, pd.DataFrame(dec, columns=['date', 'minute', 'base', 'target', 'forecast', 'check', 'close'])


def band_days(cont):
    marks = []; R = B.run(B.five(cont), live=True, warmup=WARMUP, bars=marks)
    out = {d.date(): dict(pts=round(float(r.points), 2), trips=float(r.sides) / 2) for d, r in R.iterrows()}
    tg = pd.concat(marks, ignore_index=True) if marks else pd.DataFrame(columns=['time', 'target'])
    return out, tg


def csharp_days(bars):
    if not os.path.exists(EXE): return None, 'C# test program not built'
    tmp = os.path.join(DATA, 'bars_for_engine.csv')
    b = bars.assign(key=bars.code.str.replace('-', '').astype(int))
    with open(tmp, 'w', encoding='ascii', newline='\n') as fh:
        for r in b.itertuples(): fh.write(f"{r.end:%Y-%m-%d %H:%M},{r.o!r},{r.h!r},{r.l!r},{r.c!r},{r.v!r},{r.key}\n")
    try:
        p = subprocess.run([EXE, MODELS, tmp, 'bars', str(WARMUP)], capture_output=True, text=True, timeout=600, creationflags=NOWIN)
    finally:
        try: os.remove(tmp)
        except OSError: pass
    if p.returncode != 0: return None, f"C# engine failed: {(p.stderr or p.stdout)[-300:]}"
    out = {}
    for ln in p.stdout.strip().split('\n')[1:]:
        q = ln.split(',')
        if len(q) >= 6: out[dt.date.fromisoformat(q[0])] = dict(gross_pts=float(q[1]), round_trips=float(q[2]), decisions=int(q[3]), model=q[4], trail=q[5].strip())
    return out, None


def csharp_band(cont):
    if not os.path.exists(EXE): return None, 'C# test program not built'
    src = os.path.join(DATA, 'bars_for_band.csv'); dst = os.path.join(DATA, 'band_from_engine.csv')
    x = pd.DataFrame({'time': cont.end.dt.strftime('%Y-%m-%d %H:%M'), 'open': cont.o, 'high': cont.h, 'low': cont.l, 'close': cont.c, 'volume': cont.v})
    x.to_csv(src, index=False, float_format='%.10g')
    try:
        p = subprocess.run([EXE, 'band1', src, dst, str(WARMUP)], capture_output=True, text=True, timeout=900, creationflags=NOWIN)
        if p.returncode != 0: return None, f"C# band engine failed: {(p.stderr or p.stdout)[-300:]}"
        y = pd.read_csv(dst, parse_dates=['date'])
        return {d.date(): dict(pts=round(float(a), 2), trips=float(b) / 2) for d, a, b in zip(y.date, y.points, y.sides)}, None
    finally:
        for f in (src, dst):
            try: os.remove(f)
            except OSError: pass


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def strategy_logs(dec, band_marks):
    out = {}; logs = os.path.join(NT_E5N, 'logs')
    key = dec.assign(d=pd.to_datetime(dec.date).dt.strftime('%Y-%m-%d')).set_index(['d', 'minute'])[['base', 'target', 'close']].rename(
        columns={'base': 'py_base', 'target': 'py_target', 'close': 'py_close'})
    bm = band_marks.assign(d=band_marks.time.dt.strftime('%Y-%m-%d'), minute=band_marks.time.dt.hour * 60 + band_marks.time.dt.minute - 570).set_index(['d', 'minute']).target.rename('py_band') \
        if len(band_marks) else pd.Series(dtype=float, name='py_band')
    for f in sorted(glob.glob(os.path.join(logs, 'E5n_decisions_*.csv'))):
        tag = os.path.basename(f)[len('E5n_decisions_'):-4]
        if '_to_' in tag: continue
        try:
            if os.path.getsize(f) == 0: out[tag] = dict(decisions=0, compared=0, note='empty - no real-time decision yet'); continue
            x = pd.read_csv(f, on_bad_lines='skip'); x = x[x.target.notna()].reset_index(drop=True); v2 = 'base_target' in x.columns
            j = x.assign(d=x.date.astype(str)).join(key, on=['d', 'minute'], how='inner')
            r = dict(decisions=int(len(x)), compared=int(len(j)), format='two legs' if v2 else 'first build',
                     same=int(((j.base_target if v2 else j.target).astype(int) == j.py_base).sum()),
                     prices_same=int((abs(j.close.astype(float) - j.py_close) < 0.011).sum()),
                     first=str(x.date.min()) if len(x) else '', last=str(x.date.max()) if len(x) else '',
                     file_time=dt.datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M'))
            if v2:
                c = j[j.check.notna()]; r['check_compared'] = int(len(c)); r['check_same'] = int((c.target.astype(int) == c.py_target).sum())
                b = x[x.band.notna()].assign(d=lambda q: q.date.astype(str)).join(bm, on=['d', 'minute'], how='inner')
                r['band_compared'] = int(len(b)); r['band_same'] = int((b.band.astype(int) == b.py_band.astype(int)).sum())
                nxt = x.position.shift(-1); ok = (x.date == x.date.shift(-1)) & (x.minute < 385)
                r['orders_checked'] = int(ok.sum()); r['orders_followed'] = int((nxt[ok] == x.net[ok]).sum())
            out[tag] = r
        except Exception as e:
            out[tag] = dict(error=f"{type(e).__name__}: {e}")
    st = {}
    for f in sorted(glob.glob(os.path.join(logs, 'E5n_status_*.json'))):
        try: st[os.path.basename(f)[len('E5n_status_'):-5]] = json.load(open(f, encoding='utf-8'))
        except Exception as e: st[os.path.basename(f)] = dict(error=str(e))
    return out, st


def dd(x):
    eq = x.cumsum(); return round(float((np.maximum(eq.cummax(), 0) - eq).max()), 2) if len(x) else 0.0


def main():
    os.makedirs(DATA, exist_ok=True); now = dt.datetime.now(); problems = []
    try: refit = RF.promote() or {}
    except Exception as e: refit = {}; problems.append(f"recording the installed yearly refit failed: {type(e).__name__}: {e}")
    M = ModelSet(MODELS); assert M.features == F.FEATS and abs(M.threshold - F.THRESHOLD) < 1e-18
    M15 = ModelSet(MODELS15) if os.path.exists(MODELS15) else None
    if M15 is None: problems.append('15-minute check model file missing in the repository')
    a, keep = all_bars(); bars = minute_bars(keep); S = Sessions(bars)
    py, dec = python_days(S, M, M15)
    cont, offs = continuous(a, keep); bd, marks = band_days(cont)
    cs, err = csharp_days(bars)
    if err: problems.append(err)
    cmp_days = same = 0; worst = 0.0; diff = []
    if cs is not None:
        for d in sorted(set(py) | set(cs)):
            p_, c_ = py.get(d), cs.get(d)
            if p_ is None or c_ is None:
                if (p_ or c_)['round_trips'] > 0: diff.append(str(d))
                continue
            cmp_days += 1; g = abs(p_['gross_pts'] - c_['gross_pts']); worst = max(worst, g)
            if p_['trail'] == c_['trail'] and g < 0.011: same += 1
            else: diff.append(str(d))
        if diff: problems.append(f"C# engine and Python differ on {len(diff)} day(s): {', '.join(diff[:6])}")
    cb, err = csharp_band(cont)
    if err: problems.append(err)
    bcmp = bsame = 0; bdiff = []
    if cb is not None:
        for d in sorted(set(bd) | set(cb)):
            p_, c_ = bd.get(d), cb.get(d)
            if p_ is None or c_ is None:
                if (p_ or c_)['trips'] > 0: bdiff.append(str(d))
                continue
            bcmp += 1
            if abs(p_['pts'] - c_['pts']) < 0.011 and p_['trips'] == c_['trips']: bsame += 1
            else: bdiff.append(str(d))
        if bdiff: problems.append(f"C# band leg and Python differ on {len(bdiff)} day(s): {', '.join(bdiff[:6])}")
    old = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    have = set(pd.to_datetime(old.date).dt.date)
    new = [d for d in sorted(py) if pd.Timestamp(d) > TRAIN_END and d not in have]
    if new:
        rows = []
        for d in new:
            b = bd.get(d, dict(pts=0.0, trips=0.0)); mu = usd(py[d]['chk_pts'], py[d]['chk_trips']); bu = usd(b['pts'], b['trips'])
            rows.append(dict(date=d, base_pts=py[d]['gross_pts'], base_trips=py[d]['round_trips'], base_usd=usd(py[d]['gross_pts'], py[d]['round_trips']),
                             model_pts=py[d]['chk_pts'], model_trips=py[d]['chk_trips'], model_usd=mu, band_pts=b['pts'], band_trips=b['trips'], band_usd=bu,
                             book_usd=round(mu + bu, 2), model=py[d]['model'],
                             csharp_same=('' if cs is None or d not in cs else bool(cs[d]['trail'] == py[d]['trail'] and (cb is None or d not in cb or d not in bd or cb[d] == bd[d]))),
                             recorded=now.strftime('%Y-%m-%d %H:%M')))
        pd.DataFrame(rows)[COLS].to_csv(LEDGER, mode='a', header=not os.path.exists(LEDGER), index=False)
    L = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame(columns=COLS)
    x = L.book_usd.astype(float) if len(L) else pd.Series(dtype=float)
    t = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else 0.0
    legs = {k: dict(total_usd=round(float(L[k + '_usd'].astype(float).sum()), 2), max_dd=dd(L[k + '_usd'].astype(float))) for k in ('base', 'model', 'band', 'book')} if len(L) else {}
    shadow_txt = ''; shadow = {}
    try:
        srows, snote = STK.shadow(S, M, M15, bd, WARMUP, now)
        if srows is None: shadow = dict(error=snote)
        else:
            SL, sadded = STK.append_ledger(srows, now); shadow = dict(STK.summary(SL), added=sadded, note=snote)
            if shadow.get('sessions'):
                shadow_txt = (f" | stock shadow {shadow['sessions']} sessions: model leg deployed ${shadow['base_usd']:+,.0f}, with stocks (all exchanges) ${shadow['sip_usd']:+,.0f}, "
                              f"(free exchange) ${shadow['iex_usd']:+,.0f}")
    except Exception as e: shadow = dict(error=f"{type(e).__name__}: {e}")
    try:
        TS, tadded = TGT.append_ledger(TGT.shadow(cont[cont.end > TRAIN_END], marks[marks.time > TRAIN_END]), TRAIN_END, now); target = dict(TGT.summary(TS), added=tadded)
    except Exception as e: target = dict(error=f"{type(e).__name__}: {e}")
    try:
        BS, badded = TGT.append_ledger(TGT.bracket(cont[cont.end > TRAIN_END], marks[marks.time > TRAIN_END]), TRAIN_END, now, TGT.LEDGER_BR)
        bracket_sh = dict(TGT.summary(BS), added=badded)
    except Exception as e: bracket_sh = dict(error=f"{type(e).__name__}: {e}")
    restated = []
    for r_ in L.itertuples() if len(L) else []:
        d_ = pd.Timestamp(r_.date).date()
        if d_ in py:
            b_ = bd.get(d_, dict(pts=0.0, trips=0.0)); now_book = usd(py[d_]['chk_pts'], py[d_]['chk_trips']) + usd(b_['pts'], b_['trips'])
            if abs(now_book - float(r_.book_usd)) > 1.0: restated.append((str(d_), float(r_.book_usd), round(now_book, 2)))
    logs, st = strategy_logs(dec, marks)
    live_st = [v for k, v in st.items() if 'Backtest' not in k and isinstance(v, dict) and 'contracts' in v]
    model_off = bool(live_st) and all(v.get('contracts') == 0 for v in live_st)
    ck_ledger = CK.ledger(L, model_off=model_off); ck_shadow = CK.shadow(shadow if shadow.get('sessions') else None)
    try:
        Tf = F.features(S); Tf['mend'] = np.array([CAL.month_end_flag(d.date()) for d in S.D.date], float)[Tf.k.values]
        Tf = Tf[(Tf.k >= WARMUP) & S.present[Tf.k.values, Tf.dec.values - 1] & (Tf.date > TRAIN_END)].reset_index(drop=True)
        ff5, _ = M.predict(Tf.date.values, Tf[F.FEATS].values) if len(Tf) else (np.zeros(0), None)
        flow = FL.nightly(Tf, ff5, now)
    except Exception as e: flow = dict(error=f"{type(e).__name__}: {e}")
    try: health = H.check(now, bars, os.path.join(NT_E5N, 'logs'), OWN_BARS, REC, system=True)
    except Exception as e: health = dict(problems=[f"live checks failed: {type(e).__name__}: {e}"], warnings=[], info={})
    problems += health['problems']
    if ck_ledger.get('status') == 'STOP AND REVIEW': problems += [f for f in ck_ledger['flags'] if f.startswith(('BOOK BELOW', 'MODEL LEG BELOW'))]
    health['warnings'] += [f for f in ck_ledger.get('flags', []) if not f.startswith(('BOOK BELOW', 'MODEL LEG BELOW'))]
    if str(ck_shadow.get('status', '')).startswith(('STOPPED', 'LIVE CANDIDATE', 'NOT ADOPTED')): health['warnings'].append(f"stock shadow test: {ck_shadow['status']}")
    if flow.get('error'): health['warnings'].append(f"order-flow pilot not updated: {flow['error']}")
    elif flow.get('buy_history'): health['warnings'].append(f"order-flow pilot: {flow['verdict']}")
    big = [x_ for x_ in restated if abs(x_[2] - x_[1]) > 50]
    if big: health['warnings'].append(f"ledger day(s) recorded with different bars than today's: " + ', '.join(f"{d} ${a:+,.0f} -> ${b:+,.0f}" for d, a, b in big[:5]))
    summary = (f"Forward test: {ck_ledger.get('sessions', 0)} sessions, book ${ck_ledger.get('book_usd', 0):+,.0f} (fair median ${ck_ledger.get('book_median', 0):+,.0f}, "
               f"stop line ${ck_ledger.get('book_p5', 0):+,.0f}) - {ck_ledger.get('status')}; {ck_ledger.get('checkpoint', '')}; {ck_ledger.get('dry_spell', '')}. "
               + (f"{ck_ledger['model_note'][0].upper()}{ck_ledger['model_note'][1:]}. " if ck_ledger.get('model_note') else '') + f"Stock shadow: {ck_shadow.get('status')}. "
               f"Order flow: {flow.get('verdict', flow.get('error', ''))}. Band profit target (shadow, not traded): {target.get('status', target.get('error', 'no sessions yet'))}. "
               f"Band 2-ATR stop + 6-ATR target (shadow, not traded): {bracket_sh.get('status', bracket_sh.get('error', 'no sessions yet'))}.")
    health['summary'] = summary
    if target.get('error'): health['warnings'].append(f"band target shadow not updated: {target['error']}")
    elif str(target.get('status', '')).startswith(('DROPPED', 'CANDIDATE', 'NOT ADOPTED')): health['warnings'].append(f"band target shadow: {target['status']}")
    rw = RF.warning()
    if rw: (problems if 'FAILED' in rw else health['warnings']).append(rw)
    if refit.get('state') == 'deployed and recorded' and refit.get('recorded', '')[:10] == now.strftime('%Y-%m-%d'):
        health['warnings'].append(f"yearly model refit to {refit['cutoff']} recorded in the repository (serving from {refit['serving_from']}) - push it")
    if shadow.get('error'): health['warnings'].append(f"stock shadow test not updated: {shadow['error']}")
    elif shadow.get('note'): health['warnings'].append(f"stock shadow test: {shadow['note']}")
    try: MAP.write(TRADE_MAP, bars, dec, marks, L, health=dict(health, problems=problems))
    except Exception as e: problems.append(f"trade map not written: {type(e).__name__}: {e}")
    for name, repo in (('e5n_models.txt', MODELS), ('e5n_models_15min.txt', MODELS15)):
        inst = os.path.join(NT_E5N, name)
        if os.path.exists(inst) and os.path.exists(repo) and sha(inst) != sha(repo): problems.append(f"{name} installed in NinjaTrader differs from the repository's")
    installed = {n: os.path.exists(os.path.join(NT_E5N, n)) for n in ('e5n_models.txt', 'e5n_models_15min.txt')}
    for tag, r in logs.items():
        if r.get('compared', 0) and r['same'] < r['compared'] * 0.97: problems.append(f"NinjaTrader decisions ({tag}) match Python on only {r['same']} of {r['compared']}")
        if r.get('compared', 0) and r['prices_same'] < r['compared'] * 0.90: problems.append(f"NinjaTrader prices ({tag}) equal the raw contract prices on only {r['prices_same']} of {r['compared']} decisions - back-adjustment not removed correctly")
        if r.get('check_compared', 0) and r['check_same'] < r['check_compared'] * 0.97: problems.append(f"NinjaTrader 15-minute check ({tag}) matches Python on only {r['check_same']} of {r['check_compared']}")
        if r.get('band_compared', 0) and r['band_same'] < r['band_compared'] * 0.97: problems.append(f"NinjaTrader band leg ({tag}) matches Python on only {r['band_same']} of {r['band_compared']}")
        if r.get('orders_checked', 0) and r['orders_followed'] < r['orders_checked'] * 0.995: problems.append(f"NinjaTrader position ({tag}) followed the net target on only {r['orders_followed']} of {r['orders_checked']} decisions")
    last_session = str(S.D.date.max().date()) if len(S.D) else ''
    own = sorted(glob.glob(os.path.join(OWN_BARS, 'MNQ_*', '*.csv')))
    status = dict(time=now.strftime('%Y-%m-%d %H:%M'), sessions_available=int(len(S.D)), last_session=last_session, ledger_sessions=int(len(L)),
                  ledger_max_dd=dd(x), ledger_total_usd=round(float(x.sum()), 2) if len(x) else 0.0, ledger_usd_per_day=round(float(x.mean()), 2) if len(x) else 0.0, ledger_t=round(t, 2),
                  legs=legs, added=[str(d) for d in new], csharp=dict(days_compared=cmp_days, identical=same, max_gross_diff_pts=round(worst, 2)) if cs is not None else None,
                  csharp_band=dict(days_compared=bcmp, identical=bsame) if cb is not None else None, roll_offsets=[[str(pd.Timestamp(d).date()), o] for d, o in offs[-4:]],
                  own_bar_files=len(own), own_bars_last=os.path.basename(own[-1])[:-4] if own else '', models_installed=installed,
                  strategy_decision_logs=logs, strategy_status=st, health=health, stock_shadow=shadow, checkpoints=dict(ledger=ck_ledger, shadow=ck_shadow, order_flow=flow),
                  ledger_restated=restated, target_shadow=target, bracket_shadow=bracket_sh, refit=RF.jload(RF.STATUS, {}), warnings=health['warnings'], problems=problems)
    text = json.dumps(status, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    tmp = STATUS + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh: fh.write(text)
    os.replace(tmp, STATUS)
    lg = " | ".join(f"{k} ${v['total_usd']:+,.0f} (max drawdown ${v['max_dd']:,.0f})" for k, v in legs.items())
    line = (f"{status['time']} | bars to {last_session} | ledger {status['ledger_sessions']} sessions, book ${status['ledger_total_usd']:+,.0f} "
            f"(${status['ledger_usd_per_day']:+,.0f}/day, t {status['ledger_t']:+.2f}), added {len(new)} | {lg}{shadow_txt} | C# vs Python "
            + (f"model {same}/{cmp_days}" if cs is not None else 'model not run') + (f", band {bsame}/{bcmp} days identical" if cb is not None else ', band not run') + " | "
            + ("; ".join(f"{k}: {v.get('same', '?')}/{v.get('compared', '?')} decisions as Python, prices equal on {v.get('prices_same', '?')}"
                         + (f", check {v['check_same']}/{v['check_compared']}, band {v['band_same']}/{v['band_compared']}, position followed the net target {v['orders_followed']}/{v['orders_checked']}" if 'check_same' in v else '')
                         for k, v in logs.items()) or "no strategy logs yet")
            + " | live: " + ("; ".join(f"{k}: {v.get('decisions_last_session', 0)}/{v.get('expected', '?')} decisions on {health['info'].get('last_full_session', '?')}, "
                                       f"{v.get('state', 'no status')}, roll in {v.get('roll', {}).get('days', '?')} days" for k, v in health['info'].items() if k in health['info'].get('live_accounts', [])) or 'not running')
            + f" | {summary}"
            + (" | WARNINGS: " + " / ".join(health['warnings']) if health['warnings'] else '')
            + (" | PROBLEMS: " + " / ".join(problems) if problems else " | ok"))
    with open(LOG, 'a', encoding='utf-8') as fh: fh.write(line + "\n")
    print(line)
    return 1 if problems else 0


def full_verify(now):
    try:
        subprocess.run([sys.executable.replace('pythonw.exe', 'python.exe'), '-B', os.path.join(HERE, 'e5n_verify.py')], cwd=HERE, capture_output=True, text=True,
                       timeout=3000, creationflags=NOWIN, encoding='utf-8', errors='replace')
        head = open(os.path.join(DATA, 'e5n_verify.txt'), encoding='utf-8').readline().strip()
        if not head.startswith(f"VERIFY {now:%Y-%m-%d}"): head = f"VERIFY {now:%Y-%m-%d %H:%M}: did not complete (no new result file)"
    except Exception as e: head = f"VERIFY {now:%Y-%m-%d %H:%M}: did not complete ({type(e).__name__}: {e})"
    try:
        with open(STATUS, encoding='utf-8') as fh: st = json.load(fh)
        st['verify'] = head; tmp = STATUS + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh: json.dump(st, fh, indent=1)
        os.replace(tmp, STATUS)
    except Exception: pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(__doc__); sys.exit(0 if sys.argv[1] in ('-h', '--help', '/?') else 2)
    try:
        rc = main(); full_verify(dt.datetime.now()); sys.exit(rc)
    except Exception as e:
        os.makedirs(DATA, exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as fh: fh.write(f"{dt.datetime.now():%Y-%m-%d %H:%M} | FAILED: {type(e).__name__}: {e}\n")
        raise
