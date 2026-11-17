r"""E5A A3 (E5A_PLAN.txt): do funds that carry other markets (volatility futures, bonds, credit, the dollar, chip stocks, the 3x Nasdaq funds)
add to E5n's 5-minute forecast? 5-minute bars since 2016, the E5S method with placebo columns.
    python e5a_funds.py fetch      download the funds' bars -> cache\funds_alpaca_5m (resumable)
    python e5a_funds.py run        X1 and X2
"""
import os, sys, glob
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(HERE)); import e5n_paths as PATHS

FUNDS = PATHS.FUND_BARS; SYMS = ['VIXY', 'TLT', 'HYG', 'UUP', 'SMH', 'TQQQ', 'SQQQ']
M5 = np.timedelta64(5, 'm'); M15 = np.timedelta64(15, 'm')


def fetch():
    import e5s_alpaca as AL
    AL.OUT = FUNDS; AL.fetch(SYMS, workers=4)
    for s in SYMS:
        fs = sorted(glob.glob(os.path.join(FUNDS, f'{s}_*.parquet'))); n = sum(len(pd.read_parquet(f, columns=['time'])) for f in fs)
        print(f"   {s}: {len(fs)} year files, {n:,} bars")


def series(ES, sym, folder):
    keep = ES.STOCKS; ES.STOCKS = folder
    try: return ES.stock_series(sym)
    finally: ES.STOCKS = keep


def features():
    import e5s_lab as ES
    now, opn = ES.times(); S = {s: series(ES, s, FUNDS) for s in SYMS}; S['QQQ'] = series(ES, 'QQQ', ES.STOCKS)
    lr = lambda s, w: ES.asof(S[s]['t'], S[s]['lc'], now, 10) - ES.asof(S[s]['t'], S[s]['lc'], now - w, 10)
    ro = lambda s: ES.asof(S[s]['t'], S[s]['lc'], now, 10) - ES.asof(S[s]['t_open'], S[s]['lo'], opn, 1)
    dv = lambda s, since: ES.asof(S[s]['t'], S[s]['cum'], now, 10) - ES.asof(S[s]['t'], S[s]['cum'], since, 10)
    x1 = pd.DataFrame({'vx_r5': lr('VIXY', M5), 'vx_r15': lr('VIXY', M15), 'vx_ro': ro('VIXY'), 'tlt_r15': lr('TLT', M15), 'hyg_r15': lr('HYG', M15), 'uup_r15': lr('UUP', M15)})
    with np.errstate(invalid='ignore', divide='ignore'):
        l15 = dv('TQQQ', now - M15); s15 = dv('SQQQ', now - M15); lo_ = dv('TQQQ', opn); so_ = dv('SQQQ', opn)
        lev15 = np.where(l15 + s15 > 0, l15 / (l15 + s15), np.nan); levo = np.where(lo_ + so_ > 0, lo_ / (lo_ + so_), np.nan)
    q = pd.concat([pd.read_parquet(f, columns=['time', 'volume', 'trades']) for f in sorted(glob.glob(os.path.join(ES.STOCKS, 'QQQ_*.parquet')))], ignore_index=True)
    q['time'] = pd.to_datetime(q.time); q = q.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    ts = q.time.values.astype('datetime64[m]'); te = ts + M5; day = ts.astype('datetime64[D]'); first = np.r_[True, day[1:] != day[:-1]]
    cv = np.cumsum(q.volume.values.astype(float)); cn = np.cumsum(np.nan_to_num(q.trades.values.astype(float)))
    tp = np.r_[te, ts[first]]; o = np.argsort(tp, kind='stable'); tp = tp[o]; CV = np.r_[cv, (cv - q.volume.values)[first]][o]; CN = np.r_[cn, (cn - np.nan_to_num(q.trades.values.astype(float)))[first]][o]
    win = lambda C, since: ES.asof(tp, C, now, 10) - ES.asof(tp, C, since, 10)
    with np.errstate(invalid='ignore', divide='ignore'):
        a5 = win(CV, now - M5) / win(CN, now - M5); ao = win(CV, opn) / win(CN, opn); qsz = np.where(np.isfinite(a5) & np.isfinite(ao) & (ao > 0), a5 / ao, np.nan)
    x2 = pd.DataFrame({'smh_r5': lr('SMH', M5) - lr('QQQ', M5), 'smh_r15': lr('SMH', M15) - lr('QQQ', M15), 'smh_ro': ro('SMH') - ro('QQQ'), 'lev15': lev15, 'levo': levo, 'qsz5': qsz})
    return x1, x2


def run():
    import e5b_lab as L, e5s_lab as ES
    ES.OUT = os.path.join(HERE, 'E5A_RESULTS.txt'); L.setup(); x1, x2 = features()
    for nm, df in (('X1', x1), ('X2', x2)):
        c = df.notna().mean(); L.say(f"{nm} columns, share of all decisions with a value: " + ', '.join(f"{k} {v:.1%}" for k, v in c.items()))
    ES.run('A3 X1', 'funds: volatility futures, long Treasuries, high-yield bonds, the dollar - 6 columns', x1, first=2018, split=2022)
    ES.run('A3 X2', 'funds: chip stocks minus the index, the 3x long share of the 3x Nasdaq funds, QQQ average trade size - 6 columns', x2, first=2018, split=2022)


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'fetch': fetch()
    elif a == 'run': run()
    else: print(__doc__)
