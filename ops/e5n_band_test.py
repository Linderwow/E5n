r"""Band-leg checks.
    python e5n_band_test.py research     Python reference (batch rules) == the research engine on its two 5-minute files, every session
    python e5n_band_test.py csharp       C# engine == Python reference (live rules) on the same files, every session
"""
import os, sys, subprocess, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_paths as PATHS
import e5n_band as B
import e5n_train as TR

FILES = {'NQ': os.path.join(PATHS.BARS, 'nq_2010-2019_5m.csv'),
         'MNQ': os.path.join(PATHS.BARS, 'nt_analyzer_bars_to_2026-10-14.csv')}
EXE = os.path.join(HERE, 'parity_test', 'E5nParity.exe'); TMP = os.path.join(HERE, 'parity')


def load(nm):
    df = pd.read_csv(FILES[nm]); df['time'] = pd.to_datetime(df.time); return df.sort_values('time').reset_index(drop=True)


def research():
    sys.path.insert(0, TR.RESEARCH)
    import nav_lab as N, livestack as LS, weekend as WK
    ok = True
    for nm in ('NQ', 'MNQ'):
        t0 = time.time(); mk = N.market(nm); df = LS.full_bars(mk); Q = LS.per_day(mk, df, LS.ema_state(df.close.values))
        p, nt = WK.engine(mk, gate=4, lunch=True, ema=True, ch=3.0, conf=lambda k, i, new: bool(Q[k][i] == new), conf_mode='skip')[:2]
        ref = pd.DataFrame({'points': p, 'sides': nt * 2}, index=mk.dates).dropna()
        mine = B.run(load(nm)); j = ref.join(mine, rsuffix='_m', how='outer')
        dp = float(np.nanmax(np.abs(j.points - j.points_m))); ds = float(np.nanmax(np.abs(j.sides - j.sides_m))); miss = int(j.isna().any(axis=1).sum())
        good = dp == 0 and ds == 0 and miss == 0; ok &= good
        print(f"{nm}: {len(ref):,} sessions, {ref.sides.sum() / 2:,.0f} trades, {ref.points.sum():+,.2f} points | reference vs research: max |points diff| {dp:.1e}, "
              f"max |sides diff| {ds:.1e}, sessions missing on either side {miss} -> {'PASS' if good else 'FAIL'}; {time.time() - t0:.0f} s", flush=True)
        live = B.run(load(nm), live=True); k = ref.join(live, rsuffix='_l', how='left').fillna(0.0); diff = k[(k.points != k.points_l) | (k.sides != k.sides_l)]
        print(f"   live rules (no trades on early-close days, flat at the 16:00 bar): {len(diff)} sessions differ, research {diff.points.sum():+,.2f} points on them vs live "
              f"{diff.points_l.sum():+,.2f}; all sessions {ref.points.sum():+,.2f} vs {live.points.sum():+,.2f}", flush=True)
    print('BAND REFERENCE == RESEARCH: ' + ('PASS' if ok else 'FAIL')); return ok


def csharp():
    ok = True
    for nm in ('NQ', 'MNQ'):
        df = load(nm); src = os.path.join(TMP, f'band_{nm.lower()}_5m.csv'); out = os.path.join(TMP, f'band_{nm.lower()}_cs.csv')
        df.assign(time=df.time.dt.strftime('%Y-%m-%d %H:%M')).to_csv(src, index=False)
        r = subprocess.run([EXE, 'band5', src, out, str(B.LOOK)], capture_output=True, text=True)
        if r.returncode != 0: print(r.stdout[-2000:], r.stderr[-2000:]); return False
        cs = pd.read_csv(out, parse_dates=['date']).set_index('date'); py = B.run(df, live=True)
        j = py.join(cs, rsuffix='_c', how='outer'); miss = int(j.isna().any(axis=1).sum())
        dp = float(np.nanmax(np.abs(j.points - j.points_c))); ds = float(np.nanmax(np.abs(j.sides - j.sides_c)))
        good = miss == 0 and dp < 1e-6 and ds == 0; ok &= good
        print(f"{nm} 5-minute feed: {len(py):,} sessions | C# vs Python: max |points diff| {dp:.1e}, max |sides diff| {ds:.1e}, sessions on one side only {miss} -> "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    print('BAND C# == PYTHON: ' + ('PASS' if ok else 'FAIL')); return ok


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else ''
    if a == 'research': sys.exit(0 if research() else 1)
    elif a == 'csharp': sys.exit(0 if csharp() else 1)
    else: print(__doc__)
