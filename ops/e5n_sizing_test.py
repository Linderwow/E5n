r"""E5n account sizing: the strategy's rule (C# E5nSizer) and its Python twin against the simulation behind the funded-account figures.
    python e5n_sizing_test.py
"""
import os, sys, subprocess
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'research'))
import numpy as np
import e5n_account as A

EXE = os.path.join(HERE, 'parity_test', 'E5nParity.exe'); DATA = os.path.join(HERE, 'data'); NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
START = 50000.0
R = []


def rec(n, ok, d=''):
    R.append(bool(ok)); print(('PASS  ' if ok else 'FAIL  ') + n + (f'  [{d}]' if d else ''), flush=True)


def replay(B, X, W, NT, p, rule):
    n_p, n_d = X.shape; pay = np.zeros((n_p, n_d)); open_until = np.full(n_p, n_d); mic = np.zeros((n_p, n_d)); ob = np.full((n_p, n_d), np.nan); cb = np.full((n_p, n_d), np.nan)
    bal = np.zeros(n_p); hi = np.zeros(n_p); thr = np.full(n_p, -float(p['dd'])); locked = np.zeros(n_p, bool); alive = np.ones(n_p, bool)
    n_pay = np.zeros(n_p, int); days = np.zeros(n_p, int); ptot = np.zeros(n_p); pbest = np.zeros(n_p); wait = np.zeros(n_p, int)
    for t in range(n_d):
        q = rule(bal - thr); mic[:, t] = np.where(alive, q, 0); ob[alive, t] = bal[alive]
        x, w = B.day(X[:, t] * q, W[:, t] * q, np.where(locked, np.inf, float(p['dll'])))
        x = np.where(alive, x, 0.0); w = np.where(alive, w, 0.0)
        br = alive & (bal + w < thr); was = alive.copy(); open_until[br] = t; alive &= ~br; bal += x; cb[was, t] = bal[was]
        hi = np.where(alive, np.maximum(hi, bal), hi)
        up = alive & ~locked; thr = np.where(up, np.maximum(thr, hi - p['dd']), thr)
        lk = up & (thr >= 100); thr[lk] = 100.0; locked |= lk
        ww = alive & (wait > 0); wait[ww] -= 1; act = alive & ~ww
        days += (act & (NT[:, t] > 0)).astype(int); ptot = np.where(act, ptot + x, ptot); pbest = np.where(act, np.maximum(pbest, x), pbest)
        amt = np.minimum(np.where(n_pay < 3, float(p['cap']), 0.5 * bal - 1), bal - p['reserve'])
        el = act & (days >= 10) & (ptot > 0) & (pbest <= 0.4 * ptot) & (amt >= 1000)
        bal[el] -= amt[el]; pay[el, t] = 0.9 * amt[el]; n_pay[el] += 1; days[el] = 0; ptot[el] = 0.0; pbest[el] = 0.0; wait[el] = 3
    return pay, open_until, mic, ob, cb


def csharp(rows, mode, mx, target=3000.0):
    src, dst = os.path.join(DATA, '_sizer_in.csv'), os.path.join(DATA, '_sizer_out.csv')
    with open(src, 'w', encoding='ascii', newline='\n') as fh:
        for o, c in rows: fh.write(f"{float(o)!r},{float(c)!r}\n")
    p = subprocess.run([EXE, 'sizer', src, dst, mode, repr(START), '2500', '100', repr(target), '2500', str(mx), '0'], capture_output=True, text=True, creationflags=NOWIN)
    out = [ln.split(',') for ln in open(dst, encoding='ascii').read().strip().split('\n')[1:]] if p.returncode == 0 else []
    for f in (src, dst):
        try: os.remove(f)
        except OSError: pass
    return out


def main():
    import e5b_lab as L, e5c_fast as C, e5k_fair as K
    C.load(); M = C.M; sys.path.insert(0, L.TR.RESEARCH)
    import biglever as BL, bulenox_routes as BR, masterscale as MS
    a = BR.SIZES['50K']; clk0 = (M.dec % 5 == 0) & (M.dec >= 15); fair = C.path(M.a5, M.a15, 'set', clk0)
    u, w = K.day_arrays(fair + M.band); nt = (u != 0).astype(float); X, W, NT = BL.paths(BL.N_FUNDED, u, w, nt, 1, BL.SEED_REPORT)
    X, W, NT = X[:2000], W[:2000], NT[:2000]
    for mx in (2, 4):
        rule = lambda cu, mx=mx: np.clip(np.floor(cu / 2500), 1, mx)
        pay0, ou0, mic0 = MS.master_dyn(X, W, NT, a, rule); pay, ou, mic, ob, cb = replay(BL, X, W, NT, a, rule)
        rec(f'max {mx}: the replay reproduces the funded-account simulation exactly (payouts, closing days, sizes on 2,000 paths)',
            np.array_equal(pay0, pay) and np.array_equal(ou0, ou) and np.array_equal(mic0, mic))
        bad_py = 0; bad_cs = 0; days = 0; changes = 0
        for i in range(300):
            rows = [(START + ob[i, t], START + cb[i, t]) for t in range(X.shape[1]) if mic[i, t] > 0]
            s = A.Sizer('Funded', START, 2500, 100, 3000, 2500, mx); py = []
            for o, c in rows: s.close(o); py.append(s.units(o)); s.close(c)
            want = [int(mic[i, t]) for t in range(X.shape[1]) if mic[i, t] > 0]
            cs = [int(r[0]) for r in csharp(rows, 'Funded', mx)]
            bad_py += sum(1 for x_, y_ in zip(py, want) if x_ != y_) + abs(len(py) - len(want)); bad_cs += sum(1 for x_, y_ in zip(cs, want) if x_ != y_) + abs(len(cs) - len(want))
            days += len(want); changes += sum(1 for k in range(1, len(want)) if want[k] != want[k - 1])
        rec(f'max {mx}: Python twin = simulation on 300 paths ({days:,} account days, {changes:,} size changes)', bad_py == 0, f'{bad_py} differences')
        rec(f'max {mx}: C# E5nSizer (the strategy\'s rule) = simulation on the same days', bad_cs == 0, f'{bad_cs} differences')
    rows = [(START, START + 1200.0), (START + 1200.0, START + 2999.99), (START + 2999.99, START + 3000.0), (START + 3000.0, START + 2500.0)]
    cs = csharp(rows, 'Evaluation', 4)
    rec('evaluation: stands down exactly when the balance reaches start + target, size stays 1', [r[3] for r in cs] == ['0', '0', '1', '0'] and all(r[0] == '1' for r in cs))
    rows = [(START, START), (START + 2600.0, START + 2600.0), (START + 9000.0, START + 9000.0), (START + 30000.0, START + 30000.0), (START - 2000.0, START - 2000.0)]
    cs = csharp(rows, 'Funded', 4); s = A.Sizer('Funded', START, 2500, 100, 3000, 2500, 4); py = []
    for o, c in rows: s.close(o); py.append((s.units(o), s.threshold)); s.close(c)
    rec('funded edge cases: 1 unit at the start, threshold locks at +$100, cap at max units, back to 1 below the threshold', [int(r[0]) for r in cs] == [1, 1, 3, 4, 1] and [x_[0] for x_ in py] == [1, 1, 3, 4, 1]
        and float(cs[1][1]) == 100.0, ', '.join(f"{r[0]} units thr {float(r[1]):+,.0f}" for r in cs))
    cs = csharp(rows, 'Fixed', 4)
    rec('fixed mode: always 1 unit, never stands down', all(r[0] == '1' and r[3] == '0' for r in cs))
    import e5n_health as H
    live = ['2026-11-16 09:00:00', 'live', 'ACC', 'model-leg contracts 1, 15-minute check on, band-leg contracts 1, account mode Funded (start 50000, drawdown 2500, lock 100, cushion per unit 2500, max units 4, known high 0)']
    good = [live, ['2026-11-16 09:30:05', 'sizing', 'ACC', 'mode Funded, balance 55200.00, highest close 55400.00, threshold 50100.00, cushion 5100.00, units 2 (model 2, band 2)'],
            ['2026-11-17 09:30:05', 'sizing', 'ACC', 'mode Funded, balance 58300.00, highest close 58300.00, threshold 50100.00, cushion 8200.00, units 3 (model 3, band 3)']]
    st = dict(state='running', account_mode='Funded', units=3, balance=58300.0, threshold_balance=50100.0, cushion=8200.0, target_reached=False)
    P, W, acc = H.account_checks('ACC', st, good)
    rec('nightly: consistent funded sizing passes, a size change is reported', not P and any('changed from 2 to 3' in w for w in W) and acc['units'] == 3, '; '.join(P + W)[:160])
    bad = good[:2] + [['2026-11-17 09:30:05', 'sizing', 'ACC', 'mode Funded, balance 58300.00, highest close 58300.00, threshold 50100.00, cushion 8200.00, units 4 (model 4, band 4)']]
    P, W, _ = H.account_checks('ACC', st, bad)
    rec('nightly: a size that breaks the rule is a problem', any('does not follow the rule' in p for p in P), '; '.join(P)[:160])
    P, W, _ = H.account_checks('ACC', dict(st, account_mode='Evaluation', target_reached=True), [])
    rec('nightly: a reached evaluation target is reported', any('evaluation target reached' in w for w in W))
    P, W, _ = H.account_checks('ACC', dict(st, balance=None), good)
    rec('nightly: an unreadable account balance is a problem', any('not readable' in p for p in P))
    P, W, acc = H.account_checks('ACC', dict(state='running', account_mode='Fixed'), [])
    rec('nightly: fixed mode adds nothing', not P and not W and acc is None)
    live_b = [live[0], 'live', 'ACC', live[3][:-1] + ', band only until lock on)']
    pre = [live_b, ['2026-11-16 09:30:05', 'sizing', 'ACC', 'mode Funded, balance 51500.00, highest close 51500.00, threshold 49000.00, cushion 2500.00, units 1 (model 0, band 1), band leg only until the threshold locks']]
    P, W, _ = H.account_checks('ACC', dict(st, units=1, balance=51500.0, threshold_balance=49000.0, cushion=2500.0, band_only=True), pre)
    wrong = [live_b, ['2026-11-16 09:30:05', 'sizing', 'ACC', 'mode Funded, balance 51500.00, highest close 51500.00, threshold 49000.00, cushion 2500.00, units 1 (model 1, band 1)']]
    P2, _, _ = H.account_checks('ACC', st, wrong)
    post = [live_b] + good[1:]
    P3, _, _ = H.account_checks('ACC', st, post)
    sb = A.Sizer('Funded', START, 2500, 100, 3000, 2500, 4, 1500.0); sl = A.Sizer('Funded', START, 2500, 100, 3000, 2500, 4, 9000.0)
    rec('band only until the lock: 1 band micro and no model leg before the threshold locks, normal units after; the nightly accepts it and flags a model leg traded too early',
        not P and any('band leg only' in w for w in W) and any('does not follow the rule' in p for p in P2) and not P3
        and sb.funded_size(START + 1500, True) == (1, True) and sl.funded_size(START + 9000, True) == (3, False) and sb.funded_size(START + 1500, False) == (1, False),
        '; '.join(P + P2 + P3)[:200])
    import tempfile, pandas as pd
    tmp = tempfile.mkdtemp(prefix='e5n_fc_', dir=DATA); os.makedirs(os.path.join(tmp, 'MNQ_12-26'))
    with open(os.path.join(tmp, 'MNQ_12-26', '2026-11-16.csv'), 'w') as fh:
        fh.write("time_et,open,high,low,close,volume\n2026-11-16 10:00,100,101,99,100.00,10\n2026-11-16 10:30,200,201,199,200.00,10\n")
    fx = pd.DataFrame({'time_et': ['2026-11-16 10:00:00.312', '2026-11-16 10:30:00.200', '2026-11-16 10:30:00.250'], 'order': ['E5nLong', 'E5nSell', 'E5nShort'],
                       'action': ['Buy', 'Sell', 'SellShort'], 'quantity': [2, 1, 1], 'price': [100.50, 199.75, 200.25], 'position_after': [2, 1, 0]})
    fc = H.fill_costs(fx, tmp)
    rec('nightly: fill cost vs the decision bar close, per contract (buy +2 ticks x2, sell +1, short -1 tick better)', list(fc.ticks) == [2.0, 2.0, 1.0, -1.0], str(list(fc.ticks)))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    ok = all(R); print('ALL PASS' if ok else f'{R.count(False)} FAILED'); return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
