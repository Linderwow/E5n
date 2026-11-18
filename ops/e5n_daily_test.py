import os, sys, shutil, tempfile
import datetime as dt
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_daily as D
import e5n_features as F
import e5n_calendar as CAL
import e5n_band as B
from e5n_model import ModelSet

FAIL = []


def check(name, ok, info=''):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{info}]" if not ok and info else '')); ok or FAIL.append(name)


def fake_log(folder, dec, shift=0.0, flip=0.0):
    os.makedirs(os.path.join(folder, 'logs'), exist_ok=True)
    x = dec.copy(); rng = np.random.default_rng(1)
    if flip: x.loc[rng.random(len(x)) < flip, 'base'] = 9
    out = pd.DataFrame({'date': pd.to_datetime(x.date).dt.strftime('%Y-%m-%d'), 'time_et': '10:00', 'minute': x.minute, 'forecast': x.forecast, 'target': x.base,
                        'model': 'wf', 'position': 0, 'close': x.close + shift, 'adjustment': 0.0, 'sessions': 30})
    out.to_csv(os.path.join(folder, 'logs', 'E5n_decisions_MNQ_Backtest.csv'), index=False)


def fake_log2(folder, dec, marks, bad_check=0.0, bad_band=0.0, bad_orders=0.0):
    os.makedirs(os.path.join(folder, 'logs'), exist_ok=True)
    x = dec.copy().reset_index(drop=True); rng = np.random.default_rng(2)
    m = marks.assign(d=marks.time.dt.normalize(), minute=marks.time.dt.hour * 60 + marks.time.dt.minute - 570)
    x = x.merge(m[['d', 'minute', 'target']].rename(columns={'d': 'date', 'target': 'band'}), on=['date', 'minute'], how='left'); x['band'] = x.band.fillna(0).astype(int)
    tgt = x.target.copy(); band = x.band.copy()
    if bad_check: tgt[rng.random(len(x)) < bad_check] = 9
    if bad_band: band[rng.random(len(x)) < bad_band] = 9
    net = tgt + band; pos = net.shift(1).fillna(0).astype(int); pos[x.date != x.date.shift(1)] = 0
    if bad_orders: pos[rng.random(len(x)) < bad_orders] = 7
    out = pd.DataFrame({'date': pd.to_datetime(x.date).dt.strftime('%Y-%m-%d'), 'time_et': '10:00', 'minute': x.minute, 'forecast': x.forecast, 'base_target': x.base,
                        'check': x.check.fillna(0.0), 'target': tgt, 'model': 'wf', 'band': band, 'net': net, 'position': pos, 'close': x.close, 'adjustment': 0.0, 'sessions': 30})
    out.to_csv(os.path.join(folder, 'logs', 'E5n_decisions_MNQ_Backtest.csv'), index=False)


check("T1 calendar: 2026-11-26 Thanksgiving and 11-27 are not full sessions, 11-25 is", not CAL.full_session(dt.date(2026, 11, 26))
      and not CAL.full_session(dt.date(2026, 11, 27)) and CAL.full_session(dt.date(2026, 11, 25)))
check("T2 calendar: Good Friday 2026-04-03 and Christmas are closed; month-end flag on 2026-11-25 and 11-30, not 11-24",
      not CAL.full_session(dt.date(2026, 4, 3)) and not CAL.full_session(dt.date(2026, 12, 25)) and CAL.month_end_flag(dt.date(2026, 11, 25))
      and CAL.month_end_flag(dt.date(2026, 11, 30)) and not CAL.month_end_flag(dt.date(2026, 11, 24)))
M = ModelSet(D.MODELS); M15 = ModelSet(D.MODELS15)
a, keep = D.all_bars(); a = a[a.end >= '2026-05-01']; keep = keep[keep.end >= '2026-05-01'].reset_index(drop=True)
_last = keep.end.max()
if _last.hour < 16:
    a = a[a.end < _last.normalize()]; keep = keep[keep.end < _last.normalize()].reset_index(drop=True)
bars = D.minute_bars(keep); bars = bars[bars.date >= '2026-06-01']; S = D.Sessions(bars)
py, dec = D.python_days(S, M, M15)
check("T3 Python decisions exist on the recent NinjaTrader bars", len(dec) > 3000 and len(py) > 60, f"{len(dec)} decisions, {len(py)} days")
cs, err = D.csharp_days(bars)
same = sum(1 for d in py if d in cs and cs[d]['trail'] == py[d]['trail'] and abs(cs[d]['gross_pts'] - py[d]['gross_pts']) < 0.011)
check("T4 the C# engine reproduces every Python day on those bars", err is None and same == len(py), f"{same} of {len(py)}; {err}")
cut = int(((dec.base != 0) & (dec.target == 0)).sum()); kept = int((dec.target != 0).sum())
check("T5 the 15-minute check only removes signals, never adds or flips one", cut > 0 and kept > 0 and bool(((dec.target == 0) | (dec.target == dec.base)).all()), f"kept {kept}, cut {cut}")
cont, offs = D.continuous(a, keep); bd, marks = D.band_days(cont); cb, err = D.csharp_band(cont)
bsame = sum(1 for d in bd if d in cb and cb[d] == bd[d])
check("T6 the C# band leg (one-minute feed) reproduces every Python day", err is None and len(bd) > 60 and bsame == len(bd) and set(cb) == set(bd), f"{bsame} of {len(bd)}; {err}")
holiday = [d for d in bd if B.is_short(d)]
check("T7 the band leg takes no trades on early-close days and is flat outside 09:35-16:00", not holiday and int(marks.target.abs().sum()) > 0
      and bool((marks[(marks.time.dt.hour * 60 + marks.time.dt.minute) >= 960].target == 0).all()), str(holiday))
tmp = tempfile.mkdtemp(); keep_dir = D.NT_E5N
try:
    D.NT_E5N = tmp
    fake_log(tmp, dec); logs, st = D.strategy_logs(dec, marks); r = logs['MNQ_Backtest']
    check("T8 a first-build Analyzer log: all decisions and prices match", r['compared'] == len(dec) and r['same'] == r['compared'] and r['prices_same'] == r['compared'], str(r))
    fake_log(tmp, dec, shift=296.5); logs, st = D.strategy_logs(dec, marks); r = logs['MNQ_Backtest']
    check("T9 a back-adjusted Analyzer log (prices +296.5): prices flagged", r['prices_same'] == 0 and r['same'] == r['compared'], str(r))
    fake_log(tmp, dec, flip=0.2); logs, st = D.strategy_logs(dec, marks); r = logs['MNQ_Backtest']
    check("T10 an Analyzer log with 20% different decisions: counted", 0.7 * r['compared'] < r['same'] < 0.9 * r['compared'], str(r))
    fake_log2(tmp, dec, marks); logs, st = D.strategy_logs(dec, marks); r = logs['MNQ_Backtest']
    check("T11 a correct two-leg log: model, check, band and position all match", r['same'] == r['compared'] and r['check_same'] == r['check_compared'] > 0
          and r['band_same'] == r['band_compared'] > 0 and r['orders_followed'] == r['orders_checked'] > 0, str(r))
    fake_log2(tmp, dec, marks, bad_check=0.2); r = D.strategy_logs(dec, marks)[0]['MNQ_Backtest']
    check("T12 a two-leg log with 20% wrong check results: counted", r['check_same'] < 0.9 * r['check_compared'], str(r))
    fake_log2(tmp, dec, marks, bad_band=0.2); r = D.strategy_logs(dec, marks)[0]['MNQ_Backtest']
    check("T13 a two-leg log with 20% wrong band targets: counted", r['band_same'] < 0.9 * r['band_compared'], str(r))
    fake_log2(tmp, dec, marks, bad_orders=0.1); r = D.strategy_logs(dec, marks)[0]['MNQ_Backtest']
    check("T14 a two-leg log where the position ignores the net target 10% of the time: counted", r['orders_followed'] < 0.95 * r['orders_checked'], str(r))
finally:
    D.NT_E5N = keep_dir; shutil.rmtree(tmp, ignore_errors=True)
import json as _json
import e5n_health as HL
day = dt.date(2026, 11, 13); fb = pd.DataFrame({'date': pd.Timestamp(day), 'pos': np.arange(390)})
now = dt.datetime(2026, 11, 13, 18, 45)


def live_env(rows=None, status=None, extra='', bars_own=None, bars_rec=None):
    t = tempfile.mkdtemp(); lg = os.path.join(t, 'logs'); own = os.path.join(t, 'bars'); rec = os.path.join(t, 'rec'); [os.makedirs(x) for x in (lg, os.path.join(own, 'MNQ_12-26'), os.path.join(rec, 'MNQ_12-26'))]
    head = "date,time_et,minute,forecast,base_target,check,target,model,band,net,position,close,adjustment,sessions"
    rows = HL.DECISIONS if rows is None else rows
    with open(os.path.join(lg, 'E5n_decisions_MNQ_Sim101.csv'), 'w') as fh:
        fh.write(head + "\n" + "".join(f"{day},{(570 + m) // 60:02d}:{(570 + m) % 60:02d},{m},0.0,0,0.0,0,final,0,0,0,28000,0,40\n" for m in rows) + extra)
    st = dict(state='running', updated_et=f"{now:%Y-%m-%d %H:%M:%S}", position='Flat', quantity=0, instrument='MNQ 12-26', connected=True, band_off=False, live_since='2026-11-12 10:00:00')
    st.update(status or {})
    with open(os.path.join(lg, 'E5n_status_MNQ_Sim101.json'), 'w') as fh: _json.dump(st, fh)
    for d_, b_ in ((own, bars_own), (rec, bars_rec)):
        if b_ is not None:
            nm = f'{day}.csv' if d_ == own else f'{day}_bars1minute.csv'
            b_.to_csv(os.path.join(d_, 'MNQ_12-26', nm), index=False)
    return t, lg, own, rec


def run(**kw):
    t, lg, own, rec = live_env(**kw)
    try: return HL.check(now, fb, lg, own, rec, free_gb={'C:\\': 500, 'D:\\': 500})
    finally: shutil.rmtree(t, ignore_errors=True)


h = run(); check("T15 a complete live day: no problems, 75/75 decisions", not h['problems'] and h['info']['MNQ_Sim101']['decisions_last_session'] == 75, str(h['problems']))
h = run(rows=[m for m in HL.DECISIONS if not 60 <= m <= 90])
check("T16 decisions missing 10:30-11:00 are found with their times", any('7 of 75 decisions missing' in x and '10:30-11:00' in x for x in h['problems']), str(h['problems']))
h = run(rows=[]); check("T17 a session with no decisions at all is reported", any('no decisions at all' in x for x in h['problems']), str(h['problems']))
t_, lg_, own_, rec_ = live_env(status=dict(updated_et='2026-11-12 17:30:00'))
try: h = HL.check(dt.datetime(2026, 11, 12, 18, 45), fb, lg_, own_, rec_, free_gb={'C:' + chr(92): 500, 'D:' + chr(92): 500})
finally: shutil.rmtree(t_, ignore_errors=True)
check("T18 a status not updated for 75 minutes on a Thursday evening (market open) is reported", any('no update for 75 minutes' in x for x in h['problems']), str(h['problems']))
hf = run(status=dict(updated_et='2026-11-13 16:58:00'))
check("T18b Friday 18:45 (market closed for the weekend): an older status is not an error", not any('no update' in x for x in hf['problems']), str(hf['problems']))
t_, lg_, own_, rec_ = live_env(status=dict(state='terminated', updated_et='2026-11-13 12:30:10'))
open(os.path.join(lg_, 'E5n_events_MNQ_Sim101.txt'), 'w').write('2026-11-13 12:30:10|terminated|Sim101|position Flat x0' + chr(10) + '2026-11-13 12:30:31|live|Sim101|restart' + chr(10))
try: hr = HL.check(now, fb, lg_, own_, rec_, free_gb={'C:' + chr(92): 500, 'D:' + chr(92): 500})
finally: shutil.rmtree(t_, ignore_errors=True)
check("T19b a restart (terminated, then live again) is not reported as stopped", not any('stopped' in x for x in hr['problems']) and any('restarted' in x for x in hr['info'].get('notes', [])), str(hr))
EV_AFTER_CLOSE = '2026-11-12 10:00:00|live|Sim101|start' + chr(10) + '2026-11-13 17:03:34|terminated|Sim101|position Flat x0' + chr(10) + '2026-11-13 17:03:45|live|Sim101|restart' + chr(10)
for name_, rows_, ev_, want_ in (
        ("T19c a restart after the close still checks the whole session (75 expected)", HL.DECISIONS, EV_AFTER_CLOSE, lambda h: h['info']['MNQ_Sim101']['expected'] == 75 and not h['problems']),
        ("T19d a decision lost in a short restart is a warning, not a problem", [m for m in HL.DECISIONS if m != 115],
         '2026-11-12 10:00:00|live|Sim101|start' + chr(10) + '2026-11-13 11:24:59|terminated|Sim101|position Short x1' + chr(10) + '2026-11-13 11:25:09|live|Sim101|restart' + chr(10),
         lambda h: not h['problems'] and any('skipped' in x and '11:25' in x for x in h['warnings'])),
        ("T19e decisions missing before an after-close restart are still a problem", [m for m in HL.DECISIONS if not 60 <= m <= 90], EV_AFTER_CLOSE,
         lambda h: any('7 of 75 decisions missing' in x for x in h['problems'])),
        ("T19f E5n off for hours mid-session: missing decisions are a problem", [m for m in HL.DECISIONS if m < 150],
         '2026-11-12 10:00:00|live|Sim101|start' + chr(10) + '2026-11-13 12:00:30|terminated|Sim101|position Flat x0' + chr(10) + '2026-11-13 17:03:45|live|Sim101|restart' + chr(10),
         lambda h: any('decisions missing' in x and '12:00-15:55' in x for x in h['problems']))):
    t_, lg_, own_, rec_ = live_env(rows=rows_, status=dict(live_since=ev_.strip().splitlines()[-1].split('|')[0]))
    open(os.path.join(lg_, 'E5n_events_MNQ_Sim101.txt'), 'w').write(ev_)
    try: hx = HL.check(now, fb, lg_, own_, rec_, free_gb={'C:' + chr(92): 500, 'D:' + chr(92): 500})
    finally: shutil.rmtree(t_, ignore_errors=True)
    check(name_, want_(hx), str(hx['problems']) + str(hx['warnings']))
h = run(status=dict(state='refused', halt_note='not started: test')); check("T19 a refused start is reported with its reason", any('refused' in x and 'test' in x for x in h['problems']), str(h['problems']))
h = run(extra="garbage line\n2026-11-13,09:45,15,0.0,0,0.0,0,final,0,0,0,28000,0,40\n")
check("T20 broken lines and duplicate decisions are reported", any('broken line' in x for x in h['problems']) and any('duplicate' in x for x in h['problems']), str(h['problems']))
tt = [f"{day} {(570 + i) // 60:02d}:{(570 + i) % 60:02d}" for i in range(1, 391)]
a_ = pd.DataFrame({'time_et': tt, 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 28000.0, 'volume': 5}); b_ = a_.copy(); b_.loc[::10, 'close'] = 28001.0
h = run(bars_own=a_, bars_rec=a_); check("T21 strategy bar log and recorder that agree pass", not h['problems'] and '100.0%' in h['info'].get('own_vs_recorder', ''), str(h))
h = run(bars_own=a_, bars_rec=b_); b2 = a_.copy(); b2.loc[::10, 'close'] = 28003.0
h = run(bars_own=a_, bars_rec=b2)
check("T22 strategy bar log and recorder 3 points apart on 10% of minutes: warning; 1 tick apart: nothing", any('differ by more than 2 ticks' in x for x in h['warnings']) and not h['problems'] and not any('2 ticks' in x for x in run(bars_own=a_, bars_rec=a_.assign(close=np.where(np.arange(len(a_)) % 10 == 0, 28000.25, 28000.0)))['warnings']), str(h))
r1 = HL.roll_info('MNQ 12-26', dt.date(2026, 12, 8)); r2 = HL.roll_info('MNQ 12-26', dt.date(2026, 12, 14)); r3 = HL.roll_info('MNQ 03-27', dt.date(2026, 12, 15))
check("T23 rollover dates: MNQ 12-26 rolls Mon 2026-12-14 (next 03-27), 03-27 rolls Mon 2027-03-15",
      str(r1['roll']) == '2026-12-14' and r1['days_to_roll'] == 6 and r1['next_contract'] == '03-27' and r2['days_to_roll'] == 0 and str(r3['roll']) == '2027-03-15', f"{r1} {r3}")
t_, lg_, own_, rec_ = live_env()
try:
    hw = HL.check(dt.datetime(2026, 12, 8, 18, 45), fb, lg_, own_, rec_, free_gb={'C:\\': 500, 'D:\\': 10})
    hp = HL.check(dt.datetime(2026, 12, 14, 18, 45), fb, lg_, own_, rec_, free_gb={'C:\\': 500, 'D:\\': 500})
finally: shutil.rmtree(t_, ignore_errors=True)
check("T24 a roll within 7 days is a warning, a passed roll and a nearly full disk are problems",
      any('rolls on Mon 2026-12-14' in x for x in hw['warnings']) and any('has rolled' in x for x in hp['problems']) and any('disk D: only 10 GB' in x for x in hw['problems']), f"{hw} {hp}")
import e5n_stocks as STK
Tq = F.features(S); Tq['mend'] = np.array([CAL.month_end_flag(d.date()) for d in S.D.date], float)[Tq.k.values]
Tq = Tq[(Tq.k >= D.WARMUP) & S.present[Tq.k.values, Tq.dec.values - 1]].reset_index(drop=True)
X0 = Tq[F.FEATS].values; q5, _ = M.predict(Tq.date.values, X0); q15, _ = M15.predict(Tq.date.values, X0); lg = STK.leg_days(Tq, q5, q15, M.threshold)
same = sum(1 for d in py if d in lg and abs(lg[d][0] - py[d]['chk_pts']) < 0.011 and lg[d][1] == py[d]['chk_trips'])
check("T25 the shadow's model-leg scorer reproduces the nightly ledger's model leg on every recent day", same == len(py) and len(py) > 60, f"{same} of {len(py)}")
tdir = tempfile.mkdtemp()
try:
    day0, day1 = pd.Timestamp('2026-11-12'), pd.Timestamp('2026-11-13')
    for sym, drift, vol in (('AAA', 1.0, 100.0), ('BBB', -1.0, 300.0)):
        rows = []
        for dday, base in ((day0, 100.0), (day1, 100.0)):
            for i in range(78):
                c = base + drift * i * 0.1; rows.append((dday + pd.Timedelta(minutes=570 + 5 * i), c - drift * 0.05, c + 0.1, c - 0.1, c, vol))
        pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume']).to_parquet(os.path.join(tdir, f'{sym}_2026.parquet'), index=False)
    Tt = pd.DataFrame({'date': [day1], 'dec': [60], 'r5': [0.0], 'r15': [0.0], 'ropen': [0.0]})
    ag = STK.aggregates(Tt, tdir, syms=['AAA', 'BBB'], min_n=2).iloc[0]
    ra = np.log((100 + 0.1 * 11) / (100 + 0.1 * 10)); rb = np.log((100 - 0.1 * 11) / (100 - 0.1 * 10))
    check("T26 stock columns on a two-stock example: half the stocks up, basket weighted by the previous day's dollar volume, share of volume rising",
          abs(ag.up5 - 0.5) < 1e-12 and abs(ag.bk_r5 - (ra * 1 + rb * 3) / 4 * 1.0) < 2e-4 and 0.2 < ag.upv5 < 0.3, str(ag.to_dict()))
finally: shutil.rmtree(tdir, ignore_errors=True)
t_, lg_, own_, rec_ = live_env()
tk = os.path.join(t_, 'ticks', 'MNQ_12-26'); os.makedirs(tk)
try:
    with open(os.path.join(tk, f'{day}.csv'), 'w') as fh:
        fh.write('time_et,price,size,bid,ask,bid_size,ask_size' + chr(10))
        line = f'{day} 10:00:00.123,28000.25,1,28000,28000.25,12,9' + chr(10)
        fh.write(line * 300000)
    rs = os.path.join(t_, 'rec_status.json'); open(rs, 'w').write('{"state": "terminated", "updated_utc": "2026-11-12 16:42:14"}')
    hk = HL.check(now, fb, lg_, own_, rec_, free_gb={'C:' + chr(92): 500, 'D:' + chr(92): 500}, rec_status=rs, ticks_dir=os.path.join(t_, 'ticks'))
    check("T27 a full E5n tick file is accepted and then the separate recorder is not required", not hk['warnings'] and hk['info'].get('own_ticks_mb', 0) > 10, str(hk))
    with open(os.path.join(tk, f'{day}.csv'), 'w') as fh: fh.write('time_et,price,size,bid,ask,bid_size,ask_size' + chr(10) + line * 1000)
    hs = HL.check(now, fb, lg_, own_, rec_, free_gb={'C:' + chr(92): 500, 'D:' + chr(92): 500}, rec_status=rs, ticks_dir=os.path.join(t_, 'ticks'))
    check("T28 a short E5n tick file is reported, and the recorder's absence is then reported too", any('tick writing stopped' in x for x in hs['warnings']) and any('recorder' in x for x in hs['warnings']), str(hs['warnings']))
finally: shutil.rmtree(t_, ignore_errors=True)
import e5n_checkpoints as CK
Lk = pd.DataFrame({'book_usd': [-100.0] * 60, 'model_usd': [-30.0] * 60})
r1 = CK.ledger(Lk); r2 = CK.ledger(pd.DataFrame({'book_usd': [40.0] * 60, 'model_usd': [10.0] * 60}))
check("T29 checkpoints: a book of -$6,000 after 60 sessions is STOP AND REVIEW at the 60-session checkpoint; +$2,400 is on track",
      r1['status'] == 'STOP AND REVIEW' and r1['checkpoint'].startswith('60-session') and r2['status'] == 'on track' and r1['book_p5'] == -1391, str((r1, r2)))
s1 = CK.shadow(dict(sessions=130, sip_lead=-1200.0, sip_lead_t=-1.0, iex_lead=-900.0)); s2 = CK.shadow(dict(sessions=252, sip_lead=3000.0, sip_lead_t=2.3, iex_lead=2500.0))
s3 = CK.shadow(dict(sessions=252, sip_lead=3000.0, sip_lead_t=2.3, iex_lead=1000.0))
check("T30 stock shadow rules: stop at 126 below -$1,000; at 252 free-feed candidate only if IEX keeps 75% of the lead",
      s1['status'].startswith('STOPPED') and s2['status'].startswith('LIVE CANDIDATE (free') and s3['status'].startswith('LIVE CANDIDATE only with the paid'), str((s1, s2, s3)))
import e5n_flow as FL
px = np.array([100.0, 100.25, 100.25, 100.0, 100.0]); bid = np.array([100.0, 100.0, 100.0, 100.0, np.nan]); ask = np.array([100.25, 100.25, 100.5, 100.25, np.nan])
check("T31 trade classification: at the ask = buy, at the bid = sell, in between / no quote = tick rule", list(FL.classify(px, bid, ask)) == [-1.0, 1.0, 1.0, -1.0, -1.0], str(FL.classify(px, bid, ask)))
import e5n_target as TGT
tm = pd.date_range('2026-11-16 09:31', '2026-11-16 16:00', freq='min'); op = np.full(len(tm), 100.0)
op[31:60] = 100.0 - np.arange(29) * 1.0; op[60:] = 95.0
cont_t = pd.DataFrame({'end': tm, 'o': op, 'h': op, 'l': op, 'c': op, 'v': 1.0})
mt = pd.date_range('2026-11-16 09:35', '2026-11-16 16:00', freq='5min'); tgv = np.zeros(len(mt)); tgv[5:20] = -1.0
marks_t = pd.DataFrame({'time': mt, 'target': tgv, 'ub': 105.0, 'lb': 95.0})
sh_t = TGT.shadow(cont_t, marks_t); r_t = sh_t.iloc[0]
check("T32 band target shadow: a short from 100 with a 5-point half-width leaves one minute after the open profit reaches 10 points (at 89), not at the leg's exit (95)",
      len(sh_t) == 1 and r_t.trades == 1 and r_t.hits == 1 and abs(r_t.diff_pts - 6.0) < 1e-9, str(sh_t.values.tolist()))
sh_n = TGT.shadow(cont_t, marks_t.assign(ub=120.0, lb=80.0)); check('T33 band target shadow: no hit when the target is not reached', sh_n.hits.sum() == 0 and sh_n.diff_pts.sum() == 0.0)
s_a = TGT.summary(pd.DataFrame({'trades': [1] * 252, 'hits': [1] * 20 + [0] * 232, 'diff_usd': [40.0] * 20 + [0.0] * 232}))
s_b = TGT.summary(pd.DataFrame({'trades': [1] * 252, 'hits': [1] * 20 + [0] * 232, 'diff_usd': [-40.0] * 20 + [0.0] * 232})); s_c = TGT.summary(pd.DataFrame({'trades': [1] * 30, 'hits': [0] * 30, 'diff_usd': [0.0] * 30}))
check('T34 band target shadow lines: +$800 at 252 sessions = candidate, -$800 = dropped, earlier = collecting', s_a['status'].startswith('CANDIDATE') and s_b['status'].startswith('DROPPED') and s_c['status'].startswith('collecting'), str((s_a, s_b, s_c)))
import gzip, glob, time
d_ = tempfile.mkdtemp()
try:
    c_ = os.path.join(d_, 'MNQ_12-26'); os.makedirs(c_); body = 'time_et,side,op,level,price,size\n' + ''.join(f'2026-11-16 09:30:00.{i:03d},B,U,{i % 10},25000.25,{i}\n' for i in range(1000))
    old_, new_ = os.path.join(c_, '2026-11-16.csv'), os.path.join(c_, '2026-11-17.csv')
    for p_ in (old_, new_):
        with open(p_, 'w', newline='') as fh_: fh_.write(body)
        os.utime(p_, (time.time() - 7200, time.time() - 7200))
    n_ = HL.compress_depth(d_, dt.date(2026, 11, 17))
    ok_ = n_ == 1 and not os.path.exists(old_) and os.path.exists(new_) and gzip.open(old_ + '.gz', 'rt', newline='').read() == body and not glob.glob(os.path.join(c_, '*.tmp'))
    check('T35 order-book files: finished days are gzip-compressed byte for byte, the current day is left alone', ok_, f'compressed {n_}')
finally: shutil.rmtree(d_, ignore_errors=True)
import e5n_checkpoints as CK2
g4 = 1008 / 252
ok36 = (CK2.line(1008, CK2.BOOK_MED) == CK2.BOOK_MED[-1] * g4 and abs(CK2.line(1008, CK2.BOOK_P5, CK2.BOOK_MED) - (CK2.BOOK_MED[-1] * g4 - (CK2.BOOK_MED[-1] - CK2.BOOK_P5[-1]) * 2)) < 1e-9
        and abs(CK2.line(1008, CK2.MODEL_NOEDGE_P95, zero_mean=True) - CK2.MODEL_NOEDGE_P95[-1] * 2) < 1e-9 and CK2.line(252, CK2.BOOK_P5, CK2.BOOK_MED) == CK2.BOOK_P5[-1])
check('T36 checkpoint lines after 252 sessions: medians grow with sessions, 5% / no-edge lines with the square root', ok36)
n1, s1_ = CK2.dry_spell([10, 20, -5, -5]); n2, s2_ = CK2.dry_spell([50] + [-1] * 40); n3, s3_ = CK2.dry_spell([100] + [-0.1] * 260); n4, _ = CK2.dry_spell([5, 5])
import json as _j
check('T39 checkpoint result is plain JSON (no numpy values)', isinstance(n1, int) and bool(_j.dumps(CK2.ledger(pd.DataFrame({'book_usd': [10.0, -5.0], 'model_usd': [1.0, 2.0]})))))
check('T38 dry-spell gauge: sessions since the last high, graded normal / long / longer than ever', n1 == 2 and 'normal' in s1_ and n2 == 40 and 'long but seen' in s2_ and n3 == 260 and 'LONGER' in s3_ and n4 == 0,
      str((n1, s1_, n2, n3, n4)))
mt9 = marks_t.assign(atr=1.0); b9 = TGT.bracket(cont_t, mt9).iloc[0]
b9s = TGT.bracket(cont_t.assign(o=200.0 - cont_t.o, h=200.0 - cont_t.h, l=200.0 - cont_t.l, c=200.0 - cont_t.c), mt9).iloc[0]
check('T41 band bracket shadow: a short running 6+ points in its favour leaves at the next minute after +6 ATR; one running 2 points against it leaves after -2 ATR',
      b9.trades == 1 and b9.hits == 1 and abs(b9.diff_pts - (95.0 - 93.0)) < 1e-9 and b9s.hits == 1 and abs(b9s.diff_pts - ((200 - 95.0) - (200 - 97.0))) < 1e-9, str((b9.values.tolist(), b9s.values.tolist())))
d8 = tempfile.mkdtemp()
try:
    os.makedirs(os.path.join(d8, 'log'))
    with open(os.path.join(d8, 'log', 'log.20261117.00000.txt'), 'w') as fh8:
        fh8.write("2026-11-17 12:45:00:422|1|64|Instrument='MNQ 12-26' Account='Sim101' Average price=27005 Quantity=1 Market position=Short Operation=Operation_Add\n"
                  "2026-11-17 16:00:01:145|1|64|Instrument='MNQ 12-26' Account='Sim101' Average price=26953 Quantity=1 Market position=Short Operation=Update\n"
                  "2026-11-17 16:00:02:000|1|64|Instrument='MNQ 12-26' Account='Other' Average price=0 Quantity=0 Market position=Flat Operation=Remove\n")
    ap8 = HL.account_position_from_nt_log(d8, 'Sim101', dt.date(2026, 11, 17))
    check('T40 a position left on the account after the close is read from NinjaTrader\'s own log (per account)', ap8 == ('Short', 1, '16:00:01'), str(ap8))
finally: shutil.rmtree(d8, ignore_errors=True)
d7 = tempfile.mkdtemp()
try:
    for c7, px7 in (('MNQ_12-26', 25000.0), ('MNQ_03-27', 25280.0)):
        os.makedirs(os.path.join(d7, c7))
        with open(os.path.join(d7, c7, '2026-12-14.csv'), 'w', newline='') as fh7: fh7.write('time_et,open,high,low,close,volume\n2026-12-14 10:00,1,1,1,%s,1\n' % px7)
    fx7 = pd.DataFrame({'time_et': ['2026-12-14 10:00:00'], 'action': ['Buy'], 'price': [25280.5], 'quantity': [1]})
    fc7 = HL.fill_costs(fx7, d7)
    check('T37 fill cost on a roll day is measured against the contract that was traded (not the other one, ~280 points away)', len(fc7) == 1 and abs(fc7.ticks.iloc[0] - 2.0) < 1e-9, str(fc7.values.tolist()))
finally: shutil.rmtree(d7, ignore_errors=True)
d8 = tempfile.mkdtemp()
try:
    bd8, td8 = os.path.join(d8, 'bars', 'MNQ_12-26'), os.path.join(d8, 'ticks', 'MNQ_12-26'); os.makedirs(bd8); os.makedirs(td8)
    for dd, start, end in ((dt.date(2026, 11, 12), 18 * 60, 24 * 60), (dt.date(2026, 11, 13), 0, 17 * 60)):
        mins = range(start, end)
        pd.DataFrame({'time_et': [f"{pd.Timestamp(dd) + pd.Timedelta(minutes=m + 1):%Y-%m-%d %H:%M}" for m in mins], 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 3}).to_csv(os.path.join(bd8, f'{dd}.csv'), index=False)
        tk = [(f"{pd.Timestamp(dd) + pd.Timedelta(minutes=m, seconds=s):%Y-%m-%d %H:%M:%S}.000", 1.0, z, 1.0, 1.0, 1, 1) for m in mins for s, z in ((10, 1), (40, 2))]
        pd.DataFrame(tk, columns=['time_et', 'price', 'size', 'bid', 'ask', 'bid_size', 'ask_size']).to_csv(os.path.join(td8, f'{dd}.csv'), index=False)
    tc8 = HL.tick_completeness(os.path.join(d8, 'ticks'), os.path.join(d8, 'bars'), dt.date(2026, 11, 13))
    check('T42 ticks complete: tick volume equals bar volume on every minute of the session (18:00 the evening before to 17:00)', tc8['runs'] == [] and tc8['minutes'] == 23 * 60 and tc8['ticks'] == tc8['bar'] == 3 * 23 * 60, str({k: tc8[k] for k in ('bar', 'ticks', 'minutes')}))
    f8 = os.path.join(td8, '2026-11-13.csv'); t8 = pd.read_csv(f8)
    keep = ~t8.time_et.str.slice(11, 16).between('10:00', '10:04') & (t8.time_et != '2026-11-13 14:30:40.000')
    t8[keep].to_csv(f8, index=False)
    tc8 = HL.tick_completeness(os.path.join(d8, 'ticks'), os.path.join(d8, 'bars'), dt.date(2026, 11, 13))
    rr = [HL.fmt_run(r) for r in tc8['runs']]
    check('T43 missing ticks are found to the minute and the contract: 10:01-10:05 (15) and 14:31 (2)', rr == ['11-13 10:01-10:05 (15 contracts)', '11-13 14:31 (2 contracts)'] and tc8['bar'] - tc8['ticks'] == 17, str(rr))
finally: shutil.rmtree(d8, ignore_errors=True)
t_, lg_, own_, rec_ = live_env()
with open(os.path.join(lg_, 'E5n_fills_MNQ_Sim101.csv'), 'w') as fh9:
    fh9.write("time_et,order,action,quantity,price,position_after\n2026-11-13 11:20:00.420,E5nShort,SellShort,1,26964.75,-1\n"
              "2026-11-13 12:11:00.000,E5nCover,BuyToCover,1,26935.75,0\n2026-11-13 12:35:00.404,E5nLong,Buy,1,27014.75,1\n")
try: h9 = HL.check(now, fb, lg_, own_, rec_, free_gb={'C:' + chr(92): 500, 'D:' + chr(92): 500})
finally: shutil.rmtree(t_, ignore_errors=True)
check('T44 a paper exit while waiting until flat (stamped on the bar, :00.000) is named and left out of the fills', h9['info']['MNQ_Sim101'].get('fills_last_session') == 2 and any('paper exits' in x and '12:11' in x for x in h9['warnings']), str(h9['warnings']))
print('ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL))
sys.exit(1 if FAIL else 0)
