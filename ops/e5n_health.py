import os, glob, gzip, hashlib, json, shutil, subprocess, datetime as dt
import numpy as np, pandas as pd
import e5n_calendar as CAL

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), 'nt8', 'bin', 'Custom', 'AddOns', 'E5nCore.cs')
PARITY_DIR = os.path.join(HERE, 'parity_test'); EXE = os.path.join(PARITY_DIR, 'E5nParity.exe')
CSC = r'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
DECISIONS = list(range(15, 386, 5))
NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
LOW_GB, WARN_GB = 15, 50


def third_friday(y, m):
    d = dt.date(y, m, 1); return d + dt.timedelta(days=(4 - d.weekday()) % 7 + 14)


def roll_info(instrument, today):
    try:
        mm, yy = instrument.split()[-1].split('-'); y, m = 2000 + int(yy), int(mm)
    except Exception:
        return None
    exp = third_friday(y, m); roll = exp - dt.timedelta(days=4); nm = m + 3; ny = y + (nm > 12); nm = nm - 12 if nm > 12 else nm
    return dict(contract=f"{m:02d}-{y % 100:02d}", expiry=exp, roll=roll, days_to_roll=(roll - today).days, next_contract=f"{nm:02d}-{ny % 100:02d}")


def market_open(t):
    wd = t.weekday(); m = t.hour * 60 + t.minute
    if wd == 5 or (wd == 6 and m < 18 * 60) or (wd == 4 and m >= 17 * 60): return False
    return not (17 * 60 <= m < 18 * 60)


def heartbeat_due(t):
    m = t.hour * 60 + t.minute
    if not market_open(t) or not CAL.full_session(t.date()) and t.weekday() < 5: return False
    if t.weekday() < 5 and 9 * 60 + 50 <= m <= 16 * 60 + 55: return True
    return 18 * 60 + 10 <= m <= 23 * 60 + 55


def ranges(mins):
    if not mins: return ''
    out = []; a = b = mins[0]
    for x in mins[1:]:
        if x == b + 5: b = x
        else: out.append((a, b)); a = b = x
    out.append((a, b)); f = lambda p: f"{(570 + p) // 60}:{(570 + p) % 60:02d}"
    return ', '.join(f(a) if a == b else f"{f(a)}-{f(b)}" for a, b in out)


def read_log(path):
    try:
        with open(path, encoding='ascii', errors='replace') as fh: lines = fh.read().splitlines()
    except OSError as e:
        return None, 0, f"cannot be read ({e})"
    if not lines: return pd.DataFrame(), 0, ''
    head = lines[0].split(','); good = []; bad = 0
    for ln in lines[1:]:
        p = ln.split(',')
        if len(p) == len(head): good.append(p)
        elif ln.strip(): bad += 1
    df = pd.DataFrame(good, columns=head)
    for c in df.columns:
        if c not in ('date', 'time_et', 'model', 'order', 'action'): df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, bad, ''


def rebuild_parity():
    if not os.path.exists(CORE): return 'engine source missing'
    if os.path.exists(EXE) and os.path.getmtime(EXE) >= os.path.getmtime(CORE) and os.path.getmtime(EXE) >= os.path.getmtime(os.path.join(PARITY_DIR, 'E5nParity.cs')): return ''
    p = subprocess.run([CSC, '-nologo', '-optimize+', '-out:E5nParity.exe', 'E5nParity.cs', CORE], cwd=PARITY_DIR, capture_output=True, text=True, creationflags=NOWIN)
    return 'rebuilt the C# test program (the engine source was newer)' if p.returncode == 0 else f"C# test program rebuild FAILED: {(p.stdout or p.stderr)[-300:]}"


def fill_costs(fx, own_dir):
    out = []
    if fx is None or not len(fx): return pd.DataFrame(columns=['day', 'ticks', 'n'])
    for day, g in fx.groupby(fx.time_et.str.slice(0, 10)):
        files = sorted(glob.glob(os.path.join(own_dir, 'MNQ_*', f'{day}.csv')))
        if not files: continue
        best = None
        for f in files:
            b, _, _ = read_log(f)
            if b is None or not len(b): continue
            cl = dict(zip(b.time_et.astype(str).str.slice(0, 16), pd.to_numeric(b.close, errors='coerce')))
            ref = [cl.get(str(r.time_et)[:16]) for r in g.itertuples()]
            gap = [abs(float(r.price) - v) for r, v in zip(g.itertuples(), ref) if v is not None and v == v]
            score = float(np.median(gap)) if gap else np.inf
            if best is None or score < best[0]: best = (score, cl)
        if best is None or not np.isfinite(best[0]): continue
        close = best[1]
        for r in g.itertuples():
            ref = close.get(str(r.time_et)[:16])
            if ref is None or ref != ref: continue
            act = str(r.action); sgn = 1.0 if act in ('Buy', 'BuyToCover') else -1.0 if act in ('Sell', 'SellShort') else 0.0
            if sgn == 0.0: continue
            out.append((day, sgn * (float(r.price) - ref) / 0.25, int(r.quantity)))
    if not out: return pd.DataFrame(columns=['day', 'ticks', 'n'])
    d = pd.DataFrame(out, columns=['day', 'ticks', 'n'])
    return d.loc[d.index.repeat(d.n)].assign(n=1).reset_index(drop=True)


def account_checks(tag, st, ev):
    import re
    import e5n_account as A
    P, W = [], []; mode = st.get('account_mode') or 'Fixed'
    if mode == 'Fixed': return P, W, None
    acc = dict(mode=mode, units=st.get('units'), balance=st.get('balance'), threshold_balance=st.get('threshold_balance'), cushion=st.get('cushion'), target_reached=st.get('target_reached'))
    live = [e for e in ev if e[1] == 'live']
    cfg = re.search(r'Funded \(start (\S+), drawdown (\S+), lock (\S+), cushion per unit (\S+), max units (\d+)', live[-1][3]) if live else None
    if st.get('balance') is None and st.get('state') in ('running', 'halted'):
        P.append(f"{tag}: account mode {mode} but the account balance is not readable - sizing stays at 1 unit and the evaluation target is not watched")
    if any(e[1] == 'balance_missing' for e in ev[-50:]): W.append(f"{tag}: the strategy logged that the account's cash value was not available")
    if mode == 'Evaluation' and st.get('target_reached'):
        W.append(f"{tag}: evaluation target reached - E5n is flat for good on this account; once Bulenox confirms the pass, run E5n on the funded account (Account mode Funded)")
    if mode == 'Funded' and cfg:
        start, dd, lock, per = (float(cfg.group(i)) for i in range(1, 5)); mx = int(cfg.group(5))
        bad = []
        for e in [e for e in ev if e[1] == 'sizing'][-30:]:
            q = re.search(r'balance (\S+), highest close (\S+), threshold (\S+), cushion (\S+), units (\d+)', e[3])
            if not q or q.group(1) == 'unknown': continue
            s = A.Sizer('Funded', start, dd, lock, 0, per, mx, float(q.group(2)) - start)
            if abs(start + s.threshold - float(q.group(3))) > 0.011 or s.units(float(q.group(1))) != int(q.group(5)): bad.append(e[0][:10])
        if bad: P.append(f"{tag}: funded sizing does not follow the rule on {', '.join(bad[-5:])} - check the 'Account' settings")
        sz = [e for e in ev if e[1] == 'sizing']
        if len(sz) >= 2:
            a = re.search(r'units (\d+)', sz[-2][3]); b = re.search(r'units (\d+)', sz[-1][3])
            if a and b and a.group(1) != b.group(1): W.append(f"{tag}: funded size changed from {a.group(1)} to {b.group(1)} unit(s) on {sz[-1][0][:10]} ({sz[-1][3][:120]})")
    return P, W, acc


def compress_depth(ddir, before):
    if before is None: return 0
    n = 0
    for f in glob.glob(os.path.join(ddir, 'MNQ_*', '*.csv')):
        day = os.path.basename(f)[:10]
        try:
            if dt.date.fromisoformat(day) >= before or dt.datetime.now().timestamp() - os.path.getmtime(f) < 3600: continue
        except ValueError: continue
        gz, tmp = f + '.gz', f + '.gz.tmp'
        try:
            h1 = hashlib.sha256()
            with open(f, 'rb') as src, gzip.open(tmp, 'wb', compresslevel=6) as dst:
                for chunk in iter(lambda: src.read(1 << 22), b''): h1.update(chunk); dst.write(chunk)
            h2 = hashlib.sha256()
            with gzip.open(tmp, 'rb') as chk:
                for chunk in iter(lambda: chk.read(1 << 22), b''): h2.update(chunk)
            if h1.digest() != h2.digest(): os.remove(tmp); continue
            os.replace(tmp, gz); os.remove(f); n += 1
        except OSError:
            try: os.remove(tmp)
            except OSError: pass
    return n


def check(now, bars, logs_dir, own_dir, rec_dir, disks=('C:\\', 'D:\\'), free_gb=None, rec_status=None, ticks_dir=None):
    P, W, I = [], [], {}
    today = now.date()
    fix = rebuild_parity()
    if fix:
        (P if 'FAILED' in fix or 'missing' in fix else I.setdefault('repairs', [])).append(fix)
    for dsk in disks:
        try: free = free_gb[dsk] if free_gb else shutil.disk_usage(dsk).free / 1e9
        except OSError: continue
        I[f'free_{dsk[0]}_gb'] = round(free, 1)
        if free < LOW_GB: P.append(f"disk {dsk[0]}: only {free:.0f} GB free")
        elif free < WARN_GB: W.append(f"disk {dsk[0]}: {free:.0f} GB free")
    bars = bars if bars is not None else pd.DataFrame(columns=['date', 'pos'])
    full = sorted(d for d in pd.to_datetime(bars.date).dt.date.unique() if d <= today and CAL.full_session(d)) if len(bars) else []
    last = full[-1] if full else None; I['last_full_session'] = str(last) if last else ''
    if last is None or (today.weekday() < 5 and CAL.full_session(today) and now.hour * 60 + now.minute >= 16 * 60 + 5 and last < today):
        W.append(f"no one-minute bars for {today if last is None or last < today else last} yet - the strategy's bar log and the recorder both lack it")
    present = set(bars[pd.to_datetime(bars.date).dt.date == last].pos.astype(int)) if last else set()
    tags = [os.path.basename(f)[len('E5n_decisions_'):-4] for f in glob.glob(os.path.join(logs_dir, 'E5n_decisions_*.csv'))]
    live = sorted(t for t in tags if 'Backtest' not in t and '_to_' not in t)
    I['live_accounts'] = live
    if not live: W.append('E5n is not running on any account (no live decision log)')
    for tag in live:
        r = {}; I[tag] = r
        sp = os.path.join(logs_dir, f'E5n_status_{tag}.json'); st = None
        if os.path.exists(sp):
            try: st = json.load(open(sp, encoding='utf-8'))
            except Exception as e: P.append(f"{tag}: status file unreadable ({type(e).__name__})")
        ev = []
        try:
            with open(os.path.join(logs_dir, f'E5n_events_{tag}.txt'), encoding='utf-8', errors='replace') as fh: ev = [ln.split('|') for ln in fh.read().splitlines() if ln.count('|') >= 3]
        except OSError: pass
        last_ev = ev[-1] if ev else None
        if st is not None and last_ev and last_ev[1] == 'live' and pd.Timestamp(last_ev[0]) > pd.Timestamp(st.get('updated_et', '1970-01-01')):
            I.setdefault('notes', []).append(f"{tag}: E5n restarted at {last_ev[0]}, waiting for its first real-time bar (the status file is from the previous instance)")
            st = dict(st, state='running', updated_et=last_ev[0])
        if st is None:
            W.append(f"{tag}: no status file yet (written from the first real-time bar after 15:59 ET or the first decision)")
        else:
            upd = pd.Timestamp(st.get('updated_et', '1970-01-01')); age = (pd.Timestamp(now) - upd).total_seconds() / 60
            r.update(state=st.get('state'), updated=str(upd), position=st.get('position'), instrument=st.get('instrument'), connected=st.get('connected'))
            if st.get('state') == 'refused': P.append(f"{tag}: E5n refused to start - {st.get('halt_note', '')}")
            elif st.get('state') == 'terminated': P.append(f"{tag}: E5n was stopped (state terminated at {upd:%Y-%m-%d %H:%M}) - re-enable it")
            elif st.get('state') == 'halted': W.append(f"{tag}: daily loss stop hit - {st.get('halt_note', '')}")
            if st.get('state') in ('running', 'halted') and heartbeat_due(now) and age > 10:
                P.append(f"{tag}: no update for {age:.0f} minutes while the market is open - NinjaTrader closed, the strategy disabled or the data feed down")
            if st.get('connected') is False and st.get('state') in ('running', 'halted'): P.append(f"{tag}: account connection was down at the last update ({upd:%H:%M})")
            if st.get('band_off'): P.append(f"{tag}: band leg switched itself off - the chart has no overnight session (use the instrument's default trading hours)")
            P_, W_, acc = account_checks(tag, st, ev); P += P_; W += W_
            if acc: r['account'] = acc
            m = now.hour * 60 + now.minute
            if st.get('position') not in (None, 'Flat') and not (9 * 60 + 30 <= m < 16 * 60 + 5) and age < 60: P.append(f"{tag}: holding {st.get('position')} x{st.get('quantity')} outside 09:30-16:00 ET")
            ri = roll_info(st.get('instrument', ''), today)
            if ri:
                r['roll'] = dict(contract=ri['contract'], roll=str(ri['roll']), days=ri['days_to_roll'])
                if ri['days_to_roll'] <= 0: P.append(f"{tag}: contract {ri['contract']} has rolled ({ri['roll']}) - disable E5n, change the instrument to MNQ {ri['next_contract']}, enable again")
                elif ri['days_to_roll'] <= 7: W.append(f"{tag}: MNQ {ri['contract']} rolls on {ri['roll']:%a %Y-%m-%d} - after that session close, switch E5n to MNQ {ri['next_contract']}")
        x, bad, err = read_log(os.path.join(logs_dir, f'E5n_decisions_{tag}.csv'))
        if err: P.append(f"{tag}: decision log {err}"); continue
        if bad: P.append(f"{tag}: decision log has {bad} broken line(s)")
        if x is None: continue
        if len(x):
            dup = int(x.duplicated(['date', 'minute']).sum())
            if dup: P.append(f"{tag}: decision log has {dup} duplicate decision(s)")
        if last is None: continue
        since = pd.Timestamp(st['live_since']) if st and st.get('live_since') else (pd.Timestamp(x.date.min()) if len(x) else pd.Timestamp(now))
        y = x[x.date == str(last)] if len(x) else pd.DataFrame(columns=['minute']); got = set(y.minute.astype(int))
        exp = [p for p in DECISIONS if (p - 1) in present and pd.Timestamp(last) + pd.Timedelta(minutes=570 + p) > since]
        miss = [p for p in exp if p not in got]
        r.update(decisions_last_session=len(got), expected=len(exp))
        if exp and miss and len(miss) == len(exp): P.append(f"{tag}: no decisions at all on {last} although bars exist - the strategy was not running")
        elif miss: P.append(f"{tag}: {len(miss)} of {len(exp)} decisions missing on {last} ({ranges(miss)}) - the strategy or its data stopped then")
        if 'net' in y.columns and len(y) > 1:
            yy = y.sort_values('minute'); nxt = yy.position.shift(-1); ok = (yy.minute.shift(-1) - yy.minute == 5)
            r['followed'] = f"{int((nxt[ok] == yy.net[ok]).sum())}/{int(ok.sum())}"
            if int(ok.sum()) and (nxt[ok] != yy.net[ok]).sum(): P.append(f"{tag}: on {last} the position did not follow the net target {int((nxt[ok] != yy.net[ok]).sum())} time(s)")
        fx, fbad, ferr = read_log(os.path.join(logs_dir, f'E5n_fills_{tag}.csv'))
        if fx is not None and len(fx):
            f1 = fx[fx.time_et.str.startswith(str(last))]; r['fills_last_session'] = int(len(f1))
            if fbad: P.append(f"{tag}: fill log has {fbad} broken line(s)")
            late = f1[~f1.time_et.str.slice(11, 16).between('09:30', '16:02')]
            if len(late): W.append(f"{tag}: {len(late)} fill(s) on {last} outside 09:30-16:02 ET")
            ca = fill_costs(fx, own_dir); c1 = ca[ca.day == str(last)] if len(ca) else ca
            if len(ca):
                r['fill_cost_all'] = f"{ca.ticks.mean():+.2f} ticks a side over {int(ca.n.sum())} fills ({ca.day.nunique()} sessions; 2.35 assumed)"
            if len(c1):
                r['fill_cost_last_session'] = f"{c1.ticks.mean():+.2f} ticks a side over {int(c1.n.sum())} fills"
                if c1.ticks.mean() > 4.0: W.append(f"{tag}: fills on {last} cost {c1.ticks.mean():+.2f} ticks a side against the decision bar's close (2.35 assumed)")
    own = sorted(glob.glob(os.path.join(own_dir, 'MNQ_*', f'{last}.csv'))) if last else []
    rec = sorted(glob.glob(os.path.join(rec_dir, 'MNQ_*', f'{last}_bars1minute.csv'))) if last else []
    if own:
        o, obad, _ = read_log(own[-1]); I['own_bars_last_session'] = int(len(o))
        if obad: P.append(f"strategy bar log {os.path.basename(own[-1])} has {obad} broken line(s)")
        t = pd.to_datetime(o.time_et, errors='coerce'); rth = ((t.dt.hour * 60 + t.dt.minute) > 570) & ((t.dt.hour * 60 + t.dt.minute) <= 960)
        I['own_bars_regular_session'] = int(rth.sum())
        if live and rth.sum() < 300: W.append(f"strategy bar log for {last} has only {int(rth.sum())} regular-session minutes")
        if rec:
            rr, _, _ = read_log(rec[-1])
            if rr is not None and len(rr):
                a = pd.DataFrame({'t': t, 'c': o.close}).dropna(); b = pd.DataFrame({'t': pd.to_datetime(rr.time_et, errors='coerce'), 'c2': rr.close}).dropna()
                j = a.merge(b, on='t'); same = float((np.abs(j.c - j.c2) < 0.011).mean()) if len(j) else float('nan')
                near = float((np.abs(j.c - j.c2) <= 0.51).mean()) if len(j) else float('nan')
                I['own_vs_recorder'] = f"{len(j)} common minutes, closes equal on {same:.1%}, within 2 ticks on {near:.1%}" if len(j) else 'no common minutes'
                if len(j) > 100 and near < 0.95:
                    W.append(f"on {last} the strategy's bars (NinjaTrader) and the recorder's live bars differ by more than 2 ticks on {1 - near:.1%} of minutes (normal: about 1%) - one of the two feeds was incomplete that day")
    tdir = ticks_dir if ticks_dir else os.path.join(os.path.dirname(own_dir), 'ticks')
    own_ticks = sorted(glob.glob(os.path.join(tdir, 'MNQ_*', f'{last}.csv'))) if last else []
    own_ok = False
    if own_ticks:
        mb = os.path.getsize(own_ticks[-1]) / 1e6; I['own_ticks_mb'] = round(mb, 1)
        try:
            with open(own_ticks[-1], encoding='ascii', errors='replace') as fh: head = fh.readline().strip(); row = fh.readline().strip().split(',')
            if head != 'time_et,price,size,bid,ask,bid_size,ask_size' or len(row) != 7: W.append(f"E5n tick file for {last} has an unexpected layout")
            elif mb < 10: W.append(f"E5n tick file for {last} is only {mb:.1f} MB (a full session is normally about 50-80 MB) - tick writing stopped part-way")
            else: own_ok = True
        except OSError as e: W.append(f"E5n tick file for {last} unreadable ({e})")
    elif live and last:
        dirs = [d for d in glob.glob(os.path.join(tdir, 'MNQ_*')) if os.path.isdir(d)]
        since = min((dt.datetime.fromtimestamp(os.path.getctime(d)) for d in dirs), default=None)
        if since is not None and since < dt.datetime.combine(last, dt.time(9, 30)):
            W.append(f"E5n wrote no tick file for {last} - check 'Write ticks' in the strategy settings")
    for tag in live:
        sp = os.path.join(logs_dir, f'E5n_status_{tag}.json')
        try: st_ = json.load(open(sp, encoding='utf-8'))
        except Exception: continue
        I.setdefault(tag, {}).update(ticks_written=st_.get('ticks_written'), tick_errors=st_.get('tick_errors'))
        if st_.get('tick_errors'): W.append(f"{tag}: the tick writer reported {st_['tick_errors']} write error(s)")
        if (st_.get('tick_queue') or 0) > 200000: W.append(f"{tag}: {st_['tick_queue']:,} ticks waiting to be written - the disk is not keeping up")
        if st_.get('account_mode', 'Fixed') == 'Fixed':
            drift = [f"{k} {st_.get(k)} (tested {v})" for k, v in (('contracts', 1), ('check15', True), ('band_contracts', 1), ('max_daily_loss', 0)) if k in st_ and st_.get(k) != v]
            if drift: W.append(f"{tag}: settings differ from the tested book the forward ledger assumes - " + ', '.join(drift))
        if 'depth_written' in st_:
            I[tag].update(depth_written=st_.get('depth_written'), depth_errors=st_.get('depth_errors'))
            if st_.get('depth_errors'): W.append(f"{tag}: the order-book writer reported {st_['depth_errors']} write error(s)")
            if (st_.get('depth_queue') or 0) > 2000000: W.append(f"{tag}: {st_['depth_queue']:,} order-book changes waiting to be written - the disk is not keeping up")
    ddir = os.path.join(os.path.dirname(tdir), 'depth')
    I['depth_compressed'] = compress_depth(ddir, last)
    own_depth = sorted(glob.glob(os.path.join(ddir, 'MNQ_*', f'{last}.csv*'))) if last else []
    if own_depth:
        mb = os.path.getsize(own_depth[-1]) / 1e6; I['own_depth_mb'] = round(mb, 1)
        try:
            op = gzip.open if own_depth[-1].endswith('.gz') else open
            with op(own_depth[-1], 'rt', encoding='ascii', errors='replace') as fh: head = fh.readline().strip(); row = fh.readline().strip().split(',')
            if head != 'time_et,side,op,level,price,size' or len(row) != 6: W.append(f"E5n order-book file for {last} has an unexpected layout")
        except (OSError, EOFError) as e: W.append(f"E5n order-book file for {last} unreadable ({e})")
    elif live and last and any(os.path.isdir(d) and dt.datetime.fromtimestamp(os.path.getctime(d)) < dt.datetime.combine(last, dt.time(9, 30)) for d in glob.glob(os.path.join(ddir, 'MNQ_*'))):
        refused = False
        for lf in sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(logs_dir)), 'log', 'log.*.txt')))[-10:]:
            try:
                with open(lf, encoding='utf-8', errors='replace') as fh: refused = refused or 'get_order_book error' in fh.read()
            except OSError: pass
        if refused: I['order_book'] = "the data feed refuses the order book (Rithmic 'get_order_book error : 13' = no permission) - nothing to record until the feed includes market depth"
        else: W.append(f"E5n wrote no order-book file for {last} - check 'Write order book' in the strategy settings")
    tapes =sorted(glob.glob(os.path.join(rec_dir, 'MNQ_*', '*_tape.csv')), key=os.path.basename)
    if tapes: I['recorder_last_tape'] = os.path.basename(tapes[-1])[:10]
    if own_ok: return dict(problems=P, warnings=W, info=I)
    if last:
        tp = [f for f in tapes if os.path.basename(f).startswith(str(last))]
        if not tp: W.append(f"tick recorder: no tick file for {last} - order-flow history is not being collected (keep the TapeRecorder indicator on a chart)")
        else:
            gb = os.path.getsize(tp[-1]) / 1e9; I['recorder_tape_gb'] = round(gb, 2)
            if gb < 0.3: W.append(f"tick recorder: the tick file for {last} is only {gb * 1000:.0f} MB (a full session is normally 3-4 GB) - recording stopped part-way")
            try:
                with open(tp[-1], encoding='ascii', errors='replace') as fh: head = fh.readline().strip().split(','); row = fh.readline().strip().split(',')
                if head[:7] != ['kind', 'feed_utc', 'recv_utc', 'price', 'size', 'bid', 'ask'] or len(row) < 7: W.append(f"tick recorder: unexpected tick file layout {head[:7]}")
            except OSError as e: W.append(f"tick recorder: tick file unreadable ({e})")
    rsp = rec_status if rec_status else os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8', 'agent_exports', 'TapeRecorder_status.json')
    if os.path.exists(rsp):
        try:
            rs = json.load(open(rsp, encoding='utf-8')); upd = pd.Timestamp(rs.get('updated_utc', '1970-01-01')).tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)
            I['recorder_state'] = f"{rs.get('state')} (updated {upd:%Y-%m-%d %H:%M} ET)"
            if heartbeat_due(now) and (rs.get('state') != 'running' or (pd.Timestamp(now) - upd).total_seconds() > 1800):
                W.append(f"tick recorder not running (state {rs.get('state')}, last update {upd:%Y-%m-%d %H:%M} ET) while the market is open")
            if rs.get('dropped') or rs.get('lost') or rs.get('stream_failures'): W.append(f"tick recorder reports dropped {rs.get('dropped')}, lost {rs.get('lost')}, stream failures {rs.get('stream_failures')}")
        except Exception as e: W.append(f"tick recorder status unreadable ({type(e).__name__})")
    else:
        W.append('tick recorder status file missing - the TapeRecorder indicator has never run on this PC')
    return dict(problems=P, warnings=W, info=I)
