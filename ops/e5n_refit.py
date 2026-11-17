r"""E5n yearly model refit - automatic, gated, staged (Windows task "E5n yearly refit", every 2 January, catches up after a missed start).
Adds every session since the research data ends (bars recorded by E5n itself, the minute store, the recorder) to the training set,
refits both models with the same settings, keeps every earlier model unchanged, checks everything and stages an install package.
Nothing reaches NinjaTrader by itself: the user runs the staged deploy.ps1 while E5n is flat and re-enables the strategy; the new
models serve from the first session after the install, so every earlier session keeps the model it was traded with. The nightly job
then records the installed files in the repository (local commit; the user pushes).
    python e5n_refit.py run [--cutoff YYYY-MM-DD | --cutoff last] [--dry]   default cutoff 31 December of last year; --dry -> data\_refit_dry
    python e5n_refit.py promote                                            record an installed refit in the repository (nightly job)
    python e5n_refit.py status
"""
import os, sys, re, glob, json, shutil, hashlib, subprocess, datetime as dt
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5n_features as F
import e5n_calendar as CAL
from e5n_model import ModelSet

RESEARCH_END = pd.Timestamp('2026-10-15')
NT_E5N = os.path.join(os.path.expanduser('~'), 'Documents', 'NinjaTrader 8', 'e5n')
KEYS = ('e5n_models.txt', 'e5n_models_15min.txt'); PREFIX = {'e5n_models.txt': ('live', 'final'), 'e5n_models_15min.txt': ('chklive', 'chkfinal')}
DATA = os.path.join(HERE, 'data'); DRY = os.path.join(DATA, '_refit_dry')
STATUS = os.path.join(DATA, 'e5n_refit_status.json'); LOG = os.path.join(DATA, 'e5n_refit_log.txt'); HISTORY = os.path.join(DATA, 'e5n_refit_history.json')
NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0); GIT_ID = ['-c', 'user.name=Linderwow', '-c', 'user.email=93158227+Linderwow@users.noreply.github.com']
EXE = os.path.join(HERE, 'parity_test', 'E5nParity.exe')


def repo_m(root=ROOT): return {k: os.path.join(root, 'nt8', 'e5n', k) for k in KEYS}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
    return h.hexdigest().upper()


def say(msg, out):
    print(msg, flush=True); out.append(msg)


def jload(p, default):
    try: return json.load(open(p, encoding='utf-8'))
    except (OSError, ValueError): return default


def stamp(txt, live, fin, s0):
    s1 = str((pd.Timestamp(s0) - pd.Timedelta(days=1)).date())
    txt, a = re.subn(r'(?m)^(model ' + live + r' \S+ )\S+ ', lambda m: m.group(1) + s1 + ' ', txt)
    txt, b = re.subn(r'(?m)^model ' + fin + r' \S+ ', 'model ' + fin + ' ' + str(pd.Timestamp(s0).date()) + ' ', txt)
    return txt if a == 1 and b == 1 else None


def extend_file(path, model, cutoff, live_name, final_name):
    import e5n_train as TR
    L = open(path, encoding='ascii').read().split('\n')
    fin = [i for i, ln in enumerate(L) if ln.startswith('model ') and ln.split()[1] == final_name]
    if len(fin) != 1: raise ValueError(f'{os.path.basename(path)}: expected one model named {final_name}')
    h = L[fin[0]].split(); L[fin[0]] = ' '.join(['model', live_name, h[2], str(cutoff.date())] + h[4:])
    mi = [i for i, ln in enumerate(L) if ln.startswith('models ')][0]; L[mi] = f"models {int(L[mi].split()[1]) + 1}"
    base, trees = TR.nodes_of(model)
    while L and L[-1] == '': L.pop()
    L.append(f"model {final_name} {(cutoff + pd.Timedelta(days=1)).date()} 9999-12-31 {base!r} {len(trees)}")
    for tr in trees:
        L.append(f"tree {len(tr)}")
        for f, th, le, ri, ml, leaf, val in tr: L.append(f"L {val!r}" if leaf else f"N {f} {th!r} {le} {ri} {ml}")
    return '\n'.join(L) + '\n'


def training_tables(cutoff):
    import e5n_train as TR, e5n_daily as D
    L2, ML, MR, IL = TR.research(); parts = []
    for nm, sel in (('nq', lambda d: d <= TR.JOIN), ('mnqx', lambda d: (d > TR.JOIN) & (d <= RESEARCH_END))):
        R = TR.Raw(L2, nm); B = F.features(R); k = B.k.values; p_ = B.dec.values
        B['y15'] = np.log(R.C[k, np.minimum(p_ + 14, 389)] / R.O[k, p_]); parts.append(B[sel(B.date)])
    T0 = pd.concat(parts, ignore_index=True)
    a, keep = D.all_bars(); bars = D.minute_bars(keep); S = D.Sessions(bars)
    T = F.features(S); k = T.k.values; p_ = T.dec.values
    T['mend'] = np.array([CAL.month_end_flag(d.date()) for d in S.D.date], float)[k]
    T['y15'] = np.log(S.C[k, np.minimum(p_ + 14, 389)] / S.O[k, p_])
    T1 = T[(T.date > RESEARCH_END) & (T.date <= cutoff) & S.present[k, p_ - 1]].reset_index(drop=True)
    want = [d for d in pd.bdate_range(RESEARCH_END + pd.Timedelta(days=1), cutoff) if CAL.full_session(d.date())]
    have = set(pd.to_datetime(T1.date).dt.normalize())
    return T0, T1, want, [d for d in want if d not in have], MR, S, bars


def csharp_check_days(bars, models, check, warmup):
    tmp = os.path.join(DATA, 'bars_for_refit.csv')
    b = bars.assign(key=bars.code.str.replace('-', '').astype(int))
    with open(tmp, 'w', encoding='ascii', newline='\n') as fh:
        for r in b.itertuples(): fh.write(f"{r.end:%Y-%m-%d %H:%M},{r.o!r},{r.h!r},{r.l!r},{r.c!r},{r.v!r},{r.key}\n")
    try:
        p = subprocess.run([EXE, models, tmp, 'bars', str(warmup), check], capture_output=True, text=True, timeout=900, creationflags=NOWIN)
    finally:
        try: os.remove(tmp)
        except OSError: pass
    if p.returncode != 0: return None
    out = {}
    for ln in p.stdout.strip().split('\n')[1:]:
        q = ln.split(',')
        if len(q) >= 6: out[dt.date.fromisoformat(q[0])] = dict(model=q[4], trail=q[5].strip())
    return out


def last_training_end():
    h = jload(HISTORY, [])
    return pd.Timestamp(h[-1]['cutoff']) if h else RESEARCH_END


def run(cutoff=None, dry=False):
    t0 = dt.datetime.now(); out = []; today = pd.Timestamp(t0.date())
    if cutoff == 'last': cutoff = today - pd.offsets.BDay(1)
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else pd.Timestamp(dt.date(t0.year - 1, 12, 31))
    status, log = (os.path.join(DRY, 'refit_status.json'), os.path.join(DRY, 'refit_log.txt')) if dry else (STATUS, LOG)
    end0 = last_training_end(); M = repo_m()
    say(f"E5n refit {t0:%Y-%m-%d %H:%M}: models in use trained to {end0.date()}; new training end {cutoff.date()}" + (' (DRY RUN)' if dry else ''), out)
    if cutoff <= end0:
        say("nothing to do: the models in use already include data to this date", out)
        return finish(out, dict(state='nothing to do', cutoff=str(cutoff.date())), status, log)
    cur = jload(STATUS, {})
    if not dry and cur.get('state') == 'staged - awaiting deploy' and cur.get('cutoff') == str(cutoff.date()) and os.path.isdir(cur.get('folder', '')):
        say(f"already staged: {cur['folder']}", out); return cur
    old = {k: ModelSet(M[k]) for k in KEYS}
    fin_row = {k: [m for m in old[k].models if m['name'] == PREFIX[k][1]] for k in KEYS}
    if not all(len(v) == 1 for v in fin_row.values()):
        say("REFIT STOPPED: the repository model files do not have exactly one final model each", out)
        return finish(out, dict(state='failed', cutoff=str(cutoff.date()), fails=['model files']), status, log)
    old_d0 = fin_row['e5n_models.txt'][0]['d0']; tag = f"{old_d0:%Y%m%d}"
    names = {k: (PREFIX[k][0] + tag, PREFIX[k][1]) for k in KEYS}
    T0, T1, want, missing, MR, S, bars = training_tables(cutoff)
    gates = {}; cov = 1 - len(missing) / max(len(want), 1)
    gates['sessions since the research data available (>= 95% of the regular sessions)'] = cov >= 0.95 and len(T1) > 0
    say(f"training rows: {len(T0):,} to {RESEARCH_END.date()} + {len(T1):,} on {T1.date.nunique()} sessions to {cutoff.date()} ({cov:.0%} of {len(want)} regular sessions; "
        f"missing: {', '.join(str(d.date()) for d in missing[:8]) or 'none'}{' ...' if len(missing) > 8 else ''})", out)
    T = pd.concat([T0, T1], ignore_index=True); Xm = T[F.FEATS].values; fits = {}
    for key, tgt in (('e5n_models.txt', 'y'), ('e5n_models_15min.txt', 'y15')):
        y = T[tgt].values; ok = np.isfinite(y); m = MR.model(); m.fit(Xm[ok], y[ok]); fits[key] = m
        say(f"   {key}: refit on {int(ok.sum()):,} rows", out)
    stage = DRY if dry else os.path.join(ROOT, 'staged', f"{t0:%Y%m%d}_e5n_refit_{cutoff:%Y%m%d}")
    if os.path.isdir(stage): shutil.rmtree(stage)
    os.makedirs(stage); X1 = T1[F.FEATS].values; smp = T0.sample(min(40000, len(T0)), random_state=0); Xs = smp[F.FEATS].values
    after = pd.DatetimeIndex([cutoff + pd.Timedelta(days=1)] * len(smp)); new = {}
    for key in KEYS:
        live, fin = names[key]; p = os.path.join(stage, key)
        with open(p, 'w', encoding='ascii', newline='\n') as fh: fh.write(extend_file(M[key], fits[key], cutoff, live, fin))
        new[key] = ModelSet(p); o, n = old[key], new[key]
        gates[f'{key}: every earlier model unchanged'] = bool(np.array_equal(o.predict(smp.date.values, Xs)[0], n.predict(smp.date.values, Xs)[0], equal_nan=True))
        gates[f'{key}: the sessions to {cutoff.date()} keep the model they were traded with'] = bool(np.array_equal(o.predict(T1.date.values, X1)[0], n.predict(T1.date.values, X1)[0], equal_nan=True))
        c3, nm3 = n.predict(after, Xs); d3 = fits[key].predict(Xs)
        gates[f'{key}: the new final model reads back exactly'] = bool(np.max(np.abs(c3 - d3)) < 1e-15) and set(nm3) == {fin}
        a4 = o.predict(after, Xs)[0]; r = float(np.corrcoef(a4, c3)[0, 1]); sr = float(c3.std() / a4.std())
        ra, rb = float(np.mean(np.abs(a4) > o.threshold)), float(np.mean(np.abs(c3) > o.threshold))
        gates[f'{key}: new vs old final on the training rows - correlation >= 0.6 (yearly refits 0.77-0.89), forecast size 0.67x-1.5x'] = r >= 0.6 and 0.67 <= sr <= 1.5
        gates[f'{key}: signal rate 0.5x-2x of the old final'] = ra > 0 and 0.5 <= rb / ra <= 2.0
        say(f"   {key}: new vs old final on 40,000 training rows: correlation {r:.3f}, forecast size {sr:.2f}x, signals {ra:.1%} -> {rb:.1%}", out)
    p = subprocess.run([EXE, os.path.join(stage, KEYS[0]), os.path.join(HERE, 'parity'), 'live'], capture_output=True, text=True, creationflags=NOWIN)
    gates['C# engine loads the new file and reproduces the research (live mode)'] = p.returncode == 0 and 'PARITY PASS' in p.stdout
    p = subprocess.run([EXE, os.path.join(stage, KEYS[0]), os.path.join(HERE, 'parity'), 'check', os.path.join(stage, KEYS[1])], capture_output=True, text=True, creationflags=NOWIN)
    gates['C# 15-minute check with the new files reproduces the research'] = p.returncode == 0 and 'CHECK PARITY PASS' in p.stdout
    import e5n_daily as D
    tdir = os.path.join(DATA, '_refit_test'); os.makedirs(tdir, exist_ok=True); tst = {}
    for key in KEYS:
        tst[key] = os.path.join(tdir, key)
        with open(tst[key], 'w', encoding='ascii', newline='\n') as fh: fh.write(stamp(open(os.path.join(stage, key), encoding='ascii').read(), *names[key], old_d0 + pd.Timedelta(days=1)))
    cs = csharp_check_days(bars, tst[KEYS[0]], tst[KEYS[1]], D.WARMUP)
    py, _ = D.python_days(S, ModelSet(tst[KEYS[0]]), ModelSet(tst[KEYS[1]]))
    same = sum(1 for d in py if cs and d in cs and cs[d]['trail'] == py[d]['chk_trail'])
    used = sum(1 for d in py if py[d]['model'] == 'final')
    gates['C# and Python agree day by day with the new models in use (base + 15-minute check, NinjaTrader bars)'] = cs is not None and same == len(py) and used > 0
    say(f"   C# vs Python with the new models (test copy serving from {(old_d0 + pd.Timedelta(days=1)).date()}): {same} of {len(py)} days identical, {used} days on the new models", out)
    shutil.rmtree(tdir, ignore_errors=True)
    fails = [g for g, v in gates.items() if not v]
    for g, v in gates.items(): say(f"   {'PASS' if v else 'FAIL'}  {g}", out)
    hashes = {k: sha(os.path.join(stage, k)) for k in KEYS}
    if fails:
        if not dry:
            bad = stage + '_FAILED'
            if os.path.isdir(bad): shutil.rmtree(bad)
            os.rename(stage, bad); stage = bad
        say(f"REFIT NOT STAGED: {len(fails)} check(s) failed - the current models stay in use", out)
        return finish(out, dict(state='failed', cutoff=str(cutoff.date()), fails=fails, folder=stage), status, log)
    write_package(stage, cutoff, len(T1), T1.date.nunique(), names, hashes, log)
    if not dry:
        for d in glob.glob(os.path.join(ROOT, 'staged', '*_e5n_refit_*')):
            if os.path.isdir(d) and d != stage and not d.endswith('_FAILED') and not os.path.exists(os.path.join(d, 'PROMOTED.txt')) and not os.path.exists(os.path.join(d, 'SUPERSEDED.txt')):
                open(os.path.join(d, 'SUPERSEDED.txt'), 'w', encoding='utf-8').write(f"superseded by {os.path.basename(stage)}\n")
    say(f"REFIT STAGED: {stage} - run its deploy.ps1 while E5n is flat, then untick and tick Enabled" if not dry else f"DRY RUN OK: package written to {stage}", out)
    return finish(out, dict(state='dry run ok' if dry else 'staged - awaiting deploy', cutoff=str(cutoff.date()), folder=stage, hashes=hashes,
                            names={k: list(v) for k, v in names.items()}), status, log)


DEPLOY = r"""param([string]$NinjaTrader = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'NinjaTrader 8'))
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$nt = $NinjaTrader
$cutoff = [datetime]::ParseExact('@@CUTOFF@@', 'yyyy-MM-dd', $null)
$files = @(
@@FILES@@
)
function Stamp([string]$text, $f, [string]$s0, [string]$s1) {
    if ([regex]::Matches($text, '(?m)^model ' + $f.Live + ' \S+ \S+ ').Count -ne 1 -or [regex]::Matches($text, '(?m)^model ' + $f.Final + ' \S+ ').Count -ne 1) { return $null }
    $text = [regex]::Replace($text, '(?m)^(model ' + $f.Live + ' \S+ )\S+ ', '${1}' + $s1 + ' ')
    return [regex]::Replace($text, '(?m)^model ' + $f.Final + ' \S+ ', 'model ' + $f.Final + ' ' + $s0 + ' ')
}
if (-not (Test-Path (Join-Path $nt 'e5n'))) { Write-Host "STOP: no e5n folder in $nt" -ForegroundColor Red; exit 1 }
foreach ($hb in @(Get-ChildItem (Join-Path $nt 'e5n\logs') -Filter 'E5n_status_*.json' -ErrorAction SilentlyContinue)) {
    try { $s = Get-Content $hb.FullName -Raw | ConvertFrom-Json } catch { continue }
    $fresh = ((Get-Date) - $hb.LastWriteTime).TotalMinutes -lt 20
    if ($s.state -eq 'running' -and $fresh -and [string]$s.position -ne 'Flat') {
        Write-Host "STOP: E5n on $($s.account) holds a position ($($s.position) x$($s.quantity)). Install only while flat." -ForegroundColor Red; exit 1
    }
}
foreach ($f in $files) {
    $src = Join-Path $here $f.Name
    if (-not (Test-Path $src)) { Write-Host "STOP: missing $src" -ForegroundColor Red; exit 1 }
    if ((Get-FileHash $src -Algorithm SHA256).Hash -ne $f.Hash) { Write-Host "STOP: $($f.Name) is not the checked refit - nothing was changed" -ForegroundColor Red; exit 1 }
}
$done = 0
foreach ($f in $files) {
    $dst = Join-Path $nt ('e5n\' + $f.Name)
    if (-not (Test-Path $dst)) { continue }
    $cur = [IO.File]::ReadAllText($dst)
    $m = [regex]::Match($cur, '(?m)^model ' + $f.Final + ' (\S+) ')
    if ($m.Success -and $cur.Contains("`nmodel " + $f.Live + ' ')) {
        $d = [datetime]::ParseExact($m.Groups[1].Value, 'yyyy-MM-dd', $null)
        if ((Stamp ([IO.File]::ReadAllText((Join-Path $here $f.Name))) $f $d.ToString('yyyy-MM-dd') $d.AddDays(-1).ToString('yyyy-MM-dd')) -eq $cur) { $done++ }
    }
}
if ($done -eq $files.Count) { Write-Host "Already installed - nothing to do." -ForegroundColor Green; exit 0 }
$et = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::Now, 'Eastern Standard Time')
$start = $et.Date
if ($et.TimeOfDay -ge [TimeSpan]::Parse('09:30')) { $start = $start.AddDays(1) }
if ($start -le $cutoff) { $start = $cutoff.AddDays(1) }
$s0 = $start.ToString('yyyy-MM-dd'); $s1 = $start.AddDays(-1).ToString('yyyy-MM-dd')
$texts = @{}
foreach ($f in $files) {
    $t = Stamp ([IO.File]::ReadAllText((Join-Path $here $f.Name))) $f $s0 $s1
    if ($null -eq $t) { Write-Host "STOP: $($f.Name) does not have the expected model headers - nothing was changed" -ForegroundColor Red; exit 1 }
    $texts[$f.Name] = $t
}
$rb = Join-Path $here ('rollback_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force $rb | Out-Null
$manifest = @(); $ascii = New-Object System.Text.ASCIIEncoding
foreach ($f in $files) {
    $dst = Join-Path $nt ('e5n\' + $f.Name)
    if (Test-Path $dst) {
        $keep = Join-Path $rb ('e5n__' + $f.Name)
        Copy-Item $dst $keep
        $manifest += ('replaced|e5n\' + $f.Name + '|' + (Split-Path -Leaf $keep))
    } else { $manifest += ('added|e5n\' + $f.Name + '|') }
    $tmp = $dst + '.new'
    [IO.File]::WriteAllText($tmp, $texts[$f.Name], $ascii)
    if ([IO.File]::ReadAllText($tmp) -ne $texts[$f.Name]) { Write-Host "STOP: the copy of $($f.Name) did not verify - run rollback.ps1" -ForegroundColor Red; exit 1 }
    Move-Item $tmp $dst -Force
    Write-Host "installed e5n\$($f.Name)"
}
Set-Content -Path (Join-Path $rb 'manifest.txt') -Value $manifest -Encoding UTF8
Write-Host "the new models serve sessions from $s0; earlier sessions keep the models they were traded with"
Write-Host "what was changed is listed in $rb (rollback.ps1 undoes it)"
Write-Host ""
Write-Host "Now in NinjaTrader: Strategies tab > tick Enabled on E5nStrategy again (it loads the new models). No F5 needed." -ForegroundColor Green
"""

ROLLBACK = r"""param([string]$NinjaTrader = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'NinjaTrader 8'))
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$nt = $NinjaTrader
$rb = Get-ChildItem $here -Directory -Filter 'rollback_*' | Where-Object Name -notmatch '_undone$' | Sort-Object Name | Select-Object -Last 1
if (-not $rb) { Write-Host "Nothing to roll back (no rollback_* folder)." -ForegroundColor Yellow; exit 0 }
$bin = Join-Path $nt ('_to_delete\' + $rb.Name + '_e5n')
foreach ($line in Get-Content (Join-Path $rb.FullName 'manifest.txt')) {
    $p = $line -split '\|'
    $dst = Join-Path $nt $p[1]
    if ($p[0] -eq 'replaced') {
        Copy-Item (Join-Path $rb.FullName $p[2]) $dst -Force
        Write-Host "restored $($p[1])"
    } elseif ($p[0] -eq 'added' -and (Test-Path $dst)) {
        $to = Join-Path $bin $p[1]
        New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
        Move-Item $dst $to
        Write-Host "moved out $($p[1])"
    }
}
Rename-Item $rb.FullName ($rb.Name + '_undone')
Write-Host "Rolled back. NinjaTrader: Strategies tab > untick and tick Enabled on E5nStrategy (it reloads the models)." -ForegroundColor Green
"""


def write_package(stage, cutoff, rows, sessions, names, hashes, log):
    files = ",\n".join(f"    @{{ Name = '{k}'; Live = '{names[k][0]}'; Final = '{names[k][1]}'; Hash = '{hashes[k]}' }}" for k in KEYS)
    for nm, body in (('deploy.ps1', DEPLOY.replace('@@CUTOFF@@', str(cutoff.date())).replace('@@FILES@@', files)), ('rollback.ps1', ROLLBACK)):
        open(os.path.join(stage, nm), 'w', encoding='utf-8', newline='\r\n').write(body)
    name = os.path.basename(stage)
    open(os.path.join(stage, 'USAGE.txt'), 'w', encoding='utf-8', newline='\n').write(
        f"{name} - E5n yearly model refit (automatic; every check passed)\n\n"
        f"WHAT CHANGES  Only the two model files: both models refit with the same settings on all data to {cutoff.date()} ({rows:,} decision rows on\n"
        f"  {sessions} sessions since {RESEARCH_END.date()} added to the research rows). Every earlier model is unchanged. The new models serve\n"
        f"  sessions from the install on (deploy.ps1 writes that date into the files), so every earlier session keeps the model it was traded\n"
        f"  with. No code change and nothing to compile; NinjaTrader reads the model files when the strategy is enabled.\n\n"
        f"INSTALL (while E5n is flat; easiest outside 09:30-16:00 ET)\n"
        f"  1. NinjaTrader: Strategies tab > untick Enabled on E5nStrategy.\n"
        f"  2. PowerShell:  & \"{os.path.join(stage, 'deploy.ps1')}\"\n"
        f"  3. Strategies tab: tick Enabled again. The nightly job then records the installed files in the repository (a local commit to push).\n\n"
        f"CHECKS  {log}\n"
        f"UNDO    & \"{os.path.join(stage, 'rollback.ps1')}\", then untick and tick Enabled.\n")


def finish(out, st, status=STATUS, log=LOG):
    st['time'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M'); os.makedirs(os.path.dirname(status), exist_ok=True)
    json.dump(st, open(status, 'w', encoding='utf-8'), indent=1)
    with open(log, 'a', encoding='utf-8') as fh: fh.write("\n".join(out) + "\n\n")
    return st


def code_package(root):
    c = [d for d in sorted(glob.glob(os.path.join(root, 'staged', '*_e5n_v*'))) if os.path.isdir(d) and not os.path.exists(os.path.join(d, 'SUPERSEDED.txt'))]
    return c[-1] if c else None


def promote(nt_e5n=NT_E5N, root=ROOT, status=STATUS, history=HISTORY, log=LOG, commit=True):
    st = jload(status, None)
    if not st or st.get('state') != 'staged - awaiting deploy': return st
    inst, start = {}, None
    for k in KEYS:
        p = os.path.join(nt_e5n, k)
        if not os.path.exists(p): return st
        cur = open(p, encoding='ascii', newline='').read(); live, fin = st['names'][k]
        m = re.search(r'(?m)^model ' + fin + r' (\S+) ', cur)
        if not m or stamp(open(os.path.join(st['folder'], k), encoding='ascii', newline='').read(), live, fin, m.group(1)) != cur: return st
        if start not in (None, m.group(1)): return st
        inst[k] = p; start = m.group(1)
    M = repo_m(root)
    for k in KEYS: shutil.copyfile(inst[k], M[k])
    pk = code_package(root); note = ''
    if pk:
        dp = os.path.join(pk, 'deploy.ps1'); txt = open(dp, encoding='utf-8', newline='').read()
        for k in KEYS: txt = re.sub(r"(Src = '[^']*\\" + re.escape(k) + r"';\s*Dst = '[^']+';\s*Hash = ')[0-9A-F]{64}'", lambda m_, h=sha(M[k]): m_.group(1) + h + "'", txt)
        open(dp, 'w', encoding='utf-8', newline='').write(txt)
        with open(os.path.join(pk, 'USAGE.txt'), 'a', encoding='utf-8', newline='\n') as fh:
            fh.write(f"\n{dt.date.today()}: model files replaced by the yearly refit to {st['cutoff']} ({os.path.basename(st['folder'])}); deploy.ps1 now carries their hashes.\n")
        note = os.path.basename(pk)
    if commit:
        msg = f"Yearly model refit: both models trained to {st['cutoff']}, serving from {start}"
        subprocess.run(['git', '-C', root, 'add', 'nt8/e5n/e5n_models.txt', 'nt8/e5n/e5n_models_15min.txt'], capture_output=True, text=True, creationflags=NOWIN)
        subprocess.run(['git', '-C', root] + GIT_ID + ['commit', '-q', '-m', msg], capture_output=True, text=True, creationflags=NOWIN)
    now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    open(os.path.join(st['folder'], 'PROMOTED.txt'), 'w', encoding='utf-8').write(f"installed by the user (new models from {start}); recorded in the repository {now}\n")
    st.update(state='deployed and recorded', serving_from=start, recorded=now)
    json.dump(st, open(status, 'w', encoding='utf-8'), indent=1)
    h = jload(history, []); h.append(dict(cutoff=st['cutoff'], serving_from=start, recorded=now, folder=st['folder'])); json.dump(h, open(history, 'w', encoding='utf-8'), indent=1)
    with open(log, 'a', encoding='utf-8') as fh:
        fh.write(f"{now} refit to {st['cutoff']} installed by the user (serving from {start}); recorded in the repository (local commit - push it)"
                 + (f"; {note} deploy.ps1 hashes updated" if note else '') + "\n\n")
    return st


def warning(now=None):
    st = jload(STATUS, None)
    if not st: return None
    if st.get('state') == 'staged - awaiting deploy':
        return f"yearly model refit to {st['cutoff']} staged {st['time'][:10]}: install it while flat - {os.path.join(st['folder'], 'USAGE.txt')}"
    if st.get('state') == 'failed':
        return f"yearly model refit to {st['cutoff']} FAILED its checks ({'; '.join(st.get('fails', []))[:200]}) - the current models stay; see {LOG}"
    return None


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['run']:
        cut = a[a.index('--cutoff') + 1] if '--cutoff' in a else None
        r = run(cut, dry='--dry' in a); sys.exit(0 if r.get('state') in ('staged - awaiting deploy', 'dry run ok', 'nothing to do') else 1)
    elif a[:1] == ['promote']: print(promote())
    elif a[:1] == ['status']: print(json.dumps(jload(STATUS, {}), indent=1)); print(warning() or 'no refit waiting')
    else: print(__doc__)
