r"""Completeness and integrity audit of the downloaded stock bars (cache\stocks_alpaca_5m).
    python e5s_audit.py            audit; writes E5S_DATA_AUDIT.txt
    python e5s_audit.py repair     audit, move every failing stock-year aside, download it again, audit again
Checks per stock: every file opens; the years form one unbroken run; every trading day between its first and last day is present;
bars per day against the reference stocks; duplicate times; prices positive and low <= open, close <= high; no missing values.
"""
import os, sys, glob, shutil, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import e5s_alpaca as AL

DIR = AL.OUT; OUT = os.path.join(HERE, 'E5S_DATA_AUDIT.txt'); REF = ('AAPL', 'MSFT', 'CSCO', 'INTC')
ASIDE = os.path.join(os.path.dirname(os.path.dirname(HERE)), '_to_delete', 'stock_files_failed_audit')
LINES = []


def say(s=''):
    print(s, flush=True); LINES.append(s)


def load(sym):
    out = {}; bad = []
    for f in sorted(glob.glob(os.path.join(DIR, f'{sym}_*.parquet'))):
        y = int(os.path.basename(f)[len(sym) + 1:-8])
        try:
            d = pd.read_parquet(f)
            if list(d.columns)[:6] != ['time', 'open', 'high', 'low', 'close', 'volume']: bad.append((y, 'unexpected columns'))
            else: out[y] = d
        except Exception as e:
            bad.append((y, f'does not open ({type(e).__name__})'))
    return out, bad


def reference():
    per = []
    for s in REF:
        d, bad = load(s)
        if not d: continue
        x = pd.concat(d.values()); per.append(x.groupby(x.time.dt.normalize()).size().rename(s))
    if not per: return None
    r = pd.concat(per, axis=1); days = r.index[r.notna().sum(axis=1) >= max(1, len(per) - 1)]
    return r.loc[days].median(axis=1)


def audit(only=None):
    del LINES[:]; t0 = time.time(); ref = reference(); end_year = (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).year; years = list(range(2016, end_year + 1))
    say("=" * 150); say(f"STOCK DATA AUDIT  e5s_audit.py  {pd.Timestamp.now():%Y-%m-%d %H:%M}  {DIR}"); say("=" * 150)
    if ref is None: say("no reference stock is downloaded yet - nothing audited"); return [], None
    full = int((ref >= 78).sum()); short = int((ref < 78).sum())
    say(f"reference calendar from {', '.join(REF)}: {len(ref):,} trading days {ref.index.min():%Y-%m-%d} .. {ref.index.max():%Y-%m-%d}; {full:,} full sessions (78 bars), {short} shorter sessions "
        f"({', '.join(f'{d:%Y-%m-%d} {int(n)}' for d, n in ref[ref < 78].head(6).items())}{' ...' if short > 6 else ''})")
    syms = list(only) if only else AL.NOW + AL.PAST; redo = []; rows = []; tot_bars = 0; nodata = []
    for s in syms:
        d, bad = load(s); have = sorted(d); badyears = {b[0] for b in bad}
        data_years = [y for y in have if len(d[y])]
        for y, why in bad: redo.append((s, y, why))
        for y in years:
            if y in d or y in badyears: continue
            if y == end_year and (not data_years or data_years[-1] < end_year - 1 or (data_years[-1] == end_year - 1 and d[end_year - 1].time.max() < pd.Timestamp(f'{end_year - 1}-12-15'))): continue
            redo.append((s, y, 'file missing'))
        if not data_years:
            nodata.append(s); continue
        x = pd.concat([d[y] for y in data_years], ignore_index=True); tot_bars += len(x)
        first, last = x.time.min().normalize(), x.time.max().normalize()
        holes = [y for y in range(data_years[0], data_years[-1] + 1) if y in d and len(d[y]) == 0]
        for y in holes: redo.append((s, y, 'empty year between years with data'))
        cnt = x.groupby(x.time.dt.normalize()).size(); span = ref[(ref.index >= first) & (ref.index <= last)]
        absent = span.index.difference(cnt.index); extra = cnt.index.difference(ref.index)
        ratio = (cnt.reindex(span.index).fillna(0) / span); thin = int((ratio < 0.5).sum()) - len(absent)
        dup = int(x.time.duplicated().sum()); m = x.time.dt.hour * 60 + x.time.dt.minute
        bad_px = int(((x[['open', 'high', 'low', 'close']] <= 0).any(axis=1) | (x.low > x[['open', 'close']].min(axis=1) + 1e-9) | (x.high < x[['open', 'close']].max(axis=1) - 1e-9)).sum())
        nan = int(x[['open', 'high', 'low', 'close', 'volume']].isna().any(axis=1).sum()); off = int(((m < 570) | (m >= 960) | (m % 5 != 0)).sum())
        jump = np.abs(np.diff(np.log(x.close.values))); big = int((jump > 0.25).sum())
        for y in sorted(set(pd.DatetimeIndex(absent).year)):
            if (pd.DatetimeIndex(absent).year == y).sum() > 2: redo.append((s, int(y), f"{int((pd.DatetimeIndex(absent).year == y).sum())} trading days absent"))
        ended = last < ref.index.max() - pd.Timedelta(days=7)
        rows.append(dict(sym=s, first=first, last=last, days=len(cnt), absent=len(absent), thin=thin, fill=float(ratio.mean()), dup=dup, bad_px=bad_px, nan=nan, off=off, big=big, ended=ended,
                         extra=len(extra), bars=len(x)))
    R = pd.DataFrame(rows)
    clean = R[(R.absent == 0) & (R.dup == 0) & (R.bad_px == 0) & (R.nan == 0) & (R.off == 0)]
    say(f"{len(R)} stocks with data, {tot_bars:,} bars, {sum(os.path.getsize(f) for f in glob.glob(os.path.join(DIR, '*.parquet'))) / 1e6:,.0f} MB in {len(glob.glob(os.path.join(DIR, '*.parquet')))} files")
    say(f"stocks with every trading day present between their first and last day, no duplicate times, sane prices, no missing values: {len(clean)} of {len(R)}")
    say(f"listed for the whole period (2016-01 to now): {int(((R['first'] <= ref.index.min() + pd.Timedelta(days=5)) & ~R.ended).sum())}; started later (listing): {int((R['first'] > ref.index.min() + pd.Timedelta(days=5)).sum())}; "
        f"ended early (taken over, renamed or delisted): {int(R.ended.sum())}")
    say(f"bars per day against the reference, average over stocks: {R.fill.mean():.1%}; lowest: " + ", ".join(f"{r.sym} {r.fill:.0%}" for r in R.sort_values('fill').head(6).itertuples()))
    prob = R[(R.absent > 0) | (R.dup > 0) | (R.bad_px > 0) | (R.nan > 0) | (R.off > 0)]
    for r in prob.itertuples():
        say(f"   {r.sym:6s} {r.first:%Y-%m-%d} .. {r.last:%Y-%m-%d}: trading days absent {r.absent}, duplicate times {r.dup}, bad prices {r.bad_px}, missing values {r.nan}, bars off the 5-minute grid {r.off}")
    if len(R[R.big > 0]): say("moves above 25% between two consecutive bars (splits not adjusted, or real gaps overnight): " + ", ".join(f"{r.sym} {r.big}" for r in R[R.big > 0].itertuples()))
    if len(R[R.thin > 0]): say("days with fewer than half the usual bars (thin trading or halts): " + ", ".join(f"{r.sym} {r.thin}" for r in R[R.thin > 0].sort_values('thin', ascending=False).head(12).itertuples()))
    if nodata: say(f"no data at all from the source ({len(nodata)}): {', '.join(nodata)}")
    late = R[R['first'] > ref.index.min() + pd.Timedelta(days=5)].sort_values('first'); gone = R[R.ended].sort_values('last')
    if len(late): say("started later: " + ", ".join(f"{r.sym} {r.first:%Y-%m}" for r in late.itertuples()))
    if len(gone): say("ended early: " + ", ".join(f"{r.sym} {r.last:%Y-%m}" for r in gone.itertuples()))
    redo = sorted(set((s, y, w) for s, y, w in redo))
    say(f"stock-years to download again: {len(redo)}" + (": " + "; ".join(f"{s} {y} ({w})" for s, y, w in redo[:25]) + (' ...' if len(redo) > 25 else '') if redo else ''))
    say(f"VERDICT: {'COMPLETE - every stock that has data is whole' if not redo and not len(prob) else 'NOT COMPLETE - see the lines above'}"); say(f"run {time.time() - t0:.0f} s")
    return redo, R


def write():
    with open(OUT, 'w', encoding='utf-8') as fh: fh.write("\n".join(LINES) + "\n")


def repair():
    redo, R = audit(); keep = list(LINES)
    if redo:
        os.makedirs(ASIDE, exist_ok=True)
        for s, y, why in redo:
            f = os.path.join(DIR, f'{s}_{y}.parquet')
            if os.path.exists(f): shutil.move(f, os.path.join(ASIDE, f'{s}_{y}_{int(time.time())}.parquet'))
        AL.fetch(sorted(set(s for s, _, _ in redo)))
        redo2, R = audit()
        if redo2:
            still = sorted(set((s, y) for s, y, _ in redo2) & set((s, y) for s, y, _ in redo))
            say(f"after the second download {len(redo2)} stock-years still fail; {len(still)} of them failed both times = the source itself has the gap")
        LINES[:0] = keep + ['', '----- after downloading the failing stock-years again -----']
    write()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'repair': repair()
    elif len(sys.argv) > 2 and sys.argv[1] == 'only': audit(sys.argv[2:])
    else:
        audit(); write()
