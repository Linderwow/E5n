import numpy as np, pandas as pd

N = (21, 60, 126, 252)
BOOK_P5 = (-1149, -1391, -620, 1964); BOOK_MED = (551, 2165, 5201, 11210)
MODEL_P5 = (-1169, -2096, -2648, -2764); MODEL_MED = (58, 569, 1657, 4057); MODEL_NOEDGE_P95 = (2477, 5863, 13774, 18033)
SHADOW_STOP_AT, SHADOW_STOP = 126, -1000.0; SHADOW_VERDICT_AT = 252
DRY_P90, DRY_MAX = 36, 245


def dry_spell(book):
    eq = np.cumsum(np.asarray(book, float)); peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    hi = np.flatnonzero(eq >= peak); n = int(len(eq) - 1 - hi[-1]) if len(hi) else len(eq)
    if n == 0: return n, "at a new high"
    note = f"{n} session{'s' if n != 1 else ''} without a new high, ${float(peak[-1] - eq[-1]):,.0f} below it"
    if n > DRY_MAX: return n, note + f" - LONGER than any spell in 2013-26 (longest about {DRY_MAX}: April 2025 - February 2026, then new highs)"
    if n > DRY_P90: return n, note + f" - long but seen before (9 in 10 spells end within {DRY_P90}; longest about {DRY_MAX}, April 2025 - February 2026)"
    return n, note + f" - normal (9 in 10 spells end within {DRY_P90} sessions)"


def line(n, pts, center=None, zero_mean=False):
    if n <= 0: return 0.0
    xs = (0,) + N; ys = (0.0,) + tuple(float(p) for p in pts)
    if n >= N[-1]:
        g = n / N[-1]
        if zero_mean: return ys[-1] * np.sqrt(g)
        if center is None: return ys[-1] * g
        c = float(center[-1]); return c * g - (c - ys[-1]) * np.sqrt(g)
    return float(np.interp(n, xs, ys))


def ledger(L):
    if L is None or not len(L): return dict(sessions=0, status='no sessions yet')
    n = len(L); book = float(L.book_usd.astype(float).sum()); model = float(L.model_usd.astype(float).sum())
    r = dict(sessions=n, book_usd=round(book, 2), model_usd=round(model, 2),
             book_p5=round(line(n, BOOK_P5, BOOK_MED)), book_median=round(line(n, BOOK_MED)), model_p5=round(line(n, MODEL_P5, MODEL_MED)), model_median=round(line(n, MODEL_MED)),
             model_noedge_p95=round(line(n, MODEL_NOEDGE_P95, zero_mean=True)))
    r['dry_sessions'], r['dry_spell'] = dry_spell(L.book_usd.astype(float).values)
    flags = []
    if r['dry_sessions'] > DRY_MAX: flags.append(f"BOOK {r['dry_spell']}")
    if book < r['book_p5']: flags.append(f"BOOK BELOW its fair 5% line (${book:+,.0f} vs ${r['book_p5']:+,.0f} at {n} sessions) - worse than 19 of 20 fair outcomes")
    if model < r['model_p5']: flags.append(f"MODEL LEG BELOW its fair 5% line (${model:+,.0f} vs ${r['model_p5']:+,.0f})")
    if model > r['model_noedge_p95']: flags.append(f"model leg above the no-edge 95% line (${model:+,.0f} vs ${r['model_noedge_p95']:+,.0f}) - its edge looks real")
    r['flags'] = flags
    due = [c for c in N[1:] if n >= c]
    r['checkpoint'] = f"{due[-1]}-session checkpoint reached" if due else f"next checkpoint at {[c for c in N[1:] if c > n][0]} sessions"
    r['status'] = ('STOP AND REVIEW' if any(f.startswith(('BOOK BELOW', 'MODEL LEG BELOW')) for f in flags) else
                   'on track' if book >= r['book_median'] else 'inside the fair range, below its median')
    return r


def shadow(sm):
    if not sm or not sm.get('sessions'): return dict(status='no sessions yet')
    n = sm['sessions']; r = dict(sessions=n, sip_lead=sm['sip_lead'], sip_lead_t=sm['sip_lead_t'], iex_lead=sm['iex_lead'])
    if n >= SHADOW_VERDICT_AT:
        ok = sm['sip_lead_t'] >= 2.0 and sm['sip_lead'] > 0 and sm['iex_lead'] >= 0.75 * sm['sip_lead']
        r['status'] = ('LIVE CANDIDATE (free real-time feed is enough)' if ok else
                       ('LIVE CANDIDATE only with the paid feed' if sm['sip_lead_t'] >= 2.0 and sm['sip_lead'] > 0 else 'NOT ADOPTED - no reliable lead after 252 sessions'))
    elif n >= SHADOW_STOP_AT and sm['sip_lead'] < SHADOW_STOP: r['status'] = f"STOPPED - the stock version trails by ${-sm['sip_lead']:,.0f} at {n} sessions"
    else: r['status'] = f"running - verdict at {SHADOW_VERDICT_AT} sessions" + (f", stop check at {SHADOW_STOP_AT}" if n < SHADOW_STOP_AT else '')
    return r
