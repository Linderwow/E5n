import os, html
import numpy as np, pandas as pd

W, H, ML, MR, MT, MB = 560, 214, 46, 22, 30, 46
CSS = """
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#1c2330;--mute:#6b7585;--grid:#e3e7ee;--price:#27303f;--band:#8a94a6;--vwap:#b7791f;
--long:#1f9d6b;--short:#d1495b;--mlong:#2b6cb0;--mshort:#dd6b20;--flat:#e9edf3}
@media (prefers-color-scheme:dark){:root{--bg:#12161d;--card:#1a2029;--ink:#e6eaf0;--mute:#93a0b4;--grid:#2a3340;--price:#dfe5ee;--band:#7d8aa0;
--vwap:#e0a83c;--long:#3fc48f;--short:#f0707f;--mlong:#63a4ea;--mshort:#f39a52;--flat:#252d39}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
main{max-width:1760px;margin:0 auto;padding:22px 16px 40px}h1{font-size:21px;margin:0 0 4px;font-weight:650}
.sub{color:var(--mute);margin:0 0 14px;max-width:70em}.sum{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 14px}
.kpi{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:9px 14px;min-width:150px}
.kpi b{display:block;font-size:19px;font-variant-numeric:tabular-nums}.kpi span{color:var(--mute);font-size:12px}
.leg{display:flex;flex-wrap:wrap;gap:16px;color:var(--mute);font-size:12.5px;margin:0 0 14px}.leg i{display:inline-block;width:22px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.day{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:8px 8px 4px}
.day h2{font-size:13.5px;margin:0 0 2px 4px;font-weight:600}
.day h2 span{display:block;font-weight:500;font-variant-numeric:tabular-nums;color:var(--mute);margin-top:1px}.pos{color:var(--long)!important}.neg{color:var(--short)!important}
svg{width:100%;height:auto;display:block}.g{stroke:var(--grid);stroke-width:1}.p{fill:none;stroke:var(--price);stroke-width:1.3}
.b{fill:none;stroke:var(--band);stroke-width:1;stroke-dasharray:4 3}.v{fill:none;stroke:var(--vwap);stroke-width:1}
.t{fill:var(--mute);font-size:10.5px}.sl{fill:var(--long)}.ss{fill:var(--short)}.ml{fill:var(--mlong)}.ms{fill:var(--mshort)}.fl{fill:var(--flat)}
.bl{fill:var(--long);opacity:.10}.bs{fill:var(--short);opacity:.10}
.hl{border-radius:10px;padding:10px 14px;margin:0 0 14px;border:1px solid var(--grid);background:var(--card)}.hl b{display:block;margin-bottom:4px}.hl.ok b{color:var(--long)}.hl.bad b{color:var(--short)}.hl.warn b{color:var(--vwap)}.hl ul{margin:4px 0 0 18px;padding:0}
"""


def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def cls(x):
    return 'pos' if x > 0 else ('neg' if x < 0 else '')


def panel(day, g, d, mk, row):
    pw, ph = W - ML - MR, H - MT - MB; x = lambda m: ML + m / 390 * pw
    c = g.c.values; mins = g.pos.values + 1; lo, hi = float(g.l.min()), float(g.h.max()); pad = (hi - lo) * 0.06 or 1.0; lo -= pad; hi += pad
    y = lambda p: MT + (hi - p) / (hi - lo) * ph
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{day:%Y-%m-%d} price and positions">']
    for hh in range(10, 17):
        xx = x(hh * 60 - 570); out.append(f'<line class="g" x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{MT + ph}"/><text class="t" x="{xx:.1f}" y="{H - 4}" text-anchor="middle">{hh}:00</text>')
    for p in np.linspace(lo + pad, hi - pad, 3):
        out.append(f'<line class="g" x1="{ML}" y1="{y(p):.1f}" x2="{W - MR}" y2="{y(p):.1f}"/><text class="t" x="{ML - 5}" y="{y(p) + 3.5:.1f}" text-anchor="end">{p:,.0f}</text>')
    if len(mk):
        bm = (mk.time.dt.hour * 60 + mk.time.dt.minute - 570).values; tg = mk.target.values
        for i in range(len(bm)):
            if tg[i] != 0:
                out.append(f'<rect class="{"bl" if tg[i] > 0 else "bs"}" x="{x(bm[i]):.1f}" y="{MT}" width="{max(x(min(bm[i] + 5, 390)) - x(bm[i]), 0):.1f}" height="{ph}"/>')
        for col, k in (('ub', 'b'), ('lb', 'b'), ('vw', 'v')):
            pts = [(x(m_), y(v_)) for m_, v_ in zip(bm, mk[col].values) if v_ == v_ and lo <= v_ <= hi]
            if len(pts) > 1: out.append(f'<polyline class="{k}" points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts) + '"/>')
    out.append('<polyline class="p" points="' + ' '.join(f'{x(m_):.1f},{y(v_):.1f}' for m_, v_ in zip(mins, c)) + '"/>')
    y1, y2 = MT + ph + 6, MT + ph + 18
    out.append(f'<text class="t" x="{ML - 5}" y="{y1 + 8}" text-anchor="end">model</text><text class="t" x="{ML - 5}" y="{y2 + 8}" text-anchor="end">band</text>')
    out.append(f'<rect class="fl" x="{ML}" y="{y1}" width="{pw}" height="9" rx="2"/><rect class="fl" x="{ML}" y="{y2}" width="{pw}" height="9" rx="2"/>')
    for r in d.itertuples():
        if r.target != 0: out.append(f'<rect class="{"ml" if r.target > 0 else "ms"}" x="{x(r.minute):.1f}" y="{y1}" width="{x(min(r.minute + 5, 390)) - x(r.minute):.1f}" height="9"/>')
    if len(mk):
        for i in range(len(bm)):
            if tg[i] != 0: out.append(f'<rect class="{"sl" if tg[i] > 0 else "ss"}" x="{x(bm[i]):.1f}" y="{y2}" width="{max(x(min(bm[i] + 5, 390)) - x(bm[i]), 0):.1f}" height="9"/>')
    out.append('</svg>')
    head = f'<h2>{day:%a %Y-%m-%d}<span>'
    if row is not None:
        head += f'book <b class="{cls(row.book_usd)}">{money(row.book_usd)}</b> &nbsp; model <b class="{cls(row.model_usd)}">{money(row.model_usd)}</b> &nbsp; band <b class="{cls(row.band_usd)}">{money(row.band_usd)}</b>'
    return f'<section class="day">{head}</span></h2>' + ''.join(out) + '</section>'


def banner(h):
    if not h: return ''
    pr, wa = h.get('problems', []), h.get('warnings', []); sm = f"<p style='margin:6px 0 0'>{html.escape(h['summary'])}</p>" if h.get('summary') else ''
    items = ''.join(f'<li>{html.escape(x)}</li>' for x in pr + wa)
    if pr: return f'<div class="hl bad"><b>Nightly check: {len(pr)} problem(s){" and " + str(len(wa)) + " warning(s)" if wa else ""}</b><ul>{items}</ul>{sm}</div>'
    if wa: return f'<div class="hl warn"><b>Nightly check: {len(wa)} warning(s)</b><ul>{items}</ul>{sm}</div>'
    return f'<div class="hl ok"><b>Nightly check: everything written and matching</b>{sm}</div>'


def write(path, bars, dec, marks, ledger, n=21, health=None):
    led = ledger.copy(); led['date'] = pd.to_datetime(led.date); led = led.sort_values('date'); days = list(led.date.tail(n))
    if not days: return False
    d2 = dec.assign(date=pd.to_datetime(dec.date)); mk = marks.assign(date=marks.time.dt.normalize()) if len(marks) else marks.assign(date=pd.NaT)
    panels = []
    for day in reversed(days):
        g = bars[bars.date == day]
        if g.empty: continue
        row = led[led.date == day].iloc[-1]
        panels.append(panel(day, g, d2[d2.date == day], mk[mk.date == day], row))
    t = led[led.date.isin(days)]; x = t.book_usd.astype(float); eq = x.cumsum(); dd = float((np.maximum(eq.cummax(), 0) - eq).max()) if len(x) else 0.0
    kp = lambda lab, v, c='': f'<div class="kpi"><b class="{c}">{v}</b><span>{html.escape(lab)}</span></div>'
    body = (f'<main><h1>E5n trade map</h1><p class="sub">The last {len(panels)} sessions as the tested rules trade them, 1 micro a leg, after costs. '
            f'Price with the band (dashed) and the session average price; shaded where the band leg holds a position. The two strips show when each leg is long or short. '
            f'Rebuilt by the nightly job; it shows the rules, not the fills of a live account.</p>' + banner(health) + '<div class="sum">'
            + kp('book, these sessions', money(x.sum()), cls(x.sum())) + kp('model leg', money(t.model_usd.astype(float).sum()), cls(t.model_usd.astype(float).sum()))
            + kp('band leg', money(t.band_usd.astype(float).sum()), cls(t.band_usd.astype(float).sum())) + kp('largest drawdown of the book', f'${dd:,.0f}')
            + kp('winning sessions', f'{int((x > 0).sum())} of {len(x)}') + '</div>'
            '<div class="leg"><span><i style="background:var(--mlong)"></i>model leg long</span><span><i style="background:var(--mshort)"></i>model leg short</span>'
            '<span><i style="background:var(--long)"></i>band leg long</span><span><i style="background:var(--short)"></i>band leg short</span>'
            '<span><i style="background:var(--band);height:2px"></i>band</span><span><i style="background:var(--vwap);height:2px"></i>session average price</span></div>'
            '<div class="grid">' + ''.join(panels) + '</div></main>')
    doc = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E5n trade map</title><style>{CSS}</style></head><body>{body}</body></html>'
    with open(path, 'w', encoding='utf-8') as fh: fh.write(doc)
    return True
