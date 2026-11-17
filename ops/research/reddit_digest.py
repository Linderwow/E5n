r"""Condense the 700 Reddit thread files for reading: the whole post + every upvoted comment that talks about rules, tests or results.
    python reddit_digest.py [parts]   -> data\web\reddit\digest\part_<k>.txt (default 8 parts)
"""
import os, re, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import e5n_paths as P

SRC = os.path.join(P.WEB, 'reddit', 'text'); OUT = os.path.join(P.WEB, 'reddit', 'digest')
KEY = re.compile(r"(entry|enter|exit|stop|target|take profit|tp\b|sl\b|backtest|forward test|walk.?forward|out.of.sample|win ?rate|profit factor|sharpe|drawdown|"
                 r"\d+ ?(min|minute|m\b|tick|point|pts|handles)|ema|sma|vwap|atr|rsi|macd|bollinger|opening range|orb|overnight|session|london|new york|"
                 r"edge|expectancy|slippage|commission|statistic|tested|results|rules|algo|strategy|setup|model|ninjatrader|tradingview|python)", re.I)


def condense(text):
    head, _, rest = text.partition('--- COMMENTS (by score) ---')
    blocks = re.split(r"\n(?=\[-?\d+\] )", rest.strip()); keep = []
    for b in blocks:
        m = re.match(r"\[(-?\d+)\] ", b)
        if not m: continue
        sc = int(m.group(1)); hits = len(KEY.findall(b))
        if (sc >= 3 and hits >= 2) or (sc >= 15 and hits >= 1) or hits >= 6: keep.append(b.strip()[:1500])
    return head.strip()[:6000] + "\n--- KEPT COMMENTS ---\n" + "\n\n".join(keep[:40])


def main():
    parts = int(sys.argv[1]) if len(sys.argv) > 1 else 8; os.makedirs(OUT, exist_ok=True)
    fs = sorted(os.listdir(SRC)); docs = [(f, condense(open(os.path.join(SRC, f), encoding='utf-8').read())) for f in fs]
    total = sum(len(d) for _, d in docs); per = total / parts; k = 0; cur = 0; fh = open(os.path.join(OUT, f'part_{k + 1}.txt'), 'w', encoding='utf-8'); sizes = []
    for f, d in docs:
        if cur > per and k < parts - 1:
            fh.close(); sizes.append(cur); k += 1; cur = 0; fh = open(os.path.join(OUT, f'part_{k + 1}.txt'), 'w', encoding='utf-8')
        fh.write(f"\n\n========== THREAD {f[:-4]} ==========\n{d}\n"); cur += len(d)
    fh.close(); sizes.append(cur)
    print(f"{len(docs)} threads, {total / 1e6:.2f} M characters condensed (from {sum(os.path.getsize(os.path.join(SRC, f)) for f in fs) / 1e6:.1f} MB); parts: {', '.join(f'{s / 1e3:.0f}k' for s in sizes)}")


if __name__ == '__main__':
    main()
