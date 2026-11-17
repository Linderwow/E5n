import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
MARKET = os.path.join(DATA, 'market')
DATABENTO = os.path.join(MARKET, 'databento_1m')
TRADES = os.path.join(MARKET, 'databento_trades')
BARS = os.path.join(MARKET, 'bars')
NT_MINUTE = os.path.join(MARKET, 'nt_minute')
TAPE = os.path.join(MARKET, 'recorder_tape')
STOCKS = os.path.join(DATA, 'stocks')
STOCK_BARS = os.path.join(STOCKS, 'nasdaq100_5m')
IEX_BARS = os.path.join(STOCKS, 'nasdaq100_iex_5m')
FUND_BARS = os.path.join(STOCKS, 'funds_5m')
QQQ_TRADES = os.path.join(STOCKS, 'qqq_trades_minutes')
WEB = os.path.join(DATA, 'web')
CACHE = os.path.join(DATA, 'research')


def bars_1m(name):
    return os.path.join(BARS, f'{name}_1m.npz')
