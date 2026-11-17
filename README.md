# E5n

A NinjaTrader 8 strategy for the Micro E-mini Nasdaq-100 (MNQ). It runs two legs on one-minute bars and holds their sum as one net
position. It is flat at 16:00 ET and never holds overnight.

- **Model leg.** Every 5 minutes it forecasts the next 5 minutes from 83 indicators with a gradient-boosted model and is long, short
  or flat. It trades only when a second model of the next 15 minutes points the same way.
- **Band leg.** It trades breakouts beyond a band built from the typical move at each time of day over the last 14 days, on the side
  of the trend, and exits when price falls back inside.

**Status: experimental.** The rules passed their historical tests, but the model leg's first weeks on unseen data were negative.
Treat it as a forward-test candidate, run it in the Strategy Analyzer and on a simulation account, and do not assume the backtest
will repeat.

## What is here

| Path | What it is |
|---|---|
| `nt8/bin/Custom/AddOns/E5nCore.cs` | The engine: session building, the 83 indicators, the tree-model evaluator, the band leg, the exchange calendar. Plain C#, no NinjaTrader types. |
| `nt8/bin/Custom/Strategies/E5nStrategy.cs` | The NinjaTrader strategy: net-position orders, guards and its own logs. |
| `nt8/e5n/e5n_models.txt` | 15 frozen 5-minute models. One per year 2013-2026, each trained only on earlier years, so a backtest of any year is out of sample; `final` (trained to 2026-10-15) serves every later date. |
| `nt8/e5n/e5n_models_15min.txt` | The same 15 models for the 15-minute check. |

## Python tooling

NinjaTrader runs C# only, so the strategy is C#. Everything around it is Python 3.12 (pandas, numpy, pyarrow).

| Path | What it is |
|---|---|
| `ops/e5n_daily.py` | The nightly job: scores the frozen models and the band leg on the day's one-minute bars, keeps the forward ledger, cross-checks the strategy's own logs, writes the status file and the trade map. `--help` lists the options. |
| `ops/e5n_features.py`, `e5n_model.py`, `e5n_calendar.py`, `e5n_band.py` | Python twins of the engine: the 83 indicators, the tree-model evaluator, the exchange calendar, the band leg. |
| `ops/e5n_account.py` | Python twin of the account modes (evaluation stand-down, funded sizing from the cushion). |
| `ops/e5n_refit.py` | The yearly model refit: retrain, check, stage the new model files. |
| `ops/e5n_health.py`, `e5n_checkpoints.py`, `e5n_flow.py`, `e5n_map.py` | Live checks, forward-test decision lines, order-flow aggregation, the trade map. |
| `ops/e5n_train.py`, `e5n_train15.py` | The original walk-forward model build. |
| `ops/*_test.py` | Standalone test scripts, each prints ALL PASS. |
| `ops/research/` | The research behind every decision, including the variants that were tested and rejected. |

The data these scripts read is not in the repository.

## How it trades

Data: one-minute MNQ bars with the full trading session loaded. Half days and exchange holidays are skipped.

Model leg
- Decision every 5 minutes from 09:45 to 15:55 ET, at the close of the one-minute bar. Orders fill at the next bar's open.
- Indicators on 5-, 15- and 30-minute bars: RSI, MACD, Bollinger %b, stochastics, Williams %R, CCI, ADX and DI, ATR, rate of change,
  distance to EMA 20 and EMA 50, Donchian and Keltner position, volume z-score, OBV slope, MFI, up-volume share. Plus intraday
  returns and volatility, VWAP distance, opening range, prior-day levels, gap, days to monthly option expiration, month end.
- Long when the 5-minute forecast is above the threshold, short when it is below the negative threshold, flat in between.
  The threshold is twice a round trip's cost.
- 15-minute check: a long or short is taken only when the 15-minute forecast has the same sign.

Band leg (5-minute bars built from the one-minute bars)
- Band: the higher of the 09:30 open and the prior close, plus the average absolute move from the open at that time of day over the
  last 14 sessions; and the mirror image below the lower of the two.
- Entries at 10:00, 10:30 ... 15:30 ET except 12:00, 12:30 and 13:00: long on a close above the upper band, short below the lower.
  The close must be on the same side of the EMA 20 of regular-session closes, and the EMA 50 must be on that side of the EMA 200
  (5-minute bars of the full session).
- Exits: 4 five-minute closes in a row back inside the band or the session VWAP, or a close 3 x ATR(14) away from the best close
  since entry. After an exit it may enter again at the next half hour.

Both legs: flat at 16:00 ET. The strategy sends the difference between the wanted net position and the current one.

## Install

1. Close any open position of other strategies before compiling.
2. Copy `nt8/bin/Custom/AddOns/E5nCore.cs` to `Documents\NinjaTrader 8\bin\Custom\AddOns\`.
3. Copy `nt8/bin/Custom/Strategies/E5nStrategy.cs` to `Documents\NinjaTrader 8\bin\Custom\Strategies\`.
4. Copy both files in `nt8/e5n/` to `Documents\NinjaTrader 8\e5n\` (create the folder).
5. In NinjaTrader open the NinjaScript Editor and press F5.

## Strategy Analyzer

- Strategy `E5nStrategy`, instrument MNQ (current contract), type Minute, value 1, default trading hours (the full session).
- Slippage 2 ticks, commission included.
- The defaults are the tested setup: 1 contract a leg, 15-minute check on, warm-up 20 sessions.
- The first 20 complete sessions after the start date are warm-up and take no trades.
- To see one leg alone, set the other leg's contracts to 0.

NinjaTrader's default merge policy shifts the prices of older contracts. The models were trained on prices as traded, so the strategy
removes that shift for the model leg with the instrument's rollover offsets ("Undo back-adjustment"). The band leg uses the
continuous series.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| Contracts | 1 | Micros per leg: the model leg and the band leg each trade this many, as one net position (at most twice this). In Funded mode also times the day's units. |
| Model leg x | 1 | Group "3. Leg sizes (advanced)": the forecast leg trades this many times Contracts. 0 switches the leg off. |
| 15-minute check | on | The forecast leg needs the same sign from the 15-minute model. |
| Band leg x | 1 | Group "3. Leg sizes (advanced)": the band leg trades this many times Contracts. 0 switches the leg off; the evaluation setup uses 6. |
| Warm-up sessions | 20 | Complete sessions needed in the loaded history before the first trade. |
| Daily loss stop | 0 | Flat for the rest of the day once the day's loss reaches this many dollars. 0 is off, as tested. |
| Model file | blank | Blank uses `Documents\NinjaTrader 8\e5n\e5n_models.txt`. |
| Check model file | blank | Blank uses `Documents\NinjaTrader 8\e5n\e5n_models_15min.txt`. |
| Write decision log | on | Every decision to `Documents\NinjaTrader 8\e5n\logs`. |
| Write 1-minute bars | on | The bars the strategy sees, one file a day in `Documents\NinjaTrader 8\e5n\bars` (not in a backtest). |
| Write ticks | on | Every trade with the bid, the ask and their sizes at that moment, one file a day in `Documents\NinjaTrader 8\e5n\ticks` (not in a backtest; written on a background thread, about 60 MB a day). |
| Undo back-adjustment | on | See above. |
| Write order book | on | Every change of the 10-level order book (side, add / update / remove, level, price, size), one file a day in `Documents\NinjaTrader 8\e5n\depth` (not in a backtest; background thread; the nightly job compresses finished days). |

### Account (group "2. Account")

| Parameter | Default | Meaning |
|---|---|---|
| Account mode | Fixed | `Fixed`: the contract settings as they are. `Evaluation`: the contract settings until the account reaches start + target, then flat for good, so a passed evaluation is not given back. `Funded`: the contract settings times a number of units, set once a day at 09:30 ET from the cushion (see below). |
| Account start balance | 50000 | The account's starting balance. |
| Trailing drawdown | 2500 | How far the end-of-day threshold trails the highest end-of-day balance. |
| Drawdown lock above start | 100 | The threshold stops trailing once it reaches start + this amount. |
| Evaluation target | 3000 | Profit target. Evaluation mode only. |
| Funded: cushion per unit | 3000 | Funded mode: units = balance above the threshold divided by this, rounded down, at least 1. 3000 gave the best typical result at realistic fill costs. |
| Funded: max units | 4 | Funded mode: the most units at once. |
| Funded: band leg only until the lock | on | Funded mode only: 1 band-leg micro and no model leg until the trailing-drawdown threshold locks at start + the lock amount, then the contract settings times units. Raised the chance of a payout from 68% to 76% in testing at realistic fill costs. |
| Highest end-of-day balance | 0 | Only for an account that traded before E5n ran on it: its highest end-of-day balance so far. 0 = E5n tracks it from the start (kept in `e5n\logs\E5n_account_*.json`). |

The defaults are a 50K account with a $2,500 end-of-day trailing drawdown that locks at start + $100 and a $3,000 target;
check them against your own account's rules. A unit is the two contract settings together: with 1 + 1 and 3 units the legs
trade 3 + 3 micros. In a backtest the balance is the start balance plus the strategy's own profit. Live, it is the account's
cash value; if NinjaTrader cannot read it, Funded stays at 1 unit and Evaluation does not watch the target (both logged).
Every daily sizing is written to the event log.

## Running live or on simulation

Give the strategy an account of its own. It refuses to start on an account where another strategy is already active.
It writes its own status, decision, fill and event files to `Documents\NinjaTrader 8\e5n\logs`.

## Risk

Futures trading can lose more than the account holds. Nothing here is advice.
