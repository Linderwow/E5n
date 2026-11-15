# E5n

A NinjaTrader 8 strategy for the Micro E-mini Nasdaq-100 (MNQ). Every 5 minutes it forecasts the next 5 minutes from 83 indicators
with a gradient-boosted model and holds one position long, short or flat. It is flat at 16:00 ET and never holds overnight.

**Status: experimental.** The model passed its historical tests, but its first weeks on unseen data were negative. Treat it as a
forward-test candidate, run it in the Strategy Analyzer and on a simulation account, and do not assume the backtest will repeat.

## What is here

| Path | What it is |
|---|---|
| `nt8/bin/Custom/AddOns/E5nCore.cs` | The engine: session building, the 83 indicators, the tree-model evaluator, the exchange calendar. Plain C#, no NinjaTrader types. |
| `nt8/bin/Custom/Strategies/E5nStrategy.cs` | The NinjaTrader strategy: orders, guards and its own logs. |
| `nt8/e5n/e5n_models.txt` | 15 frozen models. One per year 2013-2026, each trained only on earlier years, so a backtest of any year is out of sample; `final` (trained to 2026-10-15) serves every later date. |

## How it trades

- Data: one-minute MNQ bars, 09:30-16:00 ET sessions. Half days and exchange holidays are skipped.
- Decision: every 5 minutes from 09:45 to 15:55 ET, at the close of the one-minute bar. Orders fill at the next bar's open.
- Indicators on 5-, 15- and 30-minute bars: RSI, MACD, Bollinger %b, stochastics, Williams %R, CCI, ADX and DI, ATR, rate of change,
  distance to EMA 20 and EMA 50, Donchian and Keltner position, volume z-score, OBV slope, MFI, up-volume share. Plus intraday
  returns and volatility, VWAP distance, opening range, prior-day levels, gap, days to monthly option expiration, month end.
- Rule: long when the forecast is above the threshold, short when it is below the negative threshold, flat in between.
  The threshold is twice a round trip's cost.
- Flat at 16:00 ET every day.

## Install

1. Close any open position of other strategies before compiling.
2. Copy `nt8/bin/Custom/AddOns/E5nCore.cs` to `Documents\NinjaTrader 8\bin\Custom\AddOns\`.
3. Copy `nt8/bin/Custom/Strategies/E5nStrategy.cs` to `Documents\NinjaTrader 8\bin\Custom\Strategies\`.
4. Copy `nt8/e5n/e5n_models.txt` to `Documents\NinjaTrader 8\e5n\` (create the folder).
5. In NinjaTrader open the NinjaScript Editor and press F5.

## Strategy Analyzer

- Strategy `E5nStrategy`, instrument MNQ (current contract), type Minute, value 1, default trading hours.
- Slippage 2 ticks, commission included.
- Parameters: Contracts 1, Warm-up sessions 20, Daily loss stop 0 (off), Model file blank, Undo back-adjustment on, Write decision log on.
- The first 20 complete sessions after the start date are warm-up and take no trades.

NinjaTrader's default merge policy shifts the prices of older contracts. The model was trained on prices as traded, so the strategy
removes that shift with the instrument's rollover offsets ("Undo back-adjustment").

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| Contracts | 1 | Micros per position. |
| Warm-up sessions | 20 | Complete sessions needed in the loaded history before the first trade. |
| Daily loss stop | 0 | Flat for the rest of the day once the day's loss reaches this many dollars. 0 is off, as tested. |
| Model file | blank | Blank uses `Documents\NinjaTrader 8\e5n\e5n_models.txt`. |
| Undo back-adjustment | on | See above. |
| Write decision log | on | Every decision to `Documents\NinjaTrader 8\e5n\logs`. |

## Running live or on simulation

Give the strategy an account of its own. It refuses to start on an account where another strategy is already active.
It writes its own status, decision, fill and event files to `Documents\NinjaTrader 8\e5n\logs`.

## Risk

Futures trading can lose more than the account holds. Nothing here is advice.
