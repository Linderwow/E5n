#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Text;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using E5nLib;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
	public class E5nStrategy : Strategy
	{
		private static readonly CultureInfo CI = CultureInfo.InvariantCulture;
		private E5nEngine engine;
		private E5nBand band;
		private TimeZoneInfo etZone;
		private string blockNote, logDir, barsDir, statusPath, eventsPath, fillsPath, decisionsPath, liveSince = "", haltNote = "", lastModel = "";
		private StreamWriter decisionLog;
		private bool refused, isBacktest, etWarned, halted, undo, addBack, bandOff, barLive, stuckNoted;
		private DateTime dayKey = DateTime.MinValue, lastBarEt = DateTime.MinValue, barDay = DateTime.MinValue;
		private double dayStartRealized, lastForecast = double.NaN, lastCheck = double.NaN;
		private int legModel, legBand, lastNet, lastMinute, decisionsToday, working, workingBar;
		private DateTime[] rollDates = new DateTime[0];
		private double[] rollOffsets = new double[0];
		private readonly List<string> barBuf = new List<string>();
		private const string TickHeader = "time_et,price,size,bid,ask,bid_size,ask_size";
		private string ticksDir;
		private System.Collections.Concurrent.ConcurrentQueue<string> tickQueue;
		private System.Threading.Thread tickThread;
		private volatile bool tickStop;
		private double lastBid = double.NaN, lastAsk = double.NaN;
		private long lastBidSize, lastAskSize, ticksWritten, tickErrors;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Name								= "E5nStrategy";
				Description							= "E5n: two legs on one-minute MNQ bars, traded as one net position. Model leg: a gradient-boosted forecast of the next 5 minutes, confirmed by a 15-minute forecast. Band leg: breakouts beyond the day's volatility band. Flat at 16:00 ET.";
				Calculate							= Calculate.OnBarClose;
				IsUnmanaged							= true;
				IsExitOnSessionCloseStrategy		= true;
				ExitOnSessionCloseSeconds			= 120;
				IsFillLimitOnTouch					= false;
				MaximumBarsLookBack					= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution					= OrderFillResolution.Standard;
				Slippage							= 2;
				StartBehavior						= StartBehavior.WaitUntilFlat;
				TimeInForce							= TimeInForce.Day;
				TraceOrders							= false;
				RealtimeErrorHandling				= RealtimeErrorHandling.StopCancelClose;
				BarsRequiredToTrade					= 0;
				IsInstantiatedOnEachOptimizationIteration = true;

				Contracts		= 1;
				Check15			= true;
				BandContracts	= 1;
				WarmupSessions	= 20;
				MaxDailyLoss	= 0;
				ModelFile		= "";
				CheckModelFile	= "";
				WriteDecisions	= true;
				WriteBars		= true;
				WriteTicks		= true;
				UndoBackAdjustment = true;
			}
			else if (State == State.DataLoaded)
			{
				try   { etZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"); }
				catch { etZone = null; }
				string acct = Account != null ? Account.Name : "account";
				isBacktest = acct == "Backtest";
				string inst = Instrument != null ? Instrument.MasterInstrument.Name : "inst";
				string root = Path.Combine(Core.Globals.UserDataDir, "e5n");
				string model = string.IsNullOrWhiteSpace(ModelFile) ? Path.Combine(root, "e5n_models.txt") : ModelFile.Trim();
				string check = string.IsNullOrWhiteSpace(CheckModelFile) ? Path.Combine(root, "e5n_models_15min.txt") : CheckModelFile.Trim();
				if (BarsPeriod == null || BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value != 1)
					blockNote = string.Format(CI, "the data series is {0} but E5n runs on 1 Minute bars only - set Data Series to Minute, 1 and run again",
						BarsPeriod == null ? "unknown" : BarsPeriod.Value.ToString(CI) + " " + BarsPeriod.BarsPeriodType);
				else if (Contracts == 0 && BandContracts == 0)
					blockNote = "both legs are set to 0 contracts - set 'Model-leg contracts' or 'Band-leg contracts' to 1 or more";
				else if (!File.Exists(model))
					blockNote = "model file not found: " + model + " - install it (see the README) or set 'Model file'";
				else if (Check15 && !File.Exists(check))
					blockNote = "15-minute check model file not found: " + check + " - install it (see the README), set 'Check model file' or switch '15-minute check' off";
				else
				{
					try { engine = new E5nEngine(E5nModelSet.Load(model), Check15 ? E5nModelSet.Load(check) : null, WarmupSessions); }
					catch (Exception ex) { blockNote = "model files could not be loaded (" + model + "): " + ex.Message; }
					if (BandContracts > 0) band = new E5nBand(WarmupSessions);
				}
				if (blockNote == null && !isBacktest)
				{
					string other = OtherStrategy();
					if (other != null)
						blockNote = string.Format(CI, "{0} is already active on account {1} - E5n needs an account of its own (a second simulation account, or its own funded account); enable it there", other, acct);
				}
				try
				{
					MergePolicy mp = Instrument.MasterInstrument.MergePolicy;
					if (mp == MergePolicy.UseGlobalSettings || mp == MergePolicy.UseDefault) mp = Core.Globals.MarketDataOptions.GlobalMergePolicy;
					undo = UndoBackAdjustment && mp == MergePolicy.MergeBackAdjusted;
					addBack = mp == MergePolicy.MergeNonBackAdjusted;
					List<DateTime> rd = new List<DateTime>();
					List<double> ro = new List<double>();
					int loaded = Instrument.Expiry.Year * 12 + Instrument.Expiry.Month;
					foreach (Rollover r in Instrument.MasterInstrument.RolloverCollection)
						if (r.ContractMonth.Year * 12 + r.ContractMonth.Month <= loaded) { rd.Add(r.Date.Date); ro.Add(r.Offset); }
					rollDates = rd.ToArray(); rollOffsets = ro.ToArray();
					Print(string.Format(CI, "E5nStrategy: merge policy {0}, {1} rollovers up to {2:MM-yy}; model leg sees prices {3}, band leg sees {4}", mp, rollDates.Length, Instrument.Expiry,
						undo ? "as traded (back-adjustment removed)" : "as loaded", addBack ? "the continuous series (roll offsets added)" : "the series as loaded"));
				}
				catch (Exception ex) { undo = false; addBack = false; Print("E5nStrategy: rollover table not read (" + ex.Message + ") - prices used as loaded"); }
				try
				{
					logDir = Path.Combine(root, "logs");
					Directory.CreateDirectory(logDir);
					string tag = inst + "_" + acct;
					statusPath = Path.Combine(logDir, "E5n_status_" + tag + ".json");
					eventsPath = Path.Combine(logDir, "E5n_events_" + tag + ".txt");
					fillsPath = Path.Combine(logDir, "E5n_fills_" + tag + ".csv");
					decisionsPath = Path.Combine(logDir, "E5n_decisions_" + tag + ".csv");
					if (WriteDecisions)
					{
						string head = "date,time_et,minute,forecast,base_target,check,target,model,band,net,position,close,adjustment,sessions";
						bool fresh = isBacktest || !File.Exists(decisionsPath);
						if (!fresh)
						{
							string first = "";
							using (StreamReader sr = new StreamReader(new FileStream(decisionsPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))) first = sr.ReadLine() ?? "";
							if (first != head)
							{
								File.Move(decisionsPath, Path.Combine(logDir, "E5n_decisions_" + tag + "_to_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CI) + ".csv"));
								fresh = true;
							}
						}
						decisionLog = new StreamWriter(new FileStream(decisionsPath, isBacktest ? FileMode.Create : FileMode.Append, FileAccess.Write, FileShare.ReadWrite), Encoding.ASCII);
						if (fresh) decisionLog.WriteLine(head);
					}
					if (Instrument != null)
					{
						string bars = Path.Combine(root, "bars");
						Directory.CreateDirectory(bars);
						StringBuilder sb = new StringBuilder("date,offset" + Environment.NewLine);
						for (int i = 0; i < rollDates.Length; i++) sb.AppendLine(string.Format(CI, "{0:yyyy-MM-dd},{1}", rollDates[i], rollOffsets[i]));
						File.WriteAllText(Path.Combine(bars, "rollovers_" + inst + ".csv"), sb.ToString());
						if (WriteBars && !isBacktest)
						{
							barsDir = Path.Combine(bars, inst + "_" + Instrument.Expiry.ToString("MM-yy", CI));
							Directory.CreateDirectory(barsDir);
						}
						if (WriteTicks && !isBacktest)
						{
							ticksDir = Path.Combine(root, "ticks", inst + "_" + Instrument.Expiry.ToString("MM-yy", CI));
							Directory.CreateDirectory(ticksDir);
							tickQueue = new System.Collections.Concurrent.ConcurrentQueue<string>();
							tickThread = new System.Threading.Thread(TickWriter);
							tickThread.IsBackground = true;
							tickThread.Name = "E5nTicks";
							tickThread.Start();
						}
					}
				}
				catch (Exception ex) { Print("E5nStrategy: cannot create log files: " + ex.Message); decisionLog = null; barsDir = null; }
			}
			else if (State == State.Transition)
			{
				working = 0;
				liveSince = ToEt(DateTime.Now).ToString("yyyy-MM-dd HH:mm:ss", CI);
				LogEvent("live", string.Format(CI, "model-leg contracts {0}, 15-minute check {1}, band-leg contracts {2}, warm-up sessions {3}, daily loss stop {4:0}, sessions in history {5}, band days {6}, start {7}, on error {8}",
					Contracts, Check15 ? "on" : "off", BandContracts, WarmupSessions, MaxDailyLoss, engine != null ? engine.Sessions : 0, band != null ? band.Days : 0, StartBehavior, RealtimeErrorHandling));
			}
			else if (State == State.Terminated)
			{
				if (decisionLog != null) { try { decisionLog.Flush(); decisionLog.Dispose(); } catch { } decisionLog = null; }
				if (!barLive) FlushBars();
				if (tickThread != null) { tickStop = true; try { tickThread.Join(3000); } catch { } tickThread = null; }
				if (liveSince != "")
				{
					LogEvent("terminated", string.Format(CI, "position {0} x{1}", Position.MarketPosition, Position.Quantity));
					WriteStatus(refused ? "refused" : "terminated");
				}
			}
		}

		protected override void OnBarUpdate()
		{
			if (BarsInProgress != 0) return;
			if (blockNote != null)
			{
				if (!refused)
				{
					refused = true; haltNote = "not started: " + blockNote;
					Print("E5nStrategy: NOT STARTED - " + blockNote);
					try { Log("E5nStrategy NOT STARTED: " + blockNote, LogLevel.Error); } catch { }
					LogEvent("start_refused", blockNote, true);
					if (State == State.Realtime) WriteStatus("refused");
					SetState(State.Terminated);
				}
				return;
			}
			DateTime et = ToEt(Time[0]);
			lastBarEt = et;
			DateTime start = et.AddMinutes(-1);
			if (start.Date != dayKey)
			{
				dayKey = start.Date; halted = false; haltNote = ""; decisionsToday = 0; legModel = 0;
				dayStartRealized = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
			}
			int cur = Position.MarketPosition == MarketPosition.Long ? Position.Quantity : Position.MarketPosition == MarketPosition.Short ? -Position.Quantity : 0;
			double shift = Adjustment(start.Date);
			double adj = undo ? shift : 0.0, add = addBack ? shift : 0.0;
			E5nDecision d = engine.OnMinute(et, Open[0] - adj, High[0] - adj, Low[0] - adj, Close[0] - adj, Volume[0], ContractKey(start.Date));
			if (d != null)
			{
				decisionsToday++;
				lastMinute = d.Minute; lastForecast = d.Forecast; lastCheck = d.Check; lastModel = d.Model;
				if (d.HasForecast) legModel = d.Target;
			}
			if (band != null)
			{
				legBand = band.OnMinute(et, Open[0] + add, High[0] + add, Low[0] + add, Close[0] + add, Volume[0]);
				if (!bandOff && band.TradingDay && band.OvernightBars == 0)
				{
					bandOff = true;
					string note = "the band leg is OFF: the loaded bars have no overnight session, and it needs the full session (trading hours template CME US Index Futures ETH)";
					Print("E5nStrategy: " + note);
					try { Log("E5nStrategy: " + note, LogLevel.Error); } catch { }
					LogEvent("band_off", note, true);
				}
				if (bandOff) legBand = 0;
			}
			if (MaxDailyLoss > 0 && !halted)
			{
				double dayPnl = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit - dayStartRealized
					+ (cur != 0 ? Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]) : 0.0);
				if (dayPnl <= -MaxDailyLoss)
				{
					halted = true;
					haltNote = string.Format(CI, "daily loss stop hit: {0:0} <= -{1:0} at {2:HH:mm} ET", dayPnl, MaxDailyLoss, et);
					Print("E5nStrategy: " + haltNote);
					LogEvent("halt", haltNote);
				}
			}
			int m = start.Hour * 60 + start.Minute;
			if (m >= 959 || m < 570) legModel = 0;
			int want = halted || m >= 959 || m < 570 ? 0 : legModel * Contracts + legBand * BandContracts;
			lastNet = want;
			if (working > 0 && CurrentBar - workingBar >= 2)
			{
				if (!stuckNoted) { stuckNoted = true; LogEvent("order_pending", string.Format(CI, "{0} order(s) still not final after 2 bars - the position is re-read and orders resume", working), true); }
				working = 0;
			}
			if (want != cur && working == 0) Trade(cur, want);
			if (d != null && decisionLog != null && (isBacktest || State == State.Realtime))
			{
				try
				{
					decisionLog.WriteLine(string.Format(CI, "{0:yyyy-MM-dd},{1:HH:mm},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}", d.Date, et, d.Minute,
						d.HasForecast ? d.Forecast.ToString("R", CI) : "", d.HasForecast ? d.BaseTarget.ToString(CI) : "", d.HasForecast && !double.IsNaN(d.Check) ? d.Check.ToString("R", CI) : "",
						d.HasForecast ? d.Target.ToString(CI) : "", d.Model, band != null ? legBand.ToString(CI) : "", want, cur, (Close[0] - adj).ToString(CI), adj.ToString(CI), engine.Sessions));
					if (State == State.Realtime) decisionLog.Flush();
				}
				catch { }
			}
			if (barsDir != null) LogBar(et, start, Open[0] - adj, High[0] - adj, Low[0] - adj, Close[0] - adj, Volume[0]);
			if (State == State.Realtime && (d != null || m >= 959)) WriteStatus(halted ? "halted" : "running");
		}

		private void Trade(int cur, int want)
		{
			workingBar = CurrentBar;
			if (cur > 0 && want < cur)
			{
				Send(OrderAction.Sell, cur - Math.Max(want, 0), "E5nSell");
				if (want < 0) Send(OrderAction.SellShort, -want, "E5nShort");
			}
			else if (cur < 0 && want > cur)
			{
				Send(OrderAction.BuyToCover, Math.Min(want, 0) - cur, "E5nCover");
				if (want > 0) Send(OrderAction.Buy, want, "E5nLong");
			}
			else if (want > cur) Send(OrderAction.Buy, want - cur, "E5nLong");
			else Send(OrderAction.SellShort, cur - want, "E5nShort");
		}

		private void Send(OrderAction action, int quantity, string name)
		{
			if (quantity <= 0) return;
			working++;
			SubmitOrderUnmanaged(0, action, OrderType.Market, quantity, 0, 0, "", name);
		}

		protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
		{
			if (order == null || order.Name == null || !order.Name.StartsWith("E5n")) return;
			if (orderState == OrderState.Filled || orderState == OrderState.Cancelled || orderState == OrderState.Rejected)
			{
				if (working > 0) working--;
				if (orderState == OrderState.Rejected) LogEvent("order_rejected", string.Format(CI, "{0} {1} x{2}: {3} {4}", order.Name, order.OrderAction, quantity, error, comment), true);
			}
		}

		protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
		{
			if (State != State.Realtime || fillsPath == null || execution == null || execution.Order == null) return;
			try
			{
				if (!File.Exists(fillsPath)) File.WriteAllText(fillsPath, "time_et,order,action,quantity,price,position_after" + Environment.NewLine);
				File.AppendAllText(fillsPath, string.Format(CI, "{0:yyyy-MM-dd HH:mm:ss.fff},{1},{2},{3},{4},{5}{6}", ToEt(time), execution.Order.Name, execution.Order.OrderAction,
					quantity, price, Position.MarketPosition == MarketPosition.Long ? Position.Quantity : Position.MarketPosition == MarketPosition.Short ? -Position.Quantity : 0, Environment.NewLine));
			}
			catch (Exception ex) { Print("E5nStrategy: fill log write failed: " + ex.Message); }
		}

		protected override void OnMarketData(MarketDataEventArgs e)
		{
			if (tickQueue == null || State != State.Realtime || e == null) return;
			if (e.MarketDataType == MarketDataType.Bid) { lastBid = e.Price; lastBidSize = e.Volume; return; }
			if (e.MarketDataType == MarketDataType.Ask) { lastAsk = e.Price; lastAskSize = e.Volume; return; }
			if (e.MarketDataType != MarketDataType.Last) return;
			tickQueue.Enqueue(string.Format(CI, "{0:yyyy-MM-dd HH:mm:ss.fff},{1},{2},{3},{4},{5},{6}", ToEt(e.Time), e.Price, e.Volume, lastBid, lastAsk, lastBidSize, lastAskSize));
		}

		private void TickWriter()
		{
			string day = null; StreamWriter w = null; string line;
			while (true)
			{
				bool any = false;
				try
				{
					while (tickQueue.TryDequeue(out line))
					{
						any = true;
						string d = line.Substring(0, 10);
						if (d != day || w == null)
						{
							if (w != null) { w.Flush(); w.Dispose(); }
							day = d; string path = Path.Combine(ticksDir, d + ".csv"); bool fresh = !File.Exists(path);
							w = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), Encoding.ASCII);
							if (fresh) w.WriteLine(TickHeader);
						}
						w.WriteLine(line); System.Threading.Interlocked.Increment(ref ticksWritten);
					}
					if (any && w != null) w.Flush();
				}
				catch (Exception)
				{
					System.Threading.Interlocked.Increment(ref tickErrors);
					try { if (w != null) w.Dispose(); } catch { }
					w = null;
				}
				if (tickStop && tickQueue.IsEmpty) break;
				System.Threading.Thread.Sleep(250);
			}
			try { if (w != null) { w.Flush(); w.Dispose(); } } catch { }
		}

		private void LogBar(DateTime et, DateTime start, double o, double h, double l, double c, double v)
		{
			for (int i = 0; i < rollDates.Length; i++) if (rollDates[i] > start.Date) return;
			if (start.Date != barDay)
			{
				if (!barLive) FlushBars();
				barDay = start.Date; barBuf.Clear(); barLive = false;
			}
			string line = string.Format(CI, "{0:yyyy-MM-dd HH:mm},{1},{2},{3},{4},{5}", et, o, h, l, c, v);
			if (State != State.Realtime) { barBuf.Add(line); return; }
			try
			{
				if (!barLive) { FlushBars(); barLive = true; }
				string path = Path.Combine(barsDir, barDay.ToString("yyyy-MM-dd", CI) + ".csv");
				if (!File.Exists(path)) File.WriteAllText(path, "time_et,open,high,low,close,volume" + Environment.NewLine);
				File.AppendAllText(path, line + Environment.NewLine);
			}
			catch (Exception ex) { Print("E5nStrategy: bar log write failed: " + ex.Message); }
		}

		private void FlushBars()
		{
			if (barsDir == null || barBuf.Count == 0 || barDay == DateTime.MinValue) return;
			try
			{
				string path = Path.Combine(barsDir, barDay.ToString("yyyy-MM-dd", CI) + ".csv");
				int have = File.Exists(path) ? File.ReadAllLines(path).Length - 1 : -1;
				if (have < barBuf.Count)
				{
					StringBuilder sb = new StringBuilder("time_et,open,high,low,close,volume" + Environment.NewLine);
					for (int i = 0; i < barBuf.Count; i++) sb.AppendLine(barBuf[i]);
					File.WriteAllText(path, sb.ToString());
				}
			}
			catch (Exception ex) { Print("E5nStrategy: bar log write failed: " + ex.Message); }
			barBuf.Clear();
		}

		private string OtherStrategy()
		{
			try
			{
				if (Account == null) return null;
				lock (Account.Strategies)
				{
					foreach (StrategyBase s in Account.Strategies)
						if (!ReferenceEquals(s, this) && (s.State == State.Realtime || s.State == State.Historical || s.State == State.Transition))
							return s.GetType().Name + (s.Instrument != null ? " on " + s.Instrument.FullName : "");
				}
			}
			catch (Exception ex) { Print("E5nStrategy: strategy list read failed: " + ex.Message); }
			return null;
		}

		private long ContractKey(DateTime date)
		{
			long k = 0;
			for (int i = 0; i < rollDates.Length; i++) if (rollDates[i] <= date) k++;
			return k;
		}

		private double Adjustment(DateTime date)
		{
			double s = 0;
			for (int i = 0; i < rollDates.Length; i++) if (rollDates[i] > date) s += rollOffsets[i];
			return s;
		}

		private DateTime ToEt(DateTime t)
		{
			if (etZone == null) return t;
			try { return TimeZoneInfo.ConvertTime(DateTime.SpecifyKind(t, DateTimeKind.Unspecified), Core.Globals.GeneralOptions.TimeZoneInfo, etZone); }
			catch (Exception ex)
			{
				if (!etWarned) { etWarned = true; Print("E5nStrategy: ET conversion failed, using platform time: " + ex.Message); }
				return t;
			}
		}

		private void LogEvent(string evt, string detail, bool force = false)
		{
			if (eventsPath == null || (liveSince == "" && !force) || (isBacktest && !force)) return;
			try
			{
				File.AppendAllText(eventsPath, string.Format(CI, "{0:yyyy-MM-dd HH:mm:ss}|{1}|{2}|{3}{4}", ToEt(DateTime.Now), evt,
					Account != null ? Account.Name : "", (detail ?? "").Replace("\r", " ").Replace("\n", " ").Replace("|", "/"), Environment.NewLine));
			}
			catch (Exception ex) { Print("E5nStrategy: event log write failed: " + ex.Message); }
		}

		private void WriteStatus(string state)
		{
			if (statusPath == null || isBacktest) return;
			try
			{
				bool connected = Account != null && Account.Connection != null && Account.Connection.Status == ConnectionStatus.Connected;
				string json = string.Format(CI,
					"{{\"state\":\"{0}\",\"updated_et\":\"{1:yyyy-MM-dd HH:mm:ss}\",\"last_bar\":\"{2:yyyy-MM-dd HH:mm:ss}\",\"account\":\"{3}\",\"instrument\":\"{4}\",\"position\":\"{5}\",\"quantity\":{6},\"net_target\":{7},\"model_target\":{8},\"band_target\":{9},\"last_minute\":{10},\"last_forecast\":{11},\"last_check\":{12},\"threshold\":{13},\"model\":\"{14}\",\"sessions\":{15},\"warm\":{16},\"band_days\":{17},\"decisions_today\":{18},\"contracts\":{19},\"check15\":{20},\"band_contracts\":{21},\"band_off\":{22},\"max_daily_loss\":{23},\"halt_note\":\"{24}\",\"connected\":{25},\"live_since\":\"{26}\"}}",
					state, ToEt(DateTime.Now), lastBarEt, Account != null ? Account.Name : "", Instrument != null ? Instrument.FullName : "", Position.MarketPosition, Position.Quantity,
					lastNet, legModel, legBand, lastMinute, double.IsNaN(lastForecast) ? "null" : lastForecast.ToString("R", CI), double.IsNaN(lastCheck) ? "null" : lastCheck.ToString("R", CI),
					engine != null ? engine.Threshold.ToString("R", CI) : "null", lastModel, engine != null ? engine.Sessions : 0, engine != null && engine.Warm ? "true" : "false",
					band != null ? band.Days : 0, decisionsToday, Contracts, Check15 ? "true" : "false", BandContracts, bandOff ? "true" : "false", MaxDailyLoss,
					(haltNote ?? "").Replace("\\", "/").Replace("\"", "'"), connected ? "true" : "false", liveSince);
				json = json.Substring(0, json.Length - 1) + string.Format(CI, ",\"ticks_written\":{0},\"tick_errors\":{1},\"tick_queue\":{2}}}", System.Threading.Interlocked.Read(ref ticksWritten), System.Threading.Interlocked.Read(ref tickErrors), tickQueue != null ? tickQueue.Count : 0);
				File.WriteAllText(statusPath, json);
			}
			catch (Exception ex) { Print("E5nStrategy: status write failed: " + ex.Message); }
		}

		#region Properties
		[NinjaScriptProperty] [Range(0, 50)] [Display(Name = "Model-leg contracts", Description = "Micros for the forecast leg (0 = leg off).", GroupName = "1. E5n", Order = 1)] public int Contracts { get; set; }
		[NinjaScriptProperty] [Display(Name = "15-minute check", Description = "The forecast leg trades only when a second model of the next 15 minutes points the same way. On = as tested for the two-leg book.", GroupName = "1. E5n", Order = 2)] public bool Check15 { get; set; }
		[NinjaScriptProperty] [Range(0, 50)] [Display(Name = "Band-leg contracts", Description = "Micros for the band-breakout leg (0 = leg off). Both legs trade as one net position.", GroupName = "1. E5n", Order = 3)] public int BandContracts { get; set; }
		[NinjaScriptProperty] [Range(0, 250)] [Display(Name = "Warm-up sessions", Description = "Complete 09:30-16:00 ET sessions needed in the loaded history before the first trade (20 fills every look-back).", GroupName = "1. E5n", Order = 4)] public int WarmupSessions { get; set; }
		[NinjaScriptProperty] [Range(0, 100000)] [Display(Name = "Daily loss stop ($, 0 = off)", Description = "Flat for the rest of the day once the day's realized plus open loss reaches this amount. 0 = as tested.", GroupName = "1. E5n", Order = 5)] public double MaxDailyLoss { get; set; }
		[NinjaScriptProperty] [Display(Name = "Model file (blank = default)", Description = "Blank uses <NinjaTrader 8 documents folder>\\e5n\\e5n_models.txt.", GroupName = "1. E5n", Order = 6)] public string ModelFile { get; set; }
		[NinjaScriptProperty] [Display(Name = "Check model file (blank = default)", Description = "Blank uses <NinjaTrader 8 documents folder>\\e5n\\e5n_models_15min.txt.", GroupName = "1. E5n", Order = 7)] public string CheckModelFile { get; set; }
		[NinjaScriptProperty] [Display(Name = "Write decision log", Description = "Every 5-minute decision to e5n\\logs (a backtest overwrites its own file).", GroupName = "1. E5n", Order = 8)] public bool WriteDecisions { get; set; }
		[NinjaScriptProperty] [Display(Name = "Write 1-minute bars", Description = "The one-minute bars the strategy sees, one file a day in e5n\\bars, for the nightly forward ledger (not in a backtest).", GroupName = "1. E5n", Order = 9)] public bool WriteBars { get; set; }
		[NinjaScriptProperty] [Display(Name = "Write ticks", Description = "Every trade with the bid, the ask and their sizes at that moment, one file a day in e5n\\ticks, for order-flow research (not in a backtest).", GroupName = "1. E5n", Order = 10)] public bool WriteTicks { get; set; }
		[NinjaScriptProperty] [Display(Name = "Undo back-adjustment", Description = "With merge policy 'Merge back adjusted' NinjaTrader shifts older contracts' prices; the forecast models were trained on prices as traded, so the shift is removed with the instrument's rollover offsets. Leave on.", GroupName = "1. E5n", Order = 11)] public bool UndoBackAdjustment { get; set; }
		#endregion
	}
}
