#region Using declarations
using System;
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
		private TimeZoneInfo etZone;
		private string blockNote, logDir, statusPath, eventsPath, fillsPath, decisionsPath, liveSince = "", haltNote = "", lastModel = "";
		private StreamWriter decisionLog;
		private bool refused, isBacktest, etWarned, halted;
		private DateTime dayKey = DateTime.MinValue, lastBarEt = DateTime.MinValue;
		private double dayStartRealized, lastForecast = double.NaN;
		private int lastTarget, lastMinute, decisionsToday;
		private bool undo;
		private DateTime[] rollDates = new DateTime[0];
		private double[] rollOffsets = new double[0];

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Name								= "E5nStrategy";
				Description							= "E5n: a gradient-boosted forecast of the next 5 minutes from 83 indicators on one-minute MNQ bars; long, short or flat every 5 minutes, flat at 16:00 ET.";
				Calculate							= Calculate.OnBarClose;
				EntriesPerDirection					= 1;
				EntryHandling						= EntryHandling.AllEntries;
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
				StopTargetHandling					= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade					= 0;
				IsInstantiatedOnEachOptimizationIteration = true;

				Contracts		= 1;
				WarmupSessions	= 20;
				MaxDailyLoss	= 0;
				ModelFile		= "";
				WriteDecisions	= true;
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
				if (BarsPeriod == null || BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value != 1)
					blockNote = string.Format(CI, "the data series is {0} but E5n runs on 1 Minute bars only - set Data Series to Minute, 1 and run again",
						BarsPeriod == null ? "unknown" : BarsPeriod.Value.ToString(CI) + " " + BarsPeriod.BarsPeriodType);
				else if (!File.Exists(model))
					blockNote = "model file not found: " + model + " - install it (see the README) or set 'Model file'";
				else
				{
					try { engine = new E5nEngine(E5nModelSet.Load(model), WarmupSessions); }
					catch (Exception ex) { blockNote = "model file could not be loaded (" + model + "): " + ex.Message; }
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
					System.Collections.Generic.List<DateTime> rd = new System.Collections.Generic.List<DateTime>();
					System.Collections.Generic.List<double> ro = new System.Collections.Generic.List<double>();
					int loaded = Instrument.Expiry.Year * 12 + Instrument.Expiry.Month;
					foreach (Rollover r in Instrument.MasterInstrument.RolloverCollection)
						if (r.ContractMonth.Year * 12 + r.ContractMonth.Month <= loaded) { rd.Add(r.Date.Date); ro.Add(r.Offset); }
					rollDates = rd.ToArray(); rollOffsets = ro.ToArray();
					Print(string.Format(CI, "E5nStrategy: merge policy {0}, {1} rollovers up to {2:MM-yy}; back-adjustment {3}", mp, rollDates.Length, Instrument.Expiry, undo ? "is removed before the model sees prices" : "not applied or left as is"));
				}
				catch (Exception ex) { undo = false; Print("E5nStrategy: rollover table not read (" + ex.Message + ") - prices used as loaded"); }
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
						bool fresh = isBacktest || !File.Exists(decisionsPath);
						decisionLog = new StreamWriter(new FileStream(decisionsPath, isBacktest ? FileMode.Create : FileMode.Append, FileAccess.Write, FileShare.ReadWrite), Encoding.ASCII);
						if (fresh) decisionLog.WriteLine("date,time_et,minute,forecast,target,model,position,close,adjustment,sessions");
					}
				}
				catch (Exception ex) { Print("E5nStrategy: cannot create log files: " + ex.Message); decisionLog = null; }
			}
			else if (State == State.Transition)
			{
				liveSince = ToEt(DateTime.Now).ToString("yyyy-MM-dd HH:mm:ss", CI);
				LogEvent("live", string.Format(CI, "contracts {0}, warm-up sessions {1}, daily loss stop {2:0}, sessions in history {3}, start {4}, on error {5}",
					Contracts, WarmupSessions, MaxDailyLoss, engine != null ? engine.Sessions : 0, StartBehavior, RealtimeErrorHandling));
			}
			else if (State == State.Terminated)
			{
				if (decisionLog != null) { try { decisionLog.Flush(); decisionLog.Dispose(); } catch { } decisionLog = null; }
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
				dayKey = start.Date; halted = false; haltNote = ""; decisionsToday = 0;
				dayStartRealized = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
			}
			int cur = Position.MarketPosition == MarketPosition.Long ? 1 : Position.MarketPosition == MarketPosition.Short ? -1 : 0;
			int want = cur;
			double adj = undo ? Adjustment(start.Date) : 0.0;
			E5nDecision d = engine.OnMinute(et, Open[0] - adj, High[0] - adj, Low[0] - adj, Close[0] - adj, Volume[0], ContractKey(start.Date));
			if (d != null)
			{
				decisionsToday++;
				lastMinute = d.Minute; lastForecast = d.Forecast; lastModel = d.Model;
				if (d.HasForecast) { want = d.Target; lastTarget = d.Target; }
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
			if (halted || m >= 959 || m < 570) want = 0;
			if (want != cur)
			{
				if (want > 0) EnterLong(Contracts, "E5nLong");
				else if (want < 0) EnterShort(Contracts, "E5nShort");
				else if (cur > 0) ExitLong("E5nFlat", "E5nLong");
				else ExitShort("E5nFlat", "E5nShort");
			}
			if (d != null && decisionLog != null && (isBacktest || State == State.Realtime))
			{
				try
				{
					decisionLog.WriteLine(string.Format(CI, "{0:yyyy-MM-dd},{1:HH:mm},{2},{3},{4},{5},{6},{7},{8},{9}", d.Date, et, d.Minute,
						d.HasForecast ? d.Forecast.ToString("R", CI) : "", d.HasForecast ? d.Target.ToString(CI) : "", d.Model, cur, (Close[0] - adj).ToString(CI), adj.ToString(CI), engine.Sessions));
					if (State == State.Realtime) decisionLog.Flush();
				}
				catch { }
			}
			if (State == State.Realtime && (d != null || m >= 959)) WriteStatus(halted ? "halted" : "running");
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
					"{{\"state\":\"{0}\",\"updated_et\":\"{1:yyyy-MM-dd HH:mm:ss}\",\"last_bar\":\"{2:yyyy-MM-dd HH:mm:ss}\",\"account\":\"{3}\",\"instrument\":\"{4}\",\"position\":\"{5}\",\"quantity\":{6},\"target\":{7},\"last_minute\":{8},\"last_forecast\":{9},\"threshold\":{10},\"model\":\"{11}\",\"sessions\":{12},\"warm\":{13},\"decisions_today\":{14},\"contracts\":{15},\"max_daily_loss\":{16},\"halt_note\":\"{17}\",\"connected\":{18},\"live_since\":\"{19}\"}}",
					state, ToEt(DateTime.Now), lastBarEt, Account != null ? Account.Name : "", Instrument != null ? Instrument.FullName : "", Position.MarketPosition, Position.Quantity,
					lastTarget, lastMinute, double.IsNaN(lastForecast) ? "null" : lastForecast.ToString("R", CI), engine != null ? engine.Threshold.ToString("R", CI) : "null", lastModel,
					engine != null ? engine.Sessions : 0, engine != null && engine.Warm ? "true" : "false", decisionsToday, Contracts, MaxDailyLoss,
					(haltNote ?? "").Replace("\\", "/").Replace("\"", "'"), connected ? "true" : "false", liveSince);
				File.WriteAllText(statusPath, json);
			}
			catch (Exception ex) { Print("E5nStrategy: status write failed: " + ex.Message); }
		}

		#region Properties
		[NinjaScriptProperty] [Range(1, 50)] [Display(Name = "Contracts", Description = "Micros per position.", GroupName = "1. E5n", Order = 1)] public int Contracts { get; set; }
		[NinjaScriptProperty] [Range(0, 250)] [Display(Name = "Warm-up sessions", Description = "Complete 09:30-16:00 ET sessions needed in the loaded history before the first trade (20 fills every look-back).", GroupName = "1. E5n", Order = 2)] public int WarmupSessions { get; set; }
		[NinjaScriptProperty] [Range(0, 100000)] [Display(Name = "Daily loss stop ($, 0 = off)", Description = "Flat for the rest of the day once the day's realized plus open loss reaches this amount. 0 = as tested.", GroupName = "1. E5n", Order = 3)] public double MaxDailyLoss { get; set; }
		[NinjaScriptProperty] [Display(Name = "Model file (blank = default)", Description = "Blank uses <NinjaTrader 8 documents folder>\\e5n\\e5n_models.txt.", GroupName = "1. E5n", Order = 4)] public string ModelFile { get; set; }
		[NinjaScriptProperty] [Display(Name = "Undo back-adjustment", Description = "With merge policy 'Merge back adjusted' NinjaTrader shifts older contracts' prices; the model was trained on prices as traded, so the shift is removed with the instrument's rollover offsets. Leave on.", GroupName = "1. E5n", Order = 6)] public bool UndoBackAdjustment { get; set; }
		[NinjaScriptProperty] [Display(Name = "Write decision log", Description = "Every 5-minute decision to e5n\\logs (a backtest overwrites its own file).", GroupName = "1. E5n", Order = 5)] public bool WriteDecisions { get; set; }
		#endregion
	}
}
