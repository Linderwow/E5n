using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace E5nLib
{
	public static class E5nCalendar
	{
		static DateTime NthWeekday(int y, int m, DayOfWeek wd, int n)
		{
			DateTime d = new DateTime(y, m, 1);
			return d.AddDays(((int)wd - (int)d.DayOfWeek + 7) % 7 + 7 * (n - 1));
		}

		static DateTime LastWeekday(int y, int m, DayOfWeek wd)
		{
			DateTime d = new DateTime(y, m, DateTime.DaysInMonth(y, m));
			return d.AddDays(-(((int)d.DayOfWeek - (int)wd + 7) % 7));
		}

		static DateTime Observed(DateTime d)
		{
			if (d.DayOfWeek == DayOfWeek.Saturday) return d.AddDays(-1);
			if (d.DayOfWeek == DayOfWeek.Sunday) return d.AddDays(1);
			return d;
		}

		static DateTime Easter(int year)
		{
			int a = year % 19, b = year / 100, c = year % 100, d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
			int h = (19 * a + b - d - g + 15) % 30, i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = (a + 11 * h + 22 * l) / 451;
			return new DateTime(year, (h + l - 7 * m + 114) / 31, (h + l - 7 * m + 114) % 31 + 1);
		}

		public static bool IsShortDay(DateTime d)
		{
			d = d.Date; int y = d.Year;
			DateTime tg = NthWeekday(y, 11, DayOfWeek.Thursday, 4);
			if (d == NthWeekday(y, 1, DayOfWeek.Monday, 3) || d == NthWeekday(y, 2, DayOfWeek.Monday, 3) || d == LastWeekday(y, 5, DayOfWeek.Monday)
				|| d == Observed(new DateTime(y, 7, 4)) || d == NthWeekday(y, 9, DayOfWeek.Monday, 1) || d == tg || d == tg.AddDays(1)) return true;
			if (y >= 2022 && d == Observed(new DateTime(y, 6, 19))) return true;
			DateTime j4 = new DateTime(y, 7, 4);
			if (j4.DayOfWeek != DayOfWeek.Saturday && j4.DayOfWeek != DayOfWeek.Sunday && j4.DayOfWeek != DayOfWeek.Monday && d == j4.AddDays(-1)) return true;
			DateTime x = new DateTime(y, 12, 24);
			if (x.DayOfWeek != DayOfWeek.Saturday && x.DayOfWeek != DayOfWeek.Sunday && new DateTime(y, 12, 25).DayOfWeek != DayOfWeek.Saturday && d == x) return true;
			return false;
		}

		public static bool IsClosed(DateTime d)
		{
			d = d.Date; int y = d.Year;
			if (d == Easter(y).AddDays(-2)) return true;
			DateTime x = new DateTime(y, 12, 25);
			DateTime xo = x.DayOfWeek == DayOfWeek.Saturday ? x.AddDays(-1) : x.DayOfWeek == DayOfWeek.Sunday ? x.AddDays(1) : x;
			if (d == xo) return true;
			DateTime ny = new DateTime(y, 1, 1);
			if (ny.DayOfWeek != DayOfWeek.Saturday && ny.DayOfWeek != DayOfWeek.Sunday) return d == ny;
			if (ny.DayOfWeek == DayOfWeek.Sunday) return d == ny.AddDays(1);
			return false;
		}

		public static bool FullSession(DateTime d)
		{
			return d.DayOfWeek != DayOfWeek.Saturday && d.DayOfWeek != DayOfWeek.Sunday && !IsClosed(d) && !IsShortDay(d);
		}

		public static bool MonthEndFlag(DateTime d)
		{
			d = d.Date; int left = 0;
			for (DateTime x = d.AddDays(1); x.Month == d.Month; x = x.AddDays(1))
				if (FullSession(x)) left++;
			return left < 2;
		}

		public static DateTime ThirdFriday(int y, int m)
		{
			DateTime d = new DateTime(y, m, 1);
			return d.AddDays(((int)DayOfWeek.Friday - (int)d.DayOfWeek + 7) % 7 + 14);
		}

		public static double OpexDays(DateTime d)
		{
			d = d.Date;
			int t = (ThirdFriday(d.Year, d.Month) - d).Days;
			if (t >= 0) return t;
			DateTime n = new DateTime(d.Year, d.Month, 1).AddMonths(1);
			return (ThirdFriday(n.Year, n.Month) - d).Days;
		}
	}

	public sealed class E5nModel
	{
		public string Name;
		public DateTime From, To;
		public double Base;
		public int[][] Feat, Left, Right;
		public double[][] Thr, Val;
		public bool[][] MissLeft, Leaf;

		public double Predict(double[] x)
		{
			double s = Base;
			for (int t = 0; t < Leaf.Length; t++)
			{
				bool[] leaf = Leaf[t]; int[] feat = Feat[t], left = Left[t], right = Right[t]; double[] thr = Thr[t]; bool[] ml = MissLeft[t];
				int i = 0;
				while (!leaf[i])
				{
					double v = x[feat[i]];
					i = (double.IsNaN(v) ? ml[i] : v <= thr[i]) ? left[i] : right[i];
				}
				s += Val[t][i];
			}
			return s;
		}
	}

	public sealed class E5nModelSet
	{
		public string[] Features;
		public double Threshold;
		public List<E5nModel> Models = new List<E5nModel>();

		static double Num(string s)
		{
			if (s == "inf") return double.PositiveInfinity;
			if (s == "-inf") return double.NegativeInfinity;
			if (s == "nan") return double.NaN;
			return double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);
		}

		public static E5nModelSet Load(string path)
		{
			E5nModelSet set = new E5nModelSet();
			using (StreamReader r = new StreamReader(path))
			{
				string head = r.ReadLine();
				if (head == null || !head.StartsWith("E5N_MODELS")) throw new InvalidDataException("not an E5n model file: " + path);
				int nf = int.Parse(r.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
				set.Features = new string[nf];
				for (int i = 0; i < nf; i++) set.Features[i] = r.ReadLine().Trim();
				set.Threshold = Num(r.ReadLine().Split(' ')[1]);
				int nm = int.Parse(r.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
				for (int k = 0; k < nm; k++)
				{
					string[] h = r.ReadLine().Split(' ');
					E5nModel m = new E5nModel();
					m.Name = h[1];
					m.From = DateTime.ParseExact(h[2], "yyyy-MM-dd", CultureInfo.InvariantCulture);
					m.To = DateTime.ParseExact(h[3], "yyyy-MM-dd", CultureInfo.InvariantCulture);
					m.Base = Num(h[4]);
					int nt = int.Parse(h[5], CultureInfo.InvariantCulture);
					m.Feat = new int[nt][]; m.Left = new int[nt][]; m.Right = new int[nt][]; m.Thr = new double[nt][]; m.Val = new double[nt][];
					m.MissLeft = new bool[nt][]; m.Leaf = new bool[nt][];
					for (int t = 0; t < nt; t++)
					{
						int nn = int.Parse(r.ReadLine().Split(' ')[1], CultureInfo.InvariantCulture);
						m.Feat[t] = new int[nn]; m.Left[t] = new int[nn]; m.Right[t] = new int[nn]; m.Thr[t] = new double[nn]; m.Val[t] = new double[nn];
						m.MissLeft[t] = new bool[nn]; m.Leaf[t] = new bool[nn];
						for (int i = 0; i < nn; i++)
						{
							string[] p = r.ReadLine().Split(' ');
							if (p[0] == "L") { m.Leaf[t][i] = true; m.Val[t][i] = Num(p[1]); }
							else
							{
								m.Feat[t][i] = int.Parse(p[1], CultureInfo.InvariantCulture); m.Thr[t][i] = Num(p[2]);
								m.Left[t][i] = int.Parse(p[3], CultureInfo.InvariantCulture); m.Right[t][i] = int.Parse(p[4], CultureInfo.InvariantCulture);
								m.MissLeft[t][i] = p[5] == "1";
							}
						}
					}
					set.Models.Add(m);
				}
			}
			return set;
		}

		public E5nModel For(DateTime date)
		{
			date = date.Date;
			for (int i = 0; i < Models.Count; i++)
				if (date >= Models[i].From && date <= Models[i].To) return Models[i];
			return null;
		}
	}

	sealed class E5nRing
	{
		readonly double[] b;
		int n, h;

		public E5nRing(int size) { b = new double[size]; }
		public int Count { get { return n; } }
		public void Add(double x) { b[h] = x; h = (h + 1) % b.Length; if (n < b.Length) n++; }
		public double Last(int i) { int k = h - 1 - i; if (k < 0) k += b.Length; return b[k]; }
		public E5nRing Clone() { E5nRing r = new E5nRing(b.Length); Array.Copy(b, r.b, b.Length); r.n = n; r.h = h; return r; }

		public double Sum(int w) { double s = 0; for (int i = w - 1; i >= 0; i--) s += Last(i); return s; }
		public double Mean(int w) { return Sum(w) / w; }
		public double Max(int w) { double m = Last(0); for (int i = 1; i < w; i++) if (Last(i) > m) m = Last(i); return m; }
		public double Min(int w) { double m = Last(0); for (int i = 1; i < w; i++) if (Last(i) < m) m = Last(i); return m; }

		public double Sd(int w, int ddof)
		{
			double mu = Mean(w), s = 0;
			for (int i = w - 1; i >= 0; i--) { double d = Last(i) - mu; s += d * d; }
			return Math.Sqrt(s / (w - ddof));
		}
	}

	sealed class E5nTf
	{
		public const int NIND = 19;
		public readonly double[] Values = new double[NIND];
		public bool Has;
		int n;
		double prevC, prevH, prevL, prevTp, atr, up, dn, pdmW, ndmW, adx, e12, e26, sig, e20, e50;
		E5nRing cl = new E5nRing(20), hi = new E5nRing(20), lo = new E5nRing(20), tpR = new E5nRing(20), vol = new E5nRing(20), sv = new E5nRing(10),
			pmf = new E5nRing(14), nmf = new E5nRing(14), uv = new E5nRing(20), kk = new E5nRing(3);

		const double A14 = 1.0 / 14, A12 = 2.0 / 13, A26 = 2.0 / 27, A9 = 2.0 / 10, A20 = 2.0 / 21, A50 = 2.0 / 51;

		static double Ewm(double prev, double cur, double alpha)
		{
			if (prev == cur) return prev;
			double ow = 1.0 - alpha;
			return (ow * prev + alpha * cur) / (ow + alpha);
		}

		public E5nTf Clone()
		{
			E5nTf t = (E5nTf)MemberwiseClone();
			t.cl = cl.Clone(); t.hi = hi.Clone(); t.lo = lo.Clone(); t.tpR = tpR.Clone(); t.vol = vol.Clone(); t.sv = sv.Clone();
			t.pmf = pmf.Clone(); t.nmf = nmf.Clone(); t.uv = uv.Clone(); t.kk = kk.Clone();
			return t;
		}

		public void CopyValues(double[] dst, int at)
		{
			for (int i = 0; i < NIND; i++) dst[at + i] = Has ? Values[i] : double.NaN;
		}

		public void Update(double o, double h, double l, double c, double v)
		{
			double tp = (h + l + c) / 3, tr, d, upm, dnm, dtp;
			if (n == 0) { tr = h - l; d = 0; upm = 0; dnm = 0; dtp = 0; }
			else
			{
				tr = Math.Max(h - l, Math.Max(Math.Abs(h - prevC), Math.Abs(l - prevC)));
				d = c - prevC; upm = h - prevH; dnm = prevL - l; dtp = tp - prevTp;
			}
			double pdm = (upm > dnm && upm > 0) ? upm : 0.0, ndm = (dnm > upm && dnm > 0) ? dnm : 0.0;
			double gain = d > 0 ? d : 0.0, loss = d < 0 ? -d : 0.0, m;
			if (n == 0)
			{
				atr = tr; up = gain; dn = loss; pdmW = pdm; ndmW = ndm; e12 = c; e26 = c; e20 = c; e50 = c; m = e12 - e26; sig = m;
			}
			else
			{
				atr = Ewm(atr, tr, A14); up = Ewm(up, gain, A14); dn = Ewm(dn, loss, A14); pdmW = Ewm(pdmW, pdm, A14); ndmW = Ewm(ndmW, ndm, A14);
				e12 = Ewm(e12, c, A12); e26 = Ewm(e26, c, A26); e20 = Ewm(e20, c, A20); e50 = Ewm(e50, c, A50); m = e12 - e26; sig = Ewm(sig, m, A9);
			}
			double pdi = 100 * pdmW / atr, ndi = 100 * ndmW / atr, dx = 100 * Math.Abs(pdi - ndi) / (pdi + ndi);
			if (double.IsNaN(dx)) dx = 0; else if (double.IsPositiveInfinity(dx)) dx = double.MaxValue; else if (double.IsNegativeInfinity(dx)) dx = double.MinValue;
			adx = n == 0 ? dx : Ewm(adx, dx, A14);
			double sgn = d > 0 ? 1.0 : d < 0 ? -1.0 : 0.0;
			cl.Add(c); hi.Add(h); lo.Add(l); tpR.Add(tp); vol.Add(v); sv.Add(sgn * v);
			pmf.Add(dtp > 0 ? tp * v : 0.0); nmf.Add(dtp < 0 ? tp * v : 0.0); uv.Add(c > o ? v : 0.0);
			int cnt = n + 1; double nan = double.NaN;
			Values[0] = dn > 0 ? 100 - 100 / (1 + up / dn) : 100.0;
			Values[1] = (m - sig) / c;
			if (cnt >= 20) { double ma = cl.Mean(20), sd = cl.Sd(20, 0); Values[2] = (c - (ma - 2 * sd)) / (4 * sd); } else Values[2] = nan;
			double k = nan;
			if (cnt >= 14) { double hh = hi.Max(14), ll = lo.Min(14); k = (c - ll) / (hh - ll); Values[5] = (hh - c) / (hh - ll) * -100; } else Values[5] = nan;
			Values[3] = k; kk.Add(k);
			if (kk.Count >= 3)
			{
				double a0 = kk.Last(2), a1 = kk.Last(1), a2 = kk.Last(0);
				bool fin = !(double.IsNaN(a0) || double.IsInfinity(a0) || double.IsNaN(a1) || double.IsInfinity(a1) || double.IsNaN(a2) || double.IsInfinity(a2));
				Values[4] = fin ? (a0 + a1 + a2) / 3 : nan;
			}
			else Values[4] = nan;
			Values[6] = cnt >= 20 ? (tp - tpR.Mean(20)) / (0.015 * 0.7979 * tpR.Sd(20, 0)) : nan;
			Values[7] = adx;
			Values[8] = pdi - ndi;
			Values[9] = atr / c;
			Values[10] = cnt >= 11 ? c / cl.Last(10) - 1 : nan;
			Values[11] = (c - e20) / atr;
			Values[12] = (c - e50) / atr;
			if (cnt >= 20) { double hh = hi.Max(20), ll = lo.Min(20); Values[13] = (c - ll) / (hh - ll); } else Values[13] = nan;
			Values[14] = (c - e20) / (2 * atr);
			double vm = cnt >= 20 ? vol.Mean(20) : nan;
			Values[15] = cnt >= 20 ? (v - vm) / vol.Sd(20, 0) : nan;
			Values[16] = cnt >= 20 ? sv.Sum(10) / (vm * 10) : nan;
			Values[17] = cnt >= 14 ? 100 - 100 / (1 + pmf.Sum(14) / nmf.Sum(14)) : nan;
			Values[18] = cnt >= 20 ? uv.Sum(20) / vol.Sum(20) : nan;
			prevC = c; prevH = h; prevL = l; prevTp = tp; n = cnt; Has = true;
		}
	}

	public sealed class E5nDecision
	{
		public DateTime Date;
		public int Minute;
		public bool HasForecast;
		public double Forecast;
		public int Target;
		public int BaseTarget;
		public double Check;
		public string Model;
		public double[] Features;
	}

	public sealed class E5nEngine
	{
		public static readonly string[] FeatureNames = BuildNames();
		public const int First = 15, Last = 385, Step = 5, Bars = 390;

		readonly E5nModelSet models;
		readonly E5nModelSet check;
		readonly int warmup;
		readonly int[] tfs = { 5, 15, 30 };
		E5nTf[] tf = { new E5nTf(), new E5nTf(), new E5nTf() };
		E5nTf[] snap;
		readonly double[] O = new double[Bars], H = new double[Bars], L = new double[Bars], C = new double[Bars], V = new double[Bars],
			lc = new double[Bars], r1 = new double[Bars], cumTpV = new double[Bars], cumV = new double[Bars], runHi = new double[Bars], runLo = new double[Bars];
		readonly double[] x;
		bool open, same, sameId, hasPrev;
		DateTime sessDate, prevDate;
		long sessKey, prevKey;
		int lastPos, rawCount;
		double prevDh, prevDl, prevDc, prevO0;
		readonly E5nRing dret = new E5nRing(20), dtr = new E5nRing(14);
		readonly double[][] v15Hist = new double[20][];
		int v15Count, v15Head;

		public readonly List<E5nDecision> Replayed = new List<E5nDecision>();
		public bool ReplayGaps;
		public int Sessions { get; private set; }
		public bool SessionOpen { get { return open; } }
		public bool Warm { get { return Sessions >= warmup; } }
		public double Threshold { get { return models.Threshold; } }
		public bool HasCheck { get { return check != null; } }

		static string[] BuildNames()
		{
			string[] b = { "r5", "r15", "r30", "r60", "ropen", "gap", "prevday", "vol30", "vol120", "dvol20", "vwapd", "rngpos", "vratio", "tod", "dow", "r15z", "ropenz" };
			string[] ind = { "rsi", "macd", "bbb", "stk", "std", "wr", "cci", "adx", "did", "atrp", "roc", "e20", "e50", "don", "kel", "vz", "obv", "mfi", "upv" };
			string[] day = { "prng", "pdh", "pdl", "pdc", "orpos", "orbrk", "vwz", "opex", "mend" };
			List<string> f = new List<string>(b);
			foreach (int t in new[] { 5, 15, 30 }) foreach (string i in ind) f.Add(i + t.ToString(CultureInfo.InvariantCulture));
			f.AddRange(day);
			return f.ToArray();
		}

		public E5nEngine(E5nModelSet modelSet, int warmupSessions) : this(modelSet, null, warmupSessions) { }

		public E5nEngine(E5nModelSet modelSet, E5nModelSet checkSet, int warmupSessions)
		{
			models = modelSet; check = checkSet; warmup = warmupSessions;
			if (check != null)
			{
				if (check.Features.Length != FeatureNames.Length) throw new InvalidDataException("check model file has " + check.Features.Length + " features, the engine " + FeatureNames.Length);
				for (int i = 0; i < FeatureNames.Length; i++)
					if (check.Features[i] != FeatureNames[i]) throw new InvalidDataException("feature " + i + " is '" + check.Features[i] + "' in the check model file, '" + FeatureNames[i] + "' in the engine");
			}
			if (models.Features.Length != FeatureNames.Length) throw new InvalidDataException("model file has " + models.Features.Length + " features, the engine " + FeatureNames.Length);
			for (int i = 0; i < FeatureNames.Length; i++)
				if (models.Features[i] != FeatureNames[i]) throw new InvalidDataException("feature " + i + " is '" + models.Features[i] + "' in the model file, '" + FeatureNames[i] + "' in the engine");
			x = new double[FeatureNames.Length];
		}

		public E5nDecision OnMinute(DateTime etEnd, double o, double h, double l, double c, double v, long contractKey)
		{
			DateTime start = etEnd.AddMinutes(-1);
			int m = start.Hour * 60 + start.Minute;
			DateTime date = start.Date;
			if (m < 570 || m >= 960)
			{
				if (open) Rollback();
				return null;
			}
			int pos = m - 570;
			if (!open || date != sessDate)
			{
				if (open) Rollback();
				if (pos != 0 || !E5nCalendar.FullSession(date)) return null;
				Begin(date, contractKey);
			}
			if (pos <= lastPos) return null;
			if (ReplayGaps) Replayed.Clear();
			for (int q = lastPos + 1; q < pos; q++)
			{
				double pc = C[lastPos]; Minute(q, pc, pc, pc, pc, 0);
				if (ReplayGaps && (q + 1) % Step == 0 && q + 1 >= First && q + 1 <= Last)
				{
					E5nDecision g = Decide(q + 1); g.Features = (double[])x.Clone(); Replayed.Add(g);
				}
			}
			Minute(pos, o, h, l, c, v);
			rawCount++;
			if (pos == Bars - 1) { Commit(); return null; }
			int p_ = pos + 1;
			if (p_ % Step != 0 || p_ < First || p_ > Last) return null;
			return Decide(p_);
		}

		void Begin(DateTime date, long key)
		{
			open = true; sessDate = date; sessKey = key; lastPos = -1; rawCount = 0;
			snap = new[] { tf[0].Clone(), tf[1].Clone(), tf[2].Clone() };
			sameId = hasPrev && key == prevKey;
			same = sameId && (date - prevDate).Days <= 5;
		}

		void Rollback()
		{
			open = false;
			if (snap != null) tf = snap;
			snap = null;
		}

		void Minute(int q, double o, double h, double l, double c, double v)
		{
			O[q] = o; H[q] = h; L[q] = l; C[q] = c; V[q] = v;
			lc[q] = Math.Log(c);
			r1[q] = q == 0 ? 0.0 : lc[q] - lc[q - 1];
			double tpv = (h + l + c) / 3 * v;
			cumTpV[q] = q == 0 ? tpv : cumTpV[q - 1] + tpv;
			cumV[q] = q == 0 ? v : cumV[q - 1] + v;
			runHi[q] = q == 0 ? h : Math.Max(runHi[q - 1], h);
			runLo[q] = q == 0 ? l : Math.Min(runLo[q - 1], l);
			lastPos = q;
			for (int i = 0; i < 3; i++)
			{
				int n = tfs[i];
				if ((q + 1) % n != 0) continue;
				int a = q - n + 1; double bh = H[a], bl = L[a], bv = 0;
				for (int j = a; j <= q; j++) { if (H[j] > bh) bh = H[j]; if (L[j] < bl) bl = L[j]; bv += V[j]; }
				tf[i].Update(O[a], bh, bl, C[q], bv);
			}
		}

		void Commit()
		{
			if (rawCount < 201) { Rollback(); return; }
			open = false; snap = null;
			double dh = H[0], dl = L[0];
			for (int j = 1; j < Bars; j++) { if (H[j] > dh) dh = H[j]; if (L[j] < dl) dl = L[j]; }
			double dc = C[Bars - 1];
			dret.Add(Math.Log(dc / O[0]));
			dtr.Add(hasPrev ? Math.Max(dh - dl, Math.Max(Math.Abs(dh - prevDc), Math.Abs(dl - prevDc))) : double.NaN);
			double[] row = v15Hist[v15Head] ?? (v15Hist[v15Head] = new double[Bars]);
			for (int p = 0; p < Bars; p++) row[p] = cumV[p] - (p >= 15 ? cumV[p - 15] : 0.0);
			v15Head = (v15Head + 1) % 20; if (v15Count < 20) v15Count++;
			prevDh = dh; prevDl = dl; prevDc = dc; prevO0 = O[0]; prevDate = sessDate; prevKey = sessKey; hasPrev = true;
			Sessions++;
		}

		static double Std(double[] a, int from, int to, int ddof)
		{
			int n = to - from + 1; double s = 0;
			for (int i = from; i <= to; i++) s += a[i];
			double mu = s / n, q = 0;
			for (int i = from; i <= to; i++) { double d = a[i] - mu; q += d * d; }
			return Math.Sqrt(q / (n - ddof));
		}

		E5nDecision Decide(int p_)
		{
			int p = p_ - 1; double nan = double.NaN; int k = 0;
			double r15 = p - 15 >= 0 ? lc[p] - lc[p - 15] : nan, ropen = lc[p] - Math.Log(O[0]);
			x[k++] = p - 5 >= 0 ? lc[p] - lc[p - 5] : nan;
			x[k++] = r15;
			x[k++] = p - 30 >= 0 ? lc[p] - lc[p - 30] : nan;
			x[k++] = p - 60 >= 0 ? lc[p] - lc[p - 60] : nan;
			x[k++] = ropen;
			x[k++] = same ? Math.Log(O[0] / prevDc) : nan;
			x[k++] = same ? Math.Log(prevDc / prevO0) : nan;
			double vol30 = p >= 10 ? Std(r1, Math.Max(1, p - 29), p, 0) : nan;
			x[k++] = vol30;
			x[k++] = p >= 30 ? Std(r1, Math.Max(1, p - 119), p, 0) : nan;
			x[k++] = dret.Count >= 10 ? dret.Sd(dret.Count, 1) : nan;
			double vw = cumTpV[p] / Math.Max(cumV[p], 1.0);
			x[k++] = lc[p] - Math.Log(vw);
			x[k++] = runHi[p] > runLo[p] ? (C[p] - runLo[p]) / (runHi[p] - runLo[p]) : nan;
			if (p >= 14 && v15Count >= 10)
			{
				double s = 0; int start = (v15Head - v15Count + 20) % 20;
				for (int i = 0; i < v15Count; i++) s += v15Hist[(start + i) % 20][p];
				x[k++] = (cumV[p] - (p >= 15 ? cumV[p - 15] : 0.0)) / (s / v15Count);
			}
			else x[k++] = nan;
			x[k++] = 570 + p_;
			x[k++] = ((int)sessDate.DayOfWeek + 6) % 7;
			x[k++] = r15 / (vol30 * Math.Sqrt(15));
			x[k++] = ropen / (vol30 * Math.Sqrt(p + 1));
			for (int i = 0; i < 3; i++) { tf[i].CopyValues(x, k); k += E5nTf.NIND; }
			double datr = nan; int cnt = 0; double sum = 0;
			for (int i = dtr.Count - 1; i >= 0; i--) { double t = dtr.Last(i); if (!double.IsNaN(t)) { sum += t; cnt++; } }
			if (cnt >= 10) datr = sum / cnt;
			double pH = sameId ? prevDh : nan, pL = sameId ? prevDl : nan, pC = sameId ? prevDc : nan, px = C[p];
			x[k++] = (pH - pL) / datr;
			x[k++] = (px - pH) / datr;
			x[k++] = (px - pL) / datr;
			x[k++] = (px - pC) / datr;
			if (p_ >= 30)
			{
				double orh = H[0], orl = L[0];
				for (int j = 1; j < 30; j++) { if (H[j] > orh) orh = H[j]; if (L[j] < orl) orl = L[j]; }
				x[k++] = (px - orl) / (orh - orl);
				x[k++] = px > orh ? 1.0 : px < orl ? -1.0 : 0.0;
			}
			else { x[k++] = nan; x[k++] = nan; }
			int a = Math.Max(0, p - 29);
			x[k++] = p - a + 1 >= 10 ? (px - vw) / Std(C, a, p, 1) : nan;
			x[k++] = E5nCalendar.OpexDays(sessDate);
			x[k++] = E5nCalendar.MonthEndFlag(sessDate) ? 1.0 : 0.0;

			E5nDecision d = new E5nDecision();
			d.Date = sessDate; d.Minute = p_; d.Features = x;
			E5nModel model = models.For(sessDate);
			d.Check = nan;
			if (model == null || !Warm) { d.HasForecast = false; d.Forecast = nan; d.Model = model == null ? "" : model.Name; return d; }
			d.HasForecast = true; d.Model = model.Name; d.Forecast = model.Predict(x);
			d.BaseTarget = d.Forecast > models.Threshold ? 1 : d.Forecast < -models.Threshold ? -1 : 0;
			d.Target = d.BaseTarget;
			if (check != null)
			{
				E5nModel cm = check.For(sessDate);
				if (cm != null) d.Check = cm.Predict(x);
				if (d.BaseTarget == 0 || double.IsNaN(d.Check) || d.Check * d.BaseTarget <= 0) d.Target = 0;
			}
			return d;
		}
	}

	public sealed class E5nBand
	{
		public const int Look = 14, Gate = 4, Slots = 78;
		public const double Chandelier = 3.0, A20 = 2.0 / 21.0, A50 = 2.0 / 51.0, A200 = 2.0 / 201.0;

		readonly int warmup;
		long bars5;
		double e50, e200, atr, lastClose5;
		bool hasBar;
		DateTime barEnd;
		double bo, bh, bl, bc, bv;
		DateTime day = DateTime.MinValue;
		bool dayOn, dayTrade, hasEma, hasEmaSnap;
		int nb, dir, viol, pendTarget;
		double o0, hi, lo, cumTpV, cumV, ema20, ema20Snap, dayLast, prevClose = double.NaN;
		readonly double[][] hist = new double[Look][];
		int histCount, histHead;
		double[] cur = new double[Slots];
		readonly double[] sg = new double[Slots];
		bool pend, pendEntry;
		double size, refPx, entryPx, ext, points, sides;

		public int Target { get; private set; }
		public int Days { get; private set; }
		public long OvernightBars { get; private set; }
		public bool TradingDay { get { return dayOn && dayTrade; } }
		public bool DayClosed;
		public DateTime ClosedDate;
		public double ClosedPoints, ClosedSides;

		public E5nBand(int warmupDays)
		{
			warmup = Math.Max(Look, warmupDays);
			for (int i = 0; i < Look; i++) hist[i] = new double[Slots];
		}

		public int OnMinute(DateTime etEnd, double o, double h, double l, double c, double v)
		{
			long t = etEnd.Ticks / TimeSpan.TicksPerMinute;
			DateTime end = new DateTime(((t + 4) / 5) * 5 * TimeSpan.TicksPerMinute);
			if (hasBar && end != barEnd) { hasBar = false; OnFive(barEnd, bo, bh, bl, bc, bv); }
			if (!hasBar) { hasBar = true; barEnd = end; bo = o; bh = h; bl = l; bc = c; bv = v; }
			else { if (h > bh) bh = h; if (l < bl) bl = l; bc = c; bv += v; }
			if (t % 5 == 0) { hasBar = false; OnFive(barEnd, bo, bh, bl, bc, bv); }
			return Target;
		}

		public int OnFive(DateTime etEnd, double o, double h, double l, double c, double v)
		{
			double tr = bars5 == 0 ? h - l : Math.Max(h - l, Math.Max(Math.Abs(h - lastClose5), Math.Abs(l - lastClose5)));
			if (bars5 == 0) { e50 = c; e200 = c; atr = tr; }
			else
			{
				if (e50 != c) e50 = ((1.0 - A50) * e50 + A50 * c) / ((1.0 - A50) + A50);
				if (e200 != c) e200 = ((1.0 - A200) * e200 + A200 * c) / ((1.0 - A200) + A200);
				long k = Math.Min(bars5 + 1, 14);
				atr = ((k - 1) * atr + tr) / k;
			}
			bars5++; lastClose5 = c;
			if (pend) { Fill(pendTarget, o, pendEntry); pend = false; }
			int m = etEnd.Hour * 60 + etEnd.Minute;
			bool rth = m > 570 && m <= 960;
			DateTime date = etEnd.Date;
			if (!rth || date != day)
			{
				if (size != 0) Fill(0, o, false);
				if (dayOn && dayTrade && !DayClosed && date != day) Close();
				dir = 0; viol = 0; Target = 0;
			}
			if (!rth) { OvernightBars++; return Target; }
			if (date != day)
			{
				Commit();
				day = date; dayOn = m == 575; nb = 0;
				if (!dayOn) return Target;
				o0 = o; cumTpV = 0; cumV = 0; ema20Snap = ema20; hasEmaSnap = hasEma; points = 0; sides = 0; DayClosed = false;
				for (int i = 0; i < Slots; i++) cur[i] = double.NaN;
				dayTrade = Days >= warmup && !double.IsNaN(prevClose) && !E5nCalendar.IsShortDay(date);
				if (dayTrade)
				{
					hi = Math.Max(o0, prevClose); lo = Math.Min(o0, prevClose);
					for (int j = 0; j < Slots; j++)
					{
						double sum = 0; int cnt = 0;
						for (int q = 0; q < Look; q++)
						{
							double x = hist[(histHead + q) % Look][j];
							if (!double.IsNaN(x)) { sum += x; cnt++; }
						}
						sg[j] = cnt < 7 ? double.NaN : sum / cnt;
					}
				}
			}
			if (!dayOn) return Target;
			int slot = (m - 575) / 5;
			nb++; dayLast = c;
			cur[slot] = Math.Abs(c / o0 - 1.0);
			cumTpV += (h + l + c) / 3 * v; cumV += v;
			double vw = cumTpV / Math.Max(cumV, 1.0);
			ema20 = hasEma ? ema20 + A20 * (c - ema20) : c; hasEma = true;
			if (!dayTrade) return Target;
			if (m >= 960)
			{
				if (size != 0) Fill(0, c, false);
				dir = 0; viol = 0; Target = 0; Close();
				return Target;
			}
			double s = sg[slot];
			bool has = !double.IsNaN(s);
			double ub = hi * (1 + s), lb = lo * (1 - s);
			if (dir != 0)
			{
				ext = dir > 0 ? Math.Max(ext, c) : Math.Min(ext, c);
				if ((dir > 0 && c < ext - Chandelier * atr) || (dir < 0 && c > ext + Chandelier * atr)) { Decide(0); viol = 0; return Target; }
			}
			if (dir != 0 && has)
			{
				bool bad = dir > 0 ? c < Math.Max(ub, vw) : c > Math.Min(lb, vw);
				viol = bad ? viol + 1 : 0;
				if (viol >= Gate) { Decide(0); viol = 0; return Target; }
			}
			if (!has || m < 600 || m > 930 || m % 30 != 0) return Target;
			int nw;
			if (dir == 0)
			{
				nw = c > ub ? 1 : c < lb ? -1 : 0;
				if (m == 720 || m == 750 || m == 780) nw = 0;
			}
			else nw = dir;
			if (nw != 0 && nw != dir && ((nw == 1 && !(c > ema20)) || (nw == -1 && !(c < ema20)))) nw = 0;
			if (nw != 0 && nw != dir && (e50 > e200 ? 1 : e50 < e200 ? -1 : 0) != nw) nw = 0;
			if (nw != dir) Decide(nw);
			return Target;
		}

		void Decide(int target)
		{
			pend = true; pendTarget = target; pendEntry = target != 0 && target != dir;
			dir = target; Target = target;
		}

		void Fill(int target, double px, bool entry)
		{
			points += size * (px - refPx); sides += Math.Abs(target - size); size = target; refPx = px;
			if (entry) { entryPx = px; ext = px; }
		}

		void Close()
		{
			DayClosed = true; ClosedDate = day; ClosedPoints = points; ClosedSides = sides;
		}

		void Commit()
		{
			if (!dayOn) return;
			if (nb >= 30)
			{
				double[] slotRow = hist[histHead];
				hist[histHead] = cur; cur = slotRow; histHead = (histHead + 1) % Look;
				if (histCount < Look) histCount++;
				prevClose = dayLast; Days++;
			}
			else { ema20 = ema20Snap; hasEma = hasEmaSnap; }
			dayOn = false;
		}
	}

	public enum E5nAccountMode { Fixed, Evaluation, Funded }

	public sealed class E5nSizer
	{
		public readonly E5nAccountMode Mode;
		public readonly double Start, Drawdown, LockAbove, Target, PerUnit;
		public readonly int MaxUnits;
		public double High { get; private set; }

		public E5nSizer(E5nAccountMode mode, double start, double drawdown, double lockAbove, double target, double perUnit, int maxUnits, double high)
		{
			Mode = mode; Start = start; Drawdown = drawdown; LockAbove = lockAbove; Target = target; PerUnit = perUnit; MaxUnits = Math.Max(1, maxUnits);
			High = double.IsNaN(high) ? 0.0 : Math.Max(0.0, high);
		}

		public double Threshold { get { return Math.Min(High - Drawdown, LockAbove); } }

		public double Cushion(double balance) { return balance - Start - Threshold; }

		public void Close(double balance) { if (!double.IsNaN(balance) && !double.IsInfinity(balance)) High = Math.Max(High, balance - Start); }

		public int Units(double balance)
		{
			if (Mode != E5nAccountMode.Funded || double.IsNaN(balance) || double.IsInfinity(balance) || PerUnit <= 0) return 1;
			double u = Math.Floor(Cushion(balance) / PerUnit);
			return u < 1 ? 1 : u > MaxUnits ? MaxUnits : (int)u;
		}

		public bool TargetReached(double equity) { return Mode == E5nAccountMode.Evaluation && !double.IsNaN(equity) && equity - Start >= Target; }
	}
}
